import torch
import torch.nn.functional as F
import numpy as np
import nltk
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm

# Download NLTK resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# ---------------------------
# Helper functions
# ---------------------------

def find_gating_modules(model):
    config = model.config
    n_experts = getattr(config, 'num_local_experts', None)
    if n_experts is None:
        raise ValueError("Model config has no 'num_local_experts' attribute.")
    modules = []
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and module.weight.ndim == 2:
            if module.weight.shape[0] == n_experts:
                modules.append((name, module))
    if not modules:
        print(f"Warning: No gating modules found for n_experts={n_experts}.")
    return modules

def attach_gating_hooks(model, records):
    handles = []
    for name, module in find_gating_modules(model):
        def hook_fn(module, inputs, output, module_name=name):
            flat = output.view(-1, output.shape[-1])
            probs = torch.softmax(flat, dim=-1)
            records.append((module_name, probs.detach().cpu()))
        handles.append(module.register_forward_hook(hook_fn))
    return handles

def attach_masking_and_record_hooks(model, mask_indices, records):
    handles = []
    for layer_idx, (name, module) in enumerate(find_gating_modules(model)):
        this_mask = mask_indices[layer_idx]
        def hook_fn(module, inputs, output,
                    module_name=name,
                    mask_idx=this_mask):
            orig = output.shape
            flat = output.view(-1, orig[-1])
            for ex in mask_idx:
                flat[:, ex] = float('-inf')
            probs = torch.softmax(flat, dim=-1)
            records.append((module_name, probs.detach().cpu()))
            output.copy_(flat.view(orig))
        handles.append(module.register_forward_hook(hook_fn))
    return handles

def print_gating_entropy(records):
    print("\n=== Gating Entropy per Layer ===")
    for i, (name, scores) in enumerate(records):
        # CAST to float32 before converting to numpy
        p = scores.to(torch.float32).numpy()
        ent = -(p * np.log(p + 1e-12)).sum(axis=-1)
        print(f"Layer {i} ({name}): avg entropy = {ent.mean():.4f}")

def print_routing_distribution(counts):
    print("\n=== Routing Distribution & Entropy ===")
    for i, cnt in enumerate(counts):
        total = cnt.sum()
        if total == 0:
            print(f"Layer {i}: no records.")
            continue
        probs = cnt / total
        ent = -(probs * np.log(probs + 1e-12)).sum()
        norm = ent / np.log(len(cnt))
        print(f"Layer {i}: counts={cnt.tolist()} entropy={ent:.3f} (norm={norm:.3f})")

def get_topk_experts(counts, k=10):
    return [list(np.argsort(c)[::-1][:k]) for c in counts]

def analyze_routing_counts(dataset, tokenizer, model, max_samples=10000):
    # Warm-up on first token
    sample = dataset[0]
    enc = tokenizer(sample['text'], return_tensors='pt', truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k,v in enc.items()}
    rec = []
    hooks = attach_gating_hooks(model, rec)
    with torch.no_grad():
        model(**enc)
    for h in hooks: h.remove()

    n_experts = rec[0][1].shape[-1]
    counts    = [np.zeros(n_experts, int) for _ in rec]

    for ex in dataset.select(range(min(max_samples, len(dataset)))):
        enc = tokenizer(ex['text'], return_tensors='pt', truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k,v in enc.items()}
        rec2 = []
        hooks2 = attach_gating_hooks(model, rec2)
        with torch.no_grad():
            model(**enc)
        for h in hooks2: h.remove()
        for li, (_, scores) in enumerate(rec2):
            topk = getattr(model.config, 'num_experts_per_tok', 1)
            idxs = torch.topk(scores, k=topk, dim=-1).indices.view(-1).cpu().numpy()
            for i in idxs:
                counts[li][i] += 1

    return counts

def analyze_routing(dataset, tokenizer, model, max_samples=10000, mask_indices=None):
    masked = mask_indices is not None

    # 1) Entropy pass
    sample = dataset[0]
    enc = tokenizer(sample['text'], return_tensors='pt', truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k,v in enc.items()}
    rec = []
    if masked:
        hooks = attach_masking_and_record_hooks(model, mask_indices, rec)
    else:
        hooks = attach_gating_hooks(model, rec)

    with torch.no_grad():
        model(**enc)
    for h in hooks: h.remove()

    print(f"\n--- {'Masked' if masked else 'Baseline'} Gating Entropy ---")
    print_gating_entropy(rec)

    # 2) Routing counts
    counts = [np.zeros(rec[0][1].shape[-1], int) for _ in rec]
    desc = f"Routing counts ({'masked' if masked else 'baseline'})"
    for ex in tqdm(dataset.select(range(min(max_samples, len(dataset)))), desc=desc):
        enc = tokenizer(ex['text'], return_tensors='pt', truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k,v in enc.items()}
        rec2 = []
        if masked:
            hooks2 = attach_masking_and_record_hooks(model, mask_indices, rec2)
        else:
            hooks2 = attach_gating_hooks(model, rec2)

        with torch.no_grad():
            model(**enc)
        for h in hooks2: h.remove()

        for li, (_, scores) in enumerate(rec2):
            topk = getattr(model.config, 'num_experts_per_tok', 1)
            idxs = torch.topk(scores, k=topk, dim=-1).indices.view(-1).cpu().numpy()
            for i in idxs:
                counts[li][i] += 1

    print(f"\n--- {'Masked' if masked else 'Baseline'} Routing Distribution ---")
    print_routing_distribution(counts)
    return counts

def evaluate_classification(dataset, tokenizer, model, mask_indices=None, max_samples=10000):
    label_map = {0: 'negative', 1: 'positive'}
    y_true, y_pred = [], []

    # Pre‐tokenize label tokens once
    label_token_ids = {
        lbl: tokenizer(lbl, add_special_tokens=False)['input_ids']
        for lbl in label_map.values()
    }

    for ex in tqdm(dataset.select(range(min(max_samples, len(dataset)))), desc='Eval'):
        y_true.append(ex['label'])
        prompt = f"{ex['text']}\nSentiment: "

        scores = {}
        for lbl, tok_ids in label_token_ids.items():
            enc_prompt = tokenizer(prompt, return_tensors='pt',
                                   truncation=True, max_length=512)
            prompt_ids  = enc_prompt['input_ids'].to(model.device)
            prompt_mask = enc_prompt['attention_mask'].to(model.device)

            label_ids = torch.tensor(tok_ids, device=model.device).unsqueeze(0)
            input_ids = torch.cat([prompt_ids, label_ids], dim=1)
            attention_mask = torch.cat([prompt_mask,
                                        torch.ones_like(label_ids)], dim=1)

            labels = torch.full_like(input_ids, -100)
            labels[:, prompt_ids.shape[1]:] = input_ids[:, prompt_ids.shape[1]:]

            if mask_indices is not None:
                rec = []
                hooks = attach_masking_and_record_hooks(model, mask_indices, rec)

            with torch.no_grad():
                out = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            if mask_indices is not None:
                for h in hooks: h.remove()

            n_tok = label_ids.shape[1]
            scores[lbl] = -out.loss.item() * n_tok

        pred = max(scores, key=scores.get)
        y_pred.append(1 if pred=='positive' else 0)

    acc = 100*np.mean(np.array(y_true)==np.array(y_pred))
    print(f"\nAccuracy = {acc:.2f}%")
    return acc

if __name__ == '__main__':
    MODEL = 'ibm-research/PowerMoE-3b'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    model.eval()

    # Build a balanced train subset
    train = load_dataset('imdb', split='train')
    pos = train.filter(lambda x: x['label']==1).shuffle(seed=1).select(range(1))
    neg = train.filter(lambda x: x['label']==0).shuffle(seed=1).select(range(1))
    train_sub = concatenate_datasets([pos, neg]).shuffle(seed=1)

    print('Building mask on train…')
    base_counts = analyze_routing_counts(train_sub, tokenizer, model, max_samples=1000)
    mask_indices = get_topk_experts(base_counts, k=10)
    print('Mask indices per layer:', mask_indices)

    # **Routing analysis**
    analyze_routing(train_sub, tokenizer, model, max_samples=500, mask_indices=None)
    analyze_routing(train_sub, tokenizer, model, max_samples=500, mask_indices=mask_indices)

    # Build a balanced test subset
    test = load_dataset('imdb', split='test')
    pos_t = test.filter(lambda x: x['label']==1).shuffle(seed=1).select(range(1))
    neg_t = test.filter(lambda x: x['label']==0).shuffle(seed=1).select(range(1))
    test_sub = concatenate_datasets([pos_t, neg_t]).shuffle(seed=1)

    # Masked eval
    print('\nMasked eval:')
    evaluate_classification(test_sub, tokenizer, model,
                            mask_indices=mask_indices, max_samples=500)

    # Baseline eval
    print('\nBaseline eval:')
    evaluate_classification(test_sub, tokenizer, model,
                            mask_indices=None, max_samples=500)
