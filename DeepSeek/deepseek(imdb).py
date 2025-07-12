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
    """
    Identify Mixture-of-Experts gating layers by weight shape.
    Returns a list of (name, module).
    """
    modules = []
    for name, module in model.named_modules():
        if hasattr(model.config, 'n_routed_experts') and hasattr(module, 'weight'):
            if module.weight.ndim == 2 and module.weight.shape[0] == model.config.n_routed_experts:
                modules.append((name, module))
    return modules


def attach_gating_hooks(model, records):
    """
    Hook to record baseline gating probabilities.
    """
    handles = []
    for name, module in find_gating_modules(model):
        def hook_fn(module, inputs, output, module_name=name):
            hidden = inputs[0]
            b, s, h = hidden.shape
            flat = hidden.view(-1, h)
            logits = flat @ module.weight.t()
            probs  = torch.softmax(logits, dim=-1)
            records.append((module_name, probs.detach().cpu().float()))
        handles.append(module.register_forward_hook(hook_fn))
    return handles


def attach_masking_and_record_hooks(model, mask_indices, records):
    """
    Hook to mask top-k experts, replace gate output, and record full masked distribution.
    Returns handles.
    """
    handles = []
    for layer_idx, (name, module) in enumerate(find_gating_modules(model)):
        def hook_fn(module, inputs, output, layer_idx=layer_idx, module_name=name):
            # Unpack original gate output
            topk_idx, topk_weight, aux_loss = output
            # Recompute full logits
            hidden = inputs[0]
            b, s, h = hidden.shape
            flat = hidden.view(-1, h)
            logits = flat @ module.weight.t()
            # Mask logits
            for ex in mask_indices[layer_idx]:
                logits[:, ex] = -1e9
            # Full masked distribution
            full_probs = torch.softmax(logits, dim=-1)
            records.append((module_name, full_probs.detach().cpu().float()))
            # Recompute top-k selection
            k = topk_idx.size(-1)
            new_vals, new_idxs = torch.topk(logits, k=k, dim=-1)
            new_wts = full_probs.gather(-1, new_idxs)
            new_idxs = new_idxs.view(b, s, k)
            new_wts  = new_wts.view(b, s, k)
            return new_idxs, new_wts, aux_loss
        handles.append(module.register_forward_hook(hook_fn))
    return handles


def print_gating_entropy(records):
    """
    Print average entropy for each layer's gating distribution.
    """
    print("\n=== Gating Entropy per Layer ===")
    for i, (name, scores) in enumerate(records):
        probs = scores.numpy()
        ent = -(probs * np.log(probs + 1e-12)).sum(axis=-1)
        print(f"Layer {i} ({name}): avg entropy = {ent.mean():.4f}")


def print_routing_distribution(counts):
    """
    Print counts, entropy, and normalized entropy per layer.
    """
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
    """Select top-k experts per layer based on usage counts."""
    return [list(np.argsort(c)[::-1][:k]) for c in counts]


def analyze_routing_counts(dataset, tokenizer, model, max_samples=10000):
    """
    Compute expert usage counts over dataset subset.
    """
    # Warm up
    sample = dataset[0]
    enc = tokenizer(sample['text'], return_tensors='pt', truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    rec = []
    h = attach_gating_hooks(model, rec)
    with torch.no_grad(): model(**enc)
    for hh in h: hh.remove()

    n_layers  = len(rec)
    n_experts = rec[0][1].shape[-1]
    counts    = [np.zeros(n_experts, int) for _ in range(n_layers)]

    for ex in dataset.select(range(min(max_samples, len(dataset)))):
        enc = tokenizer(ex['text'], return_tensors='pt', truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        rec2 = []
        h2   = attach_gating_hooks(model, rec2)
        with torch.no_grad(): model(**enc)
        for hh in h2: hh.remove()
        for li, (_, scores) in enumerate(rec2):
            topk = getattr(model.config, 'num_experts_per_tok', 1)
            idxs = torch.topk(scores, k=topk, dim=-1).indices.view(-1).cpu().numpy()
            for i in idxs: counts[li][i] += 1
    return counts


def analyze_routing(dataset, tokenizer, model, max_samples=10000, mask_indices=None):
    """
    Print gating entropy and return routing counts, masked or baseline.
    """
    masked = mask_indices is not None
    sample = dataset[0]
    enc = tokenizer(sample['text'], return_tensors='pt', truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    rec = []
    if masked:
        h = attach_masking_and_record_hooks(model, mask_indices, rec)
    else:
        h = attach_gating_hooks(model, rec)
    with torch.no_grad(): model(**enc)
    for hh in h: hh.remove()
    print_gating_entropy(rec)

    counts = [np.zeros(rec[0][1].shape[-1], int) for _ in rec]
    for ex in tqdm(dataset.select(range(min(max_samples, len(dataset)))), desc=f"Routing ({'masked' if masked else 'base'})"):
        enc = tokenizer(ex['text'], return_tensors='pt', truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        rec2 = []
        if masked:
            h2 = attach_masking_and_record_hooks(model, mask_indices, rec2)
        else:
            h2 = attach_gating_hooks(model, rec2)
        with torch.no_grad(): model(**enc)
        for hh in h2: hh.remove()
        for li, (_, scores) in enumerate(rec2):
            topk = getattr(model.config, 'num_experts_per_tok', 1)
            idxs = torch.topk(scores, k=topk, dim=-1).indices.view(-1).cpu().numpy()
            for i in idxs: counts[li][i] += 1
    return counts


def evaluate_classification(dataset, tokenizer, model, mask_indices=None, max_samples=10000):
    label_map = {0: 'negative', 1: 'positive'}
    y_true, y_pred = [], []

    # Pre‐tokenize the two labels once
    label_token_ids = {
        lbl: tokenizer(lbl, add_special_tokens=False)['input_ids']
        for lbl in label_map.values()
    }

    for ex in tqdm(dataset.select(range(min(max_samples, len(dataset)))), desc='Eval'):
        y_true.append(ex['label'])
        prompt = f"{ex['text']}\nSentiment: "

        scores = {}
        for lbl, tok_ids in label_token_ids.items():
            # Tokenize prompt and immediately move to model.device
            enc_prompt = tokenizer(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512
            )
            prompt_ids = enc_prompt['input_ids'].to(model.device)
            prompt_mask = enc_prompt['attention_mask'].to(model.device)

            # Build full input_ids and attention_mask
            label_ids = torch.tensor(tok_ids, device=model.device).unsqueeze(0)
            input_ids = torch.cat([prompt_ids, label_ids], dim=1)
            attention_mask = torch.cat([
                prompt_mask,
                torch.ones_like(label_ids)
            ], dim=1)

            # Prepare labels: -100 for prompt, actual for label
            labels = torch.full_like(input_ids, -100)
            labels[:, prompt_ids.shape[1]:] = input_ids[:, prompt_ids.shape[1]:]

            # Attach mask hooks if needed
            records = []
            if mask_indices is not None:
                hooks = attach_masking_and_record_hooks(model, mask_indices, records)

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
            if mask_indices is not None:
                for h in hooks: h.remove()

            # sum log-prob = - loss * n_label_tokens
            n_tok = label_ids.shape[1]
            scores[lbl] = -out.loss.item() * n_tok

        pred = max(scores, key=scores.get)
        y_pred.append(1 if pred == 'positive' else 0)

    acc = 100.0 * np.mean(np.array(y_true) == np.array(y_pred))
    print(f"Accuracy = {acc:.2f}%")
    return acc



if __name__ == '__main__':
    MODEL = 'deepseek-ai/deepseek-moe-16b-base'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map='auto', trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.eval()

    # Balanced IMDB train subset
    train = load_dataset('imdb', split='train')
    pos = train.filter(lambda x: x['label']==1).shuffle(seed=1).select(range(500))
    neg = train.filter(lambda x: x['label']==0).shuffle(seed=1).select(range(500))
    train_sub = concatenate_datasets([pos, neg]).shuffle(seed=1)

    # Compute top-10 experts
    print('Building mask on train...')
    base_counts = analyze_routing_counts(train_sub, tokenizer, model, max_samples=10000)
    mask_indices = get_topk_experts(base_counts, k=10)
    print('Mask indices per layer:', mask_indices)

    # Balanced IMDB test subset
    test = load_dataset('imdb', split='test')
    pos_t = test.filter(lambda x: x['label']==1).shuffle(seed=1).select(range(250))
    neg_t = test.filter(lambda x: x['label']==0).shuffle(seed=1).select(range(250))
    test_sub = concatenate_datasets([pos_t, neg_t]).shuffle(seed=1)

    # Masked routing & entropy
    print('\nMasked routing & entropy:')
    masked_counts = analyze_routing(test_sub, tokenizer, model, max_samples=10000, mask_indices=mask_indices)
    print_routing_distribution(masked_counts)
    print('\nMasked eval:')
    evaluate_classification(test_sub, tokenizer, model, mask_indices=mask_indices, max_samples=10000)

    # Baseline routing & entropy
    print('\nBaseline routing & entropy:')
    base_counts2 = analyze_routing(test_sub, tokenizer, model, max_samples=10000)
    print_routing_distribution(base_counts2)
    print('\nBaseline eval:')
    evaluate_classification(test_sub, tokenizer, model, mask_indices=None, max_samples=10000)
