import evaluate
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
    enc = tokenizer(sample['context'], sample['question'], return_tensors='pt', truncation=True, max_length=512)
    enc = {k: v.to(model.device) for k,v in enc.items()}
    rec = []
    hooks = attach_gating_hooks(model, rec)
    with torch.no_grad():
        model(**enc)
    for h in hooks: h.remove()

    n_experts = rec[0][1].shape[-1]
    counts    = [np.zeros(n_experts, int) for _ in rec]

    for ex in dataset.select(range(min(max_samples, len(dataset)))):
        enc = tokenizer(sample['context'], sample['question'], return_tensors='pt', truncation=True, max_length=512)
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
    enc = tokenizer(sample['context'], sample['question'], return_tensors='pt', truncation=True, max_length=512)
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
        enc = tokenizer(sample['context'], sample['question'], return_tensors='pt', truncation=True, max_length=512)
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

def evaluate_qa(dataset, tokenizer, model, mask_indices=None, max_samples=10000):
    """
    Evaluate QA performance using SQuAD metrics.
    """
    metric = evaluate.load("squad")
    predictions = []
    references = []
    for ex in tqdm(dataset.select(range(min(max_samples, len(dataset)))), desc='Eval QA'):
        context = ex['context']
        question = ex['question']
        prompt = f"{context}\nQuestion: {question}\nAnswer:"
        if mask_indices is not None:
            hooks = attach_masking_and_record_hooks(model, mask_indices, [])
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=50)
        if mask_indices is not None:
            for hh in hooks: hh.remove()
        pred = tokenizer.decode(out_ids[0][inputs['input_ids'].size(-1):], skip_special_tokens=True).strip()
        predictions.append({'id': ex['id'], 'prediction_text': pred})
        references.append({'id': ex['id'], 'answers': {
            'text': ex['answers']['text'],
            'answer_start': ex['answers']['answer_start']
        }})
    results = metric.compute(predictions=predictions, references=references)
    print(f"Exact Match: {results['exact_match']:.2f}  F1: {results['f1']:.2f}")


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

    # Load SQuAD dataset
    train = load_dataset('squad', split='train')
    test  = load_dataset('squad', split='validation')

    # Compute top-10 experts
    print('Building mask on train...')
    base_counts = analyze_routing_counts(train, tokenizer, model, max_samples=1000)
    mask_indices = get_topk_experts(base_counts, k=10)
    print('Mask indices per layer:', mask_indices)

    # Masked routing & entropy
    print('\nMasked routing & entropy:')
    masked_counts = analyze_routing(
        test, tokenizer, model,
        max_samples=500,
        mask_indices=mask_indices
    )
    print_routing_distribution(masked_counts)
    print('\nMasked eval:')
    evaluate_qa(test, tokenizer, model, mask_indices=mask_indices, max_samples=500)

    # Baseline routing & entropy
    print('\nBaseline routing & entropy:')
    base_counts2 = analyze_routing(test, tokenizer, model, max_samples=500)
    print_routing_distribution(base_counts2)
    print('\nBaseline eval:')
    evaluate_qa(test, tokenizer, model, mask_indices=None, max_samples=500)
