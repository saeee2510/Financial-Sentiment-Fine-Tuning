"""
Fine-tunes a pretrained transformer (default: distilbert-base-uncased) for 3-class
sentiment with a plain PyTorch training loop: AdamW, linear warmup/decay, and
selection of the epoch with the best validation macro-F1.

Uses Apple Silicon (MPS) or a GPU when available, otherwise CPU.
"""
import random

import numpy as np
from sklearn.metrics import f1_score


def pick_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def finetune(train_df, val_df, model_name="distilbert-base-uncased", epochs=2, batch_size=16,
             lr=3e-5, max_len=64, seed=42, save_dir="output/model"):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = pick_device(torch)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3).to(device)

    def encode(df):
        enc = tok(list(df["text"]), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
        return TensorDataset(enc["input_ids"], enc["attention_mask"], torch.tensor(df["label"].to_numpy()))

    train_dl = DataLoader(encode(train_df), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(encode(val_df), batch_size=64)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = len(train_dl) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best_f1, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for ids, mask, y in train_dl:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            loss = model(input_ids=ids, attention_mask=mask, labels=y).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            total += loss.item()
        preds, _ = _predict_loader(model, val_dl, device, torch)
        f1 = f1_score(val_df["label"], preds, average="macro")
        history.append({"epoch": epoch, "train_loss": total / len(train_dl), "val_macro_f1": f1})
        print(f"epoch {epoch}: train loss {total / len(train_dl):.4f}, val macro-F1 {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            model.save_pretrained(save_dir); tok.save_pretrained(save_dir)

    best = AutoModelForSequenceClassification.from_pretrained(save_dir).to(device)
    return best, tok, device, history


def _predict_loader(model, loader, device, torch):
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for batch in loader:
            ids, mask = batch[0].to(device), batch[1].to(device)
            probs = torch.softmax(model(input_ids=ids, attention_mask=mask).logits, dim=-1)
            conf, pred = probs.max(dim=-1)
            preds.extend(pred.cpu().tolist()); confs.extend(conf.cpu().tolist())
    return np.array(preds), np.array(confs)


def predict(model, tok, device, texts, max_len=64):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    enc = tok(list(texts), truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")
    loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"]), batch_size=64)
    return _predict_loader(model, loader, device, torch)
