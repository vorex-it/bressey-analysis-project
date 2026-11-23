"""Treino/avaliação dos modelos CTC (CRNN, HTR-Flor, HTR-VT) no BRESSAY.

Exemplo:
    python -m src.train_ctc --model flor --data /content/bressay --out /content/drive/MyDrive/htr_runs --epochs 40
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CTCLines, Charset, collate_ctc, load_manifest
from .models import SAM, build, n_params
from .metrics import cer, full_report
from .device import amp_type, get_device


def greedy(logits, cs):
    ids = logits.argmax(-1).transpose(0, 1).cpu().numpy()  # B, T
    return [cs.decode_ctc(row) for row in ids]


@torch.no_grad()
def predict(model, loader, cs, device):
    model.eval()
    preds, ids = [], []
    for x, _, _, _, bid in loader:
        with torch.autocast(device_type=amp_type(device), dtype=torch.float16, enabled=device.type == "cuda"):
            lo = model(x.to(device, non_blocking=True))
        preds += greedy(lo.float(), cs); ids += bid
    return dict(zip(ids, preds))


@torch.no_grad()
def cpu_latency(model, ds, n=100):
    """Latência média por linha (lote 1) em CPU, 1 thread por padrão do ambiente."""
    m = model.to("cpu").eval()
    xs = [ds[i][0][None] for i in range(min(n, len(ds)))]
    for x in xs[:5]:
        m(x)
    t = time.perf_counter()
    for x in xs:
        m(x)
    return (time.perf_counter() - t) / len(xs) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["crnn", "flor", "htrvt"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--frac", type=float, default=1.0, help="fração das páginas de treino (curva de aprendizado)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_train", type=int, default=0, help="limita nº de linhas (teste rápido)")
    ap.add_argument("--max_eval", type=int, default=0)
    ap.add_argument("--no_sam", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cuda, mps ou cpu (padrão: automático)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = get_device(a.device)
    run = f"{a.model}_frac{a.frac:g}"
    out = os.path.join(a.out, run); os.makedirs(out, exist_ok=True)

    tr, va, te = load_manifest(a.data, frac=a.frac, seed=a.seed)
    # charset sempre do treino COMPLETO, para que todas as frações tenham a mesma saída
    cs = Charset(load_manifest(a.data)[0].text)
    if a.max_train: tr = tr.sample(min(a.max_train, len(tr)), random_state=0).reset_index(drop=True)
    if a.max_eval: va, te = va.head(a.max_eval), te.head(a.max_eval)

    dl = lambda df, train: DataLoader(CTCLines(df, a.data, cs, train=train, seed=a.seed), batch_size=a.bs,
                                      shuffle=train, num_workers=a.workers, collate_fn=collate_ctc,
                                      pin_memory=device.type == "cuda", drop_last=train,
                                      persistent_workers=a.workers > 0)
    dtr, dva, dte = dl(tr, True), dl(va, False), dl(te, False)

    model = build(a.model, len(cs)).to(device)
    use_sam = a.model == "htrvt" and not a.no_sam
    lr = a.lr or (1e-3 if a.model != "htrvt" else 5e-4)
    if use_sam:
        opt = SAM(model.parameters(), torch.optim.AdamW, lr=lr, weight_decay=0.05, rho=0.05)
        base = opt.base_optimizer
    else:
        opt = base = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = a.epochs * len(dtr)
    sched = torch.optim.lr_scheduler.OneCycleLR(base, max_lr=lr, total_steps=max(steps, 1), pct_start=0.1)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and not use_sam)

    def loss_fn(x, y, yl):
        with torch.autocast(device_type=amp_type(device), dtype=torch.float16, enabled=device.type == "cuda"):
            lo = model(x)
        lp = lo.float().log_softmax(-1)
        if device.type == "mps":  # CTCLoss não tem implementação em MPS: calcula na CPU (o gradiente volta para a GPU)
            lp, y, yl = lp.cpu(), y.cpu(), yl.cpu()
        return ctc(lp, y, torch.full((x.size(0),), lp.size(0), dtype=torch.long), yl)

    print(f"[{run}] params={n_params(model)/1e6:.2f}M treino={len(tr)} val={len(va)} teste={len(te)} classes={len(cs)} device={device}")
    best, bad, hist = 1e9, 0, []
    t0 = time.time()
    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for x, y, yl, _, _ in dtr:
            x, y, yl = x.to(device, non_blocking=True), y.to(device), yl.to(device)
            if use_sam:
                loss = loss_fn(x, y, yl); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.first_step()
                loss_fn(x, y, yl).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.second_step()
            else:
                loss = loss_fn(x, y, yl)
                base.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.unscale_(base)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(base); scaler.update()
            sched.step(); tot += loss.item()
        pv = predict(model, dva, cs, device)
        vcer = cer(va.text.tolist(), [pv[i] for i in va.id])
        hist.append({"epoch": ep + 1, "loss": tot / max(1, len(dtr)), "val_cer": vcer, "time_s": time.time() - t0})
        print(f"  ep {ep+1:3d} loss {hist[-1]['loss']:.3f} val CER {vcer:.4f} ({hist[-1]['time_s']/60:.1f} min)", flush=True)
        if vcer < best:
            best, bad = vcer, 0
            torch.save({"model": model.state_dict(), "charset": cs.itos, "args": vars(a)}, os.path.join(out, "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print("  parada antecipada"); break
    train_time = time.time() - t0

    ck = torch.load(os.path.join(out, "best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    pt = predict(model, dte, cs, device)
    te = te.copy(); te["pred"] = [pt[i] for i in te.id]
    te[["id", "page", "text", "pred", "has_strike", "has_insert"]].to_csv(os.path.join(out, "test_predictions.csv"), index=False)
    rep = full_report(te)
    rep.update({"model": a.model, "frac": a.frac, "n_train": len(tr), "params_M": n_params(model) / 1e6,
                "best_val_cer": best, "epochs_run": len(hist), "train_time_min": train_time / 60,
                "cpu_ms_per_line": cpu_latency(model, CTCLines(te, a.data, cs))})
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), ensure_ascii=False, indent=1)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    print(f"[{run}] TESTE  CER={rep['CER']:.4f} WER={rep['WER']:.4f}  acc_IV={rep['spelling']['acc_IV']:.3f} "
          f"acc_ERR={rep['spelling']['acc_ERR']:.3f} sobrecorreção={rep['spelling']['overcorr']:.3f}")


if __name__ == "__main__":
    main()
