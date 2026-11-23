"""Fine-tuning do TrOCR (Li et al., AAAI 2023) no BRESSAY.

Exemplo:
    python -m src.train_trocr --data /content/bressay --out /content/drive/MyDrive/htr_runs --epochs 8
"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, get_linear_schedule_with_warmup

from .data import LinesNoBOS, load_manifest, collate_trocr
from .metrics import cer, full_report
from .device import amp_type, get_device


@torch.no_grad()
def predict(model, proc, loader, device, beams=1, max_new=128):
    model.eval()
    preds, ids = [], []
    for pv, _, _, bid in loader:
        with torch.autocast(device_type=amp_type(device), dtype=torch.float16, enabled=device.type == "cuda"):
            g = model.generate(pv.to(device), num_beams=beams, max_new_tokens=max_new, early_stopping=beams > 1)
        preds += [s.strip() for s in proc.batch_decode(g, skip_special_tokens=True)]; ids += bid
    return dict(zip(ids, preds))


@torch.no_grad()
def cpu_latency(model, ds, beams, n=30):
    m = model.to("cpu").eval()
    xs = [ds[i][0][None] for i in range(min(n, len(ds)))]
    m.generate(xs[0], num_beams=beams, max_new_tokens=128)
    t = time.perf_counter()
    for x in xs:
        m.generate(x, num_beams=beams, max_new_tokens=128)
    return (time.perf_counter() - t) / len(xs) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="microsoft/trocr-small-handwritten")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--frac", type=float, default=1.0)
    ap.add_argument("--beams", type=int, default=4, help="beams na avaliação de teste (validação usa greedy)")
    ap.add_argument("--val_subset", type=int, default=1000, help="nº de linhas de validação por época (acelera)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max_train", type=int, default=0)
    ap.add_argument("--max_eval", type=int, default=0)
    ap.add_argument("--zero_shot", action="store_true", help="só avalia o checkpoint pré-treinado (sem fine-tuning)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None, help="cuda, mps ou cpu (padrão: automático)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = get_device(a.device)
    run = "trocr_zeroshot" if a.zero_shot else f"trocr_frac{a.frac:g}"
    out = os.path.join(a.out, run); os.makedirs(out, exist_ok=True)

    proc = TrOCRProcessor.from_pretrained(a.ckpt)
    model = VisionEncoderDecoderModel.from_pretrained(a.ckpt)
    tok = proc.tokenizer
    # O checkpoint do TrOCR começa a decodificação com </s> (id 2) e gera o texto direto, sem <s>.
    # Usar outro token inicial (ex.: <s>) quebra o modelo pré-treinado (saídas repetitivas).
    gc = model.generation_config
    start = gc.decoder_start_token_id if gc.decoder_start_token_id is not None else tok.eos_token_id
    model.config.decoder_start_token_id = start
    model.config.pad_token_id = tok.pad_token_id
    model.config.eos_token_id = tok.eos_token_id
    gc.decoder_start_token_id, gc.pad_token_id, gc.eos_token_id = start, tok.pad_token_id, tok.eos_token_id
    BOS = tok.bos_token_id
    model.to(device)

    tr, va, te = load_manifest(a.data, frac=a.frac, seed=a.seed)
    if a.max_train: tr = tr.sample(min(a.max_train, len(tr)), random_state=0).reset_index(drop=True)
    if a.max_eval: va, te = va.head(a.max_eval), te.head(a.max_eval)
    va_s = va.sample(min(a.val_subset, len(va)), random_state=0).reset_index(drop=True)
    col = collate_trocr

    Lines = lambda *args, **kw: LinesNoBOS(*args, bos=BOS, **kw)

    dl = lambda df, train, bs: DataLoader(Lines(df, a.data, proc, train=train, seed=a.seed), batch_size=bs,
                                          shuffle=train, num_workers=a.workers, collate_fn=col,
                                          pin_memory=device.type == "cuda", drop_last=train)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{run}] params={n_params:.1f}M treino={len(tr)} val={len(va_s)} teste={len(te)} device={device}")

    best, bad, hist, t0 = 1e9, 0, [], time.time()
    if not a.zero_shot:
        dtr, dva = dl(tr, True, a.bs), dl(va_s, False, a.bs * 2)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
        steps = a.epochs * len(dtr)
        sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
        scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
        for ep in range(a.epochs):
            model.train(); tot = 0.0
            for pv, y, _, _ in dtr:
                with torch.autocast(device_type=amp_type(device), dtype=torch.float16, enabled=device.type == "cuda"):
                    loss = model(pixel_values=pv.to(device), labels=y.to(device)).loss
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step(); tot += loss.item()
            pvd = predict(model, proc, dva, device, beams=1)
            vcer = cer(va_s.text.tolist(), [pvd[i] for i in va_s.id])
            hist.append({"epoch": ep + 1, "loss": tot / len(dtr), "val_cer": vcer, "time_s": time.time() - t0})
            print(f"  ep {ep+1:3d} loss {hist[-1]['loss']:.3f} val CER {vcer:.4f} ({hist[-1]['time_s']/60:.1f} min)", flush=True)
            if vcer < best:
                best, bad = vcer, 0
                model.save_pretrained(os.path.join(out, "best")); proc.save_pretrained(os.path.join(out, "best"))
            else:
                bad += 1
                if bad >= a.patience:
                    print("  parada antecipada"); break
        model = VisionEncoderDecoderModel.from_pretrained(os.path.join(out, "best")).to(device)
    train_time = time.time() - t0

    ds_te = Lines(te, a.data, proc)
    dte = DataLoader(ds_te, batch_size=a.bs * 2, num_workers=a.workers, collate_fn=col)
    pt = predict(model, proc, dte, device, beams=a.beams)
    te = te.copy(); te["pred"] = [pt[i] for i in te.id]
    te[["id", "page", "text", "pred", "has_strike", "has_insert"]].to_csv(os.path.join(out, "test_predictions.csv"), index=False)
    rep = full_report(te)
    rep.update({"model": "trocr_zeroshot" if a.zero_shot else "trocr", "frac": a.frac, "n_train": 0 if a.zero_shot else len(tr),
                "params_M": n_params, "best_val_cer": best, "epochs_run": len(hist), "train_time_min": train_time / 60,
                "beams": a.beams, "cpu_ms_per_line": cpu_latency(model, ds_te, a.beams)})
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), ensure_ascii=False, indent=1)
    pd.DataFrame(hist).to_csv(os.path.join(out, "history.csv"), index=False)
    print(f"[{run}] TESTE  CER={rep['CER']:.4f} WER={rep['WER']:.4f}  acc_IV={rep['spelling']['acc_IV']:.3f} "
          f"acc_ERR={rep['spelling']['acc_ERR']:.3f} sobrecorreção={rep['spelling']['overcorr']:.3f}")


if __name__ == "__main__":
    main()
