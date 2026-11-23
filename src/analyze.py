"""Agrega os runs e gera tabelas (Markdown + LaTeX), figuras e testes estatísticos para o artigo.

    python -m src.analyze --runs /content/drive/MyDrive/htr_runs --out results
"""
import argparse, glob, json, os
from itertools import combinations

import Levenshtein as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NAMES = {"crnn": "CRNN", "flor": "HTR-Flor", "htrvt": "HTR-VT", "trocr": "TrOCR-small", "trocr_zeroshot": "TrOCR-small (zero-shot)"}
ORDER = ["crnn", "flor", "htrvt", "trocr", "trocr_zeroshot"]
COLORS = {"crnn": "#4C78A8", "flor": "#F58518", "htrvt": "#54A24B", "trocr": "#B279A2", "trocr_zeroshot": "#9D9D9D"}


def page_stats(pred_csv):
    d = pd.read_csv(pred_csv, keep_default_na=False, dtype={"page": str})
    d["err"] = [L.distance(r, h) for r, h in zip(d.text, d.pred)]
    d["n"] = d.text.str.len()
    return d.groupby("page")[["err", "n"]].sum()


def bootstrap_ci(ps, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    e, n = ps.err.values, ps.n.values
    idx = rng.integers(0, len(e), (B, len(e)))
    s = e[idx].sum(1) / n[idx].sum(1)
    return np.percentile(s, [2.5, 97.5])


def paired_bootstrap(pa, pb, B=5000, seed=0):
    """p-valor bilateral para H0: CER(A) == CER(B), reamostrando páginas (escritores)."""
    j = pa.join(pb, lsuffix="_a", rsuffix="_b", how="inner")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(j), (B, len(j)))
    ea, eb, n = j.err_a.values, j.err_b.values, j.n_a.values
    diff = (ea[idx].sum(1) - eb[idx].sum(1)) / n[idx].sum(1)
    obs = (ea.sum() - eb.sum()) / n.sum()
    p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
    return obs, min(1.0, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--data", default=None, help="pasta com manifest.csv: recalcula métricas a partir das predições")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    reps = []
    seen, lex = None, None
    if a.data:  # recalcula as métricas a partir das predições (inclui o controle de palavras inéditas)
        import re as _re
        from .data import load_manifest
        from .metrics import Lexicon, full_report, spelling_fidelity
        tr, _, te_m = load_manifest(a.data)
        seen = {w.lower() for t in tr.text for w in _re.findall(r"[A-Za-zÀ-ÿ]+", t)}
        lex = Lexicon()
    for d in sorted(glob.glob(os.path.join(a.runs, "*"))):
        pred, rep = os.path.join(d, "test_predictions.csv"), os.path.join(d, "report.json")
        if not os.path.exists(pred) or os.path.basename(d).startswith("_"):
            continue
        r = json.load(open(rep)) if os.path.exists(rep) else {}
        name = os.path.basename(d)
        if not r:  # run sem report (ex.: sessão caiu na medição de latência)
            model, frac = name.rsplit("_frac", 1) if "_frac" in name else (name, "1")
            r = {"model": model, "frac": float(frac), "params_M": 61.6 if model.startswith("trocr") else np.nan,
                 "cpu_ms_per_line": np.nan, "train_time_min": np.nan, "n_train": np.nan}
            h = os.path.join(d, "history.csv")
            if os.path.exists(h) and os.path.getsize(h) > 1:
                r["train_time_min"] = pd.read_csv(h).time_s.iloc[-1] / 60
        if lex is not None:
            p = pd.read_csv(pred, keep_default_na=False, dtype={"page": str})
            r.update({k: v for k, v in full_report(p, lex).items()})
            r["spelling"] = spelling_fidelity(p.text.tolist(), p.pred.tolist(), lex, seen)
        r["dir"] = d; reps.append(r)
    if not reps:
        raise SystemExit("nenhum report.json encontrado")
    df = pd.DataFrame(reps)

    # ------------------------------------------------------------ tabela principal (fração = 1)
    full = df[df.frac == 1.0].copy()
    full["ord"] = full.model.map(ORDER.index); full = full.sort_values("ord")
    rows, pstats = [], {}
    for _, r in full.iterrows():
        ps = page_stats(os.path.join(r.dir, "test_predictions.csv")); pstats[r.model] = ps
        lo, hi = bootstrap_ci(ps)
        sp = r.spelling
        rows.append({
            "Modelo": NAMES[r.model], "Parâm. (M)": f"{r.params_M:.1f}",
            "CER (%)": f"{100*r.CER:.2f} [{100*lo:.2f}–{100*hi:.2f}]", "WER (%)": f"{100*r.WER:.2f}",
            "CER s/ rasura": f"{100*r.CER_sem_rasura:.2f}", "CER c/ rasura": f"{100*r.CER_com_rasura:.2f}",
            "Acc. palavras válidas": f"{100*sp['acc_IV']:.1f}", "Acc. erros do aluno": f"{100*sp['acc_ERR']:.1f}",
            "Sobrecorreção (%)": f"{100*sp['overcorr']:.1f}",
            **({"Acc. válidas inéditas": f"{100*sp['acc_IVu']:.1f}",
                "P(válida|erro) aluno": f"{100*sp['pvalid_wrong_ERR']:.1f}",
                "P(válida|erro) inéditas": f"{100*sp['pvalid_wrong_IVu']:.1f}"} if "acc_IVu" in sp else {}),
            "CPU ms/linha": "—" if pd.isna(r.cpu_ms_per_line) else f"{r.cpu_ms_per_line:.0f}",
            "Treino (min)": "—" if pd.isna(r.train_time_min) else f"{r.train_time_min:.0f}",
        })
    main_t = pd.DataFrame(rows)

    # ------------------------------------------------------------ CER por classe
    cls_rows = []
    for _, r in full.iterrows():
        pc = r.per_class
        cls_rows.append({"Modelo": NAMES[r.model], **{k: f"{100*v:.2f}" for k, v in pc.items() if not k.startswith("_")},
                         "subst. só de diacrítico": pc["_diacritic_only_subs"]})
    cls_t = pd.DataFrame(cls_rows)

    # ------------------------------------------------------------ significância pareada
    sig = []
    models = [m for m in ORDER if m in pstats and m != "trocr_zeroshot"]
    for ma, mb in combinations(models, 2):
        d, p = paired_bootstrap(pstats[ma], pstats[mb])
        sig.append({"A": NAMES[ma], "B": NAMES[mb], "ΔCER (A−B, p.p.)": f"{100*d:+.2f}", "p (bootstrap pareado)": ("< 0,001" if p < 1e-3 else f"{p:.3f}")})
    sig_t = pd.DataFrame(sig)

    with open(os.path.join(a.out, "tabelas.md"), "w") as f:
        for title, t in (("Resultados no teste (IC 95% por bootstrap de páginas)", main_t),
                         ("CER (%) por classe de caractere", cls_t), ("Teste de significância pareado", sig_t)):
            f.write(f"## {title}\n\n{t.to_markdown(index=False)}\n\n")
    with open(os.path.join(a.out, "tabelas.tex"), "w") as f:
        for t in (main_t, cls_t, sig_t):
            f.write(t.to_latex(index=False).replace("%", "\\%") + "\n")

    # ------------------------------------------------------------ exemplos de sobrecorreção
    with open(os.path.join(a.out, "exemplos_sobrecorrecao.md"), "w") as f:
        for _, r in full.iterrows():
            ex = r.spelling["examples"]
            f.write(f"### {NAMES[r.model]}\n\n" + "\n".join(f"- `{w}` → `{p}`" for w, p in ex[:15]) + "\n\n")

    # ------------------------------------------------------------ figuras
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # curva de aprendizado
    lc = df[df.model.isin(["crnn", "flor", "htrvt", "trocr"])]
    if lc.frac.nunique() > 1:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for m, g in lc.groupby("model"):
            g = g.sort_values("n_train")
            ax.plot(g.n_train, 100 * g.CER, "o-", label=NAMES[m], color=COLORS[m])
        ax.set_xscale("log"); ax.set_xlabel("linhas de treino (escala log)"); ax.set_ylabel("CER no teste (%)")
        ax.legend(frameon=False); ax.grid(alpha=.3); fig.tight_layout()
        fig.savefig(os.path.join(a.out, "curva_aprendizado.png"), dpi=200); plt.close(fig)
    # histórico de validação
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for _, r in full.iterrows():
        h = os.path.join(r.dir, "history.csv")
        if os.path.exists(h) and os.path.getsize(h) > 1:
            h = pd.read_csv(h); ax.plot(h.epoch, 100 * h.val_cer, label=NAMES[r.model], color=COLORS[r.model])
    ax.set_xlabel("época"); ax.set_ylabel("CER validação (%)"); ax.set_ylim(0, None); ax.legend(frameon=False); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "convergencia.png"), dpi=200); plt.close(fig)
    # fidelidade ortográfica: erros do aluno x controle (palavras válidas inéditas)
    fm = full[full.model != "trocr_zeroshot"]
    has_ctrl = all("acc_IVu" in sp for sp in fm.spelling)
    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4)); x = np.arange(len(fm)); w = .38
    a0, a1 = axs
    ctrl = [100 * sp.get("acc_IVu", sp["acc_IV"]) for sp in fm.spelling]
    a0.bar(x - w / 2, ctrl, w, color="#4C78A8", label="palavras válidas inéditas" if has_ctrl else "palavras válidas")
    a0.bar(x + w / 2, [100 * sp["acc_ERR"] for sp in fm.spelling], w, color="#F58518", label="erros do aluno")
    a0.set_title("(a) transcritas exatamente (%)", fontsize=10)
    if has_ctrl:
        a1.bar(x - w / 2, [100 * sp["pvalid_wrong_IVu"] for sp in fm.spelling], w, color="#4C78A8")
        a1.bar(x + w / 2, [100 * sp["pvalid_wrong_ERR"] for sp in fm.spelling], w, color="#F58518")
        a1.set_title("(b) quando erra, sai outra palavra válida (%)", fontsize=10)
    else:
        a1.bar(x, [100 * sp["overcorr"] for sp in fm.spelling], w, color="#E45756"); a1.set_title("sobrecorreção (%)", fontsize=10)
    for ax in axs:
        ax.set_xticks(x, [NAMES[m] for m in fm.model], fontsize=9); ax.grid(axis="y", alpha=.3)
    fig.legend(*a0.get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(os.path.join(a.out, "fidelidade_ortografica.png"), dpi=200); plt.close(fig)
    # CER x custo
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for _, r in fm[fm.cpu_ms_per_line.notna()].iterrows():
        ax.scatter(r.cpu_ms_per_line, 100 * r.CER, s=30 + 4 * r.params_M, color=COLORS[r.model])
        ax.annotate(NAMES[r.model], (r.cpu_ms_per_line, 100 * r.CER), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_xscale("log"); ax.set_xlabel("latência em CPU (ms/linha, log)"); ax.set_ylabel("CER teste (%)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "cer_vs_custo.png"), dpi=200); plt.close(fig)

    # ------------------------------------------------------------ exemplos qualitativos (mesmas linhas p/ todos)
    preds = {m: pd.read_csv(os.path.join(full[full.model == m].dir.iloc[0], "test_predictions.csv"), keep_default_na=False)
             .set_index("id") for m in models}
    common = set.intersection(*(set(d.index) for d in preds.values()))
    base = preds[models[0]].loc[[i for i in preds[models[0]].index if i in common]]
    ids = base[base.has_strike == 1].index[:3].tolist() + base[base.has_strike == 0].index[:3].tolist()
    with open(os.path.join(a.out, "exemplos_qualitativos.md"), "w") as f:
        for i in ids:
            f.write(f"### {i}\n- **Referência:** {base.loc[i, 'text']}\n")
            for m in models:
                f.write(f"- **{NAMES[m]}:** {preds[m].loc[i, 'pred']}\n")
            f.write("\n")
    print(main_t.to_string(index=False)); print(); print(cls_t.to_string(index=False)); print(); print(sig_t.to_string(index=False))


if __name__ == "__main__":
    main()
