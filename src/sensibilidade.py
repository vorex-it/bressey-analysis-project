"""Sensibilidade das métricas de fidelidade ao limiar de similaridade que define "palavra válida e parecida".

Para cada limiar, recalcula por modelo (treino completo): sobrecorreção, P(válida | erro) nos erros do
aluno e nas palavras inéditas, e a razão de viés. O objetivo é verificar se a ordem entre modelos e a
conclusão sobre o viés relativo dependem do limiar 0,6, fixado depois de uma primeira inspeção.
Uso: python -m src.sensibilidade --runs <pasta_dos_runs> --data <pasta_bressay_lines> [--out resultados/sensibilidade.csv]
"""
import argparse, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import load_manifest
from src.metrics import Lexicon, spelling_fidelity

LIMIARES = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8]
MODELOS = ["crnn", "flor", "htrvt", "trocr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="resultados/sensibilidade.csv")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    tr, _, _ = load_manifest(a.data)
    seen = {w.lower() for t in tr.text for w in re.findall(r"[A-Za-zÀ-ÿ]+", t)}
    lex = Lexicon()
    rows = []
    for m in MODELOS:
        p = pd.read_csv(os.path.join(a.runs, f"{m}_frac1", "test_predictions.csv"), keep_default_na=False, dtype={"page": str})
        for t in LIMIARES:
            s = spelling_fidelity(p.text.tolist(), p.pred.tolist(), lex, seen, min_sim=t)
            pe, pu = s["pvalid_wrong_ERR"], s["pvalid_wrong_IVu"]
            rows.append(dict(modelo=m, limiar=t, sobrecorr=100 * s["overcorr"], pv_erro=100 * pe, pv_inedita=100 * pu,
                             razao=pe / pu if pu else float("nan")))
    # diferença de razão de viés entre pares de modelos, com bootstrap pareado por página
    pares = [("crnn", "trocr"), ("crnn", "flor"), ("flor", "trocr")]
    preds = {m: pd.read_csv(os.path.join(a.runs, f"{m}_frac1", "test_predictions.csv"), keep_default_na=False, dtype={"page": str}) for m in MODELOS}
    pages = sorted(preds["crnn"].page.unique())
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(pages), (a.boot, len(pages)))
    dif = []
    for t in [0.6, 0.8]:
        cnt = {}
        for m in MODELOS:
            g = preds[m].groupby("page")
            cnt[m] = np.array([[spelling_fidelity(g.get_group(pg).text.tolist(), g.get_group(pg).pred.tolist(), lex, seen, min_sim=t)["counts"].get(k, 0)
                                for k in ("wrongvalid_ERR", "wrong_ERR", "wrongvalid_IVu", "wrong_IVu")] for pg in pages])
        def razao(m, ii):
            x = cnt[m][ii].sum(1) if False else cnt[m][ii].sum(axis=1)
            return (x[:, 0] / np.maximum(x[:, 1], 1)) / np.maximum(x[:, 2] / np.maximum(x[:, 3], 1), 1e-9)
        for A, B in pares:
            d = razao(A, idx) - razao(B, idx)
            obs = razao(A, np.arange(len(pages))[None, :])[0] - razao(B, np.arange(len(pages))[None, :])[0]
            dif.append(dict(limiar=t, A=A, B=B, dif_razao=obs, ic_inf=np.percentile(d, 2.5), ic_sup=np.percentile(d, 97.5)))
    dd = pd.DataFrame(dif)
    dd.to_csv(a.out.replace(".csv", "_diferenca_razao.csv"), index=False)
    print("\nDiferença de razão de viés (A − B), IC 95% por bootstrap pareado de páginas")
    print(dd.round(2).to_string(index=False))
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    pd.set_option("display.width", 200)
    for col, nome in [("sobrecorr", "Sobrecorreção (%)"), ("razao", "Razão de viés")]:
        print(f"\n{nome}")
        print(df.pivot(index="modelo", columns="limiar", values=col).loc[MODELOS].round(2).to_string())
    for t in LIMIARES:
        o = df[df.limiar == t].sort_values("sobrecorr")
        r = df[df.limiar == t].sort_values("razao", ascending=False)
        print(f"limiar {t}: ordem por sobrecorreção (menor→maior) {list(o.modelo)}; maior razão: {r.iloc[0].modelo}")


if __name__ == "__main__":
    main()
