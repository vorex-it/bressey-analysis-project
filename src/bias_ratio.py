"""Razão de viés lexical: P(saída é palavra válida | transcrição errada) em erros do aluno dividida
pela mesma probabilidade em palavras válidas inéditas no treino (grupo de controle).
IC 95% por bootstrap de páginas. Uso:
  python -m src.bias_ratio --runs <pasta_htr_runs> --data <pasta_bressay_lines> --out razao_vies.json
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from src.metrics import Lexicon, spelling_fidelity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="razao_vies.json")
    ap.add_argument("--models", default="crnn,flor,htrvt,trocr")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    lex = Lexicon()
    m = pd.read_csv(f"{a.data}/manifest.csv")
    tr = m[(m.split == "train") & (m.illegible == 0)] if "illegible" in m else m[m.split == "train"]
    seen = {w.lower() for t in tr.text.dropna() for w in re.findall(r"[A-Za-zÀ-ÿ]+", t)}
    rng = np.random.default_rng(a.seed)
    out = {}
    flags = {}
    for mod in a.models.split(","):
        df = pd.read_csv(f"{a.runs}/{mod}_frac1/test_predictions.csv").fillna({"text": "", "pred": ""})
        f = spelling_fidelity(df.text.tolist(), df.pred.tolist(), lex, seen)
        groups = {p: d for p, d in df.groupby("page")}
        pages = list(groups)
        rs = []
        for _ in range(a.boot):
            d = pd.concat([groups[p] for p in rng.choice(pages, len(pages))])
            x = spelling_fidelity(d.text.tolist(), d.pred.tolist(), lex, seen)
            rs.append((x["pvalid_wrong_ERR"] / max(x["pvalid_wrong_IVu"], 1e-9),
                       x["pvalid_wrong_ERR"] - x["pvalid_wrong_IVu"]))
        rs = np.array(rs) if rs else np.zeros((1, 2))
        flags[mod] = f["err_overcorr_flags"]
        out[mod] = {
            "pvalid_wrong_ERR": f["pvalid_wrong_ERR"], "pvalid_wrong_IVu": f["pvalid_wrong_IVu"],
            "ratio": f["pvalid_wrong_ERR"] / f["pvalid_wrong_IVu"],
            "ratio_ci95": np.percentile(rs[:, 0], [2.5, 97.5]).tolist(),
            "diff_ci95": np.percentile(rs[:, 1], [2.5, 97.5]).tolist(),
            "n_ERR": f["n_ERR"], "n_IVu": f["n_IVu"], "boot": a.boot,
        }
        print(mod, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[mod].items() if k != "ratio_ci95" and k != "diff_ci95"},
              "ratio IC", np.round(out[mod]["ratio_ci95"], 2))
    # McNemar exato (binomial) sobre os mesmos erros do aluno; limiar de Bonferroni: 0,05 / n_pares
    from itertools import combinations
    from scipy.stats import binomtest
    mc = {}
    mods = list(flags)
    for x, y in combinations(mods, 2):
        b = sum(1 for u, v in zip(flags[x], flags[y]) if u and not v)
        c = sum(1 for u, v in zip(flags[x], flags[y]) if v and not u)
        pv = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
        mc[f"{x} vs {y}"] = {"only_first": b, "only_second": c, "p": pv}
        print("McNemar", x, "vs", y, "b=%d c=%d p=%.4f" % (b, c, pv))
    out["_mcnemar_sobrecorrecao"] = mc
    out["_bonferroni_alpha"] = 0.05 / max(1, len(mc))
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
