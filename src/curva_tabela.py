"""Tabela da curva de aprendizado: CER, WER e métricas de fidelidade por modelo e fração de treino."""
import sys, re, glob, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.data import load_manifest
from src.metrics import Lexicon, full_report, spelling_fidelity
ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--data", required=True); ap.add_argument("--out", default="resultados/curva_tabela.csv")
a = ap.parse_args()
tr, _, _ = load_manifest(a.data)
seen = {w.lower() for t in tr.text for w in re.findall(r"[A-Za-zÀ-ÿ]+", t)}
lex = Lexicon(); rows = []
for d in sorted(glob.glob(os.path.join(a.runs, "*_frac*"))):
    n = os.path.basename(d); m, f = n.split("_frac"); f = float(f)
    p = pd.read_csv(d + "/test_predictions.csv", keep_default_na=False, dtype={"page": str})
    r = full_report(p, lex); s = spelling_fidelity(p.text.tolist(), p.pred.tolist(), lex, seen)
    rows.append(dict(model=m, frac=f, CER=100*r["CER"], WER=100*r["WER"], accERR=100*s["acc_ERR"],
                     overcorr=100*s["overcorr"], accIV=100*s["acc_IV"],
                     pvER=100*s["pvalid_wrong_ERR"], pvIVu=100*s["pvalid_wrong_IVu"]))
df = pd.DataFrame(rows); df["ratio"] = df.pvER / df.pvIVu
df = df.sort_values(["frac", "model"])
pd.set_option("display.width", 200); print(df.round(2).to_string(index=False))
df.to_csv(a.out, index=False)
