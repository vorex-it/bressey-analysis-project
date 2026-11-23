"""Figura 1 do artigo: linhas manuscritas do BRESSAY com o alvo e a saída de cada modelo.
Uso: python -m src.fig_exemplos --runs <htr_runs> --data <bressay_lines> --out artigo/figs/exemplos.png"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

VERDE, VERMELHO = "#1b7f3b", "#c0392b"
MODELS = [("crnn", "CRNN"), ("flor", "HTR-Flor"), ("htrvt", "HTR-VT"), ("trocr", "TrOCR")]
# (id da linha, palavra em destaque, rótulo do painel)
EXEMPLOS = [
    ("2187-044-03-05", {"inadimissível": VERDE, "inadmissível": VERMELHO}, "(a) erro ortográfico do aluno (verde: preservado; vermelho: corrigido pelo modelo)"),
    ("1756-020-03-01", {}, "(b) palavra riscada pelo aluno (\"importante\"), que não deve constar da transcrição"),
]


def row(ax, y, label, text, marcas):
    """Escreve `text` em y; palavras em `marcas` (palavra -> cor) ficam coloridas e em negrito."""
    ax.text(0.0, y, label, transform=ax.transAxes, fontsize=8.5, color="#444", va="center", ha="left", fontweight="bold")
    partes, resto = [], text
    while resto:
        achado = [(resto.find(w), w) for w in marcas if w in resto]
        if not achado:
            partes.append((resto, None))
            break
        i, w = min(achado)
        if i:
            partes.append((resto[:i], None))
        partes.append((w, marcas[w]))
        resto = resto[i + len(w):]
    prev = ax.text(0.14, y, partes[0][0], transform=ax.transAxes, fontsize=9, va="center", ha="left",
                   color=partes[0][1] or "black", fontweight="bold" if partes[0][1] else "normal")
    for trecho, cor in partes[1:]:
        prev = ax.annotate(trecho, xy=(1, 0.5), xycoords=prev, va="center", ha="left", fontsize=9,
                           color=cor or "black", fontweight="bold" if cor else "normal")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="exemplos.png")
    a = ap.parse_args()
    preds = {m: pd.read_csv(f"{a.runs}/{m}_frac1/test_predictions.csv").fillna("").set_index("id") for m, _ in MODELS}
    man = pd.read_csv(f"{a.data}/manifest.csv").set_index("id")
    fig, axes = plt.subplots(len(EXEMPLOS), 1, figsize=(7.2, 2.7 * len(EXEMPLOS)))
    for ax, (lid, marcas, titulo) in zip(axes, EXEMPLOS):
        ax.axis("off")
        ax.set_title(titulo, fontsize=9.5, loc="left", pad=2)
        img = Image.open(f"{a.data}/lines/{lid}.png").convert("L")
        w, h = img.size
        axi = ax.inset_axes([0.14, 0.62, 0.80, 0.34])
        axi.imshow(img, cmap="gray", aspect="auto", interpolation="bicubic")
        axi.axis("off")
        ref = man.loc[lid, "text"]
        row(ax, 0.50, "Alvo", ref, marcas)
        for k, (m, nome) in enumerate(MODELS):
            p = preds[m].loc[lid, "pred"]
            row(ax, 0.38 - 0.115 * k, nome, p, marcas)
    plt.tight_layout(h_pad=1.2)
    fig.savefig(a.out, dpi=300)


if __name__ == "__main__":
    main()
