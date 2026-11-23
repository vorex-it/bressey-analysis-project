"""
Prepara o BRESSAY (nível de linha) para os experimentos.

Uso (na máquina onde o dataset foi extraído):
    python prepare_data.py --bressay ~/Downloads/bressay --out ~/Downloads/bressay_lines.zip

Gera um .zip com:
    lines/<id>.png      imagens de linha (originais, sem reamostragem)
    manifest.csv        id, split, text (alvo limpo), raw (anotação original), flags

Política de alvo ("texto final do aluno"), descrita na seção de metodologia do artigo:
  * trechos riscados (--texto--, --xxx--, inclusive dentro de ##..## ou $$..$$) são REMOVIDOS:
    o modelo deve aprender a ignorar a rasura, como faria um corretor humano;
  * inserções sobrescritas/subscritas (##texto##, $$texto$$) são MANTIDAS sem os marcadores;
  * linhas com texto ilegível (@@???@@) recebem flag `illegible=1` e são excluídas do
    treino e da avaliação principal (relatadas à parte).
  * erros ortográficos do aluno são preservados pela anotação original e NÃO são alterados.
"""
import argparse, csv, io, os, re, struct, zipfile

STRIKE = re.compile(r"(##|\$\$)?--.*?--(##|\$\$)?")
SUPER = re.compile(r"##(.*?)##")
SUB = re.compile(r"\$\$(.*?)\$\$")
ILLEG = "@@???@@"


def clean(raw: str) -> str:
    t = STRIKE.sub(" ", raw)
    t = SUPER.sub(r"\1", t)
    t = SUB.sub(r"\1", t)
    t = t.replace(ILLEG, " ")
    return re.sub(r"\s+", " ", t).strip()


def png_size(path):
    with open(path, "rb") as f:
        h = f.read(24)
    return struct.unpack(">II", h[16:24])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bressay", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = os.path.expanduser(a.bressay)
    split_of = {}
    for s in ("training", "validation", "test"):
        for name in open(os.path.join(root, "sets", f"{s}.txt")).read().split():
            split_of[name] = {"training": "train", "validation": "val", "test": "test"}[s]

    rows = []
    zf = zipfile.ZipFile(os.path.expanduser(a.out), "w", zipfile.ZIP_STORED)
    lines_dir = os.path.join(root, "data", "lines")
    for page in sorted(os.listdir(lines_dir)):
        split = split_of.get(page)
        if split is None:
            continue
        pdir = os.path.join(lines_dir, page)
        for fn in sorted(os.listdir(pdir)):
            if not fn.endswith(".png"):
                continue
            lid = fn[:-4]
            txt = os.path.join(pdir, lid + ".txt")
            if not os.path.exists(txt):
                continue
            raw = open(txt, encoding="utf-8").read().strip()
            w, h = png_size(os.path.join(pdir, fn))
            rows.append({
                "id": lid, "page": page, "split": split, "text": clean(raw), "raw": raw,
                "has_strike": int("--" in raw), "has_insert": int("##" in raw or "$$" in raw),
                "illegible": int("@@" in raw), "width": w, "height": h,
            })
            zf.write(os.path.join(pdir, fn), f"lines/{fn}")
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)
    zf.writestr("manifest.csv", buf.getvalue())
    zf.close()
    from collections import Counter
    print("linhas por split:", Counter(r["split"] for r in rows))
    print("com rasura:", sum(r["has_strike"] for r in rows), "| ilegíveis:", sum(r["illegible"] for r in rows))


if __name__ == "__main__":
    main()
