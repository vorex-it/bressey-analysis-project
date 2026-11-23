"""Carregamento do BRESSAY (linhas) a partir do pacote gerado por prepare_data.py."""
import os, random
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_H, IMG_W = 40, 1280  # altura nativa das linhas ~20px; razão de aspecto mediana ~24 (p95 ~32)


def load_manifest(data_dir, frac=1.0, seed=42, drop_illegible=True):
    """Lê o manifesto. `frac` subamostra o TREINO por página (escritor), para a curva de aprendizado."""
    m = pd.read_csv(os.path.join(data_dir, "manifest.csv"), keep_default_na=False, dtype={"page": str, "id": str})
    if drop_illegible:
        m = m[m.illegible == 0]
    m = m[m.text.str.len() > 0]
    tr, va, te = (m[m.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    if frac < 1.0:
        pages = sorted(tr.page.unique())
        rng = random.Random(seed)
        keep = set(rng.sample(pages, max(1, int(round(frac * len(pages))))))
        tr = tr[tr.page.isin(keep)].reset_index(drop=True)
    return tr, va, te


class Charset:
    BLANK = 0

    def __init__(self, texts):
        chars = sorted(set("".join(texts)))
        self.itos = ["<blank>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode_ctc(self, ids):
        out, prev = [], None
        for i in ids:
            if i != prev and i != self.BLANK:
                out.append(self.itos[i])
            prev = i
        return "".join(out)


def load_gray(path):
    return Image.open(path).convert("L")


def augment(img: Image.Image, rng: random.Random):
    """Aumento leve e idêntico para todos os modelos: esticamento horizontal, pequena inclinação e ruído de contraste."""
    w, h = img.size
    sx = rng.uniform(0.8, 1.2)
    img = img.resize((max(8, int(w * sx)), h), Image.BILINEAR)
    if rng.random() < 0.5:
        shear = rng.uniform(-0.3, 0.3)
        w2 = img.size[0] + int(abs(shear) * h)
        img = img.transform((w2, h), Image.AFFINE, (1, shear, -shear * h if shear > 0 else 0, 0, 1, 0),
                            resample=Image.BILINEAR, fillcolor=255)
    if rng.random() < 0.3:
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr * rng.uniform(0.8, 1.2) + rng.uniform(-20, 20), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
    return img


def to_tensor_ctc(img: Image.Image, H=IMG_H, W=IMG_W):
    """Redimensiona para altura H mantendo o aspecto; comprime se exceder W; preenche à direita.
    Retorna tensor 1xHxW com tinta=1, fundo=0."""
    w, h = img.size
    nw = max(1, min(W, int(round(w * H / h))))
    img = img.resize((nw, H), Image.BILINEAR)
    arr = 1.0 - np.asarray(img, dtype=np.float32) / 255.0
    out = np.zeros((H, W), dtype=np.float32)
    out[:, :nw] = arr
    return torch.from_numpy(out)[None], nw


class CTCLines(Dataset):
    def __init__(self, df, data_dir, charset, train=False, seed=0):
        self.df, self.dir, self.cs, self.train = df, data_dir, charset, train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_gray(os.path.join(self.dir, "lines", r.id + ".png"))
        if self.train:
            img = augment(img, self.rng)
        x, nw = to_tensor_ctc(img)
        return x, torch.tensor(self.cs.encode(r.text), dtype=torch.long), r.text, r.id


def collate_ctc(batch):
    xs, ys, texts, ids = zip(*batch)
    return (torch.stack(xs), torch.cat(ys), torch.tensor([len(y) for y in ys]), list(texts), list(ids))


class TrOCRLines(Dataset):
    def __init__(self, df, data_dir, processor, train=False, max_len=128, seed=0):
        self.df, self.dir, self.p, self.train, self.max_len = df, data_dir, processor, train, max_len
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = load_gray(os.path.join(self.dir, "lines", r.id + ".png"))
        if self.train:
            img = augment(img, self.rng)
        pv = self.p(images=img.convert("RGB"), return_tensors="pt").pixel_values[0]
        lab = self.p.tokenizer(r.text, max_length=self.max_len, truncation=True).input_ids
        return pv, torch.tensor(lab), r.text, r.id


def collate_trocr(batch):
    """Função de nível de módulo (picklable) para DataLoader com workers no macOS/Windows."""
    pv, labs, texts, ids = zip(*batch)
    L = max(len(l) for l in labs)
    y = torch.full((len(labs), L), -100, dtype=torch.long)
    for i, l in enumerate(labs):
        y[i, :len(l)] = l
    return torch.stack(pv), y, list(texts), list(ids)


class LinesNoBOS(TrOCRLines):
    """Rótulo = texto + </s>, sem o <s> inicial (como no pré-treino do TrOCR).
    Definida no nível do módulo para funcionar com DataLoader multiprocesso no macOS."""

    def __init__(self, *args, bos=None, **kw):
        super().__init__(*args, **kw)
        self.bos = bos

    def __getitem__(self, i):
        pv, lab, t, idx = super().__getitem__(i)
        if len(lab) and lab[0] == self.bos:
            lab = lab[1:]
        return pv, lab, t, idx
