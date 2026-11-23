"""Três arquiteturas CTC usadas na comparação (a quarta, TrOCR, vem do HuggingFace).

Todas recebem 1x40x1280 e produzem uma sequência de 320 passos (largura/4) -> logits (T, B, C).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- CRNN
class CRNN(nn.Module):
    """CRNN (Shi, Bai & Yao, TPAMI 2017): CNN estilo VGG + 2x BiLSTM + CTC."""

    def __init__(self, n_classes, hidden=256):
        super().__init__()

        def blk(i, o, bn=False):
            layers = [nn.Conv2d(i, o, 3, 1, 1)]
            if bn:
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.ReLU(inplace=True))
            return layers

        self.cnn = nn.Sequential(
            *blk(1, 64), nn.MaxPool2d(2, 2),                      # 20 x 640
            *blk(64, 128), nn.MaxPool2d(2, 2),                    # 10 x 320
            *blk(128, 256, True), *blk(256, 256), nn.MaxPool2d((2, 1), (2, 1)),  # 5 x 320
            *blk(256, 512, True), *blk(512, 512, True), nn.MaxPool2d((2, 1), (2, 1)),  # 2 x 320
            *blk(512, 512, True),
        )
        self.rnn = nn.LSTM(512, hidden, num_layers=2, bidirectional=True, dropout=0.25, batch_first=True)
        self.fc = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        f = self.cnn(x).mean(2).transpose(1, 2)  # B, T, 512
        f, _ = self.rnn(f)
        return self.fc(f).transpose(0, 1)        # T, B, C


# --------------------------------------------------------------------------- HTR-Flor
class FullGatedConv2d(nn.Module):
    """Convolução com porta (Neto et al., 2020): h = conv_a(x) * sigmoid(conv_b(x))."""

    def __init__(self, ch, k=3):
        super().__init__()
        self.conv = nn.Conv2d(ch, 2 * ch, k, 1, k // 2)

    def forward(self, x):
        a, b = self.conv(x).chunk(2, 1)
        return a * torch.sigmoid(b)


class HTRFlor(nn.Module):
    """HTR-Flor (Neto et al., SIBGRAPI 2020), reimplementado em PyTorch.
    Mantém a sequência de blocos conv -> PReLU -> BN -> FullGatedConv e o decodificador BiGRU-Dense-BiGRU;
    os strides foram adaptados para entrada 40x1280 (original: 128x1024) e para gerar 320 passos."""

    def __init__(self, n_classes, hidden=128):
        super().__init__()

        def blk(i, o, k, s, gated=True, drop=0.0):
            L = [nn.Conv2d(i, o, k, s, padding=(k[0] // 2, k[1] // 2) if isinstance(k, tuple) else k // 2),
                 nn.PReLU(o), nn.BatchNorm2d(o)]
            if gated:
                L.append(FullGatedConv2d(o))
            if drop:
                L.append(nn.Dropout(drop))
            return L

        self.cnn = nn.Sequential(
            *blk(1, 16, 3, (2, 2)),               # 20 x 640
            *blk(16, 32, 3, 1),
            *blk(32, 40, (3, 3), (2, 2), drop=0.2),  # 10 x 320
            *blk(40, 48, 3, 1, drop=0.2),
            *blk(48, 56, (3, 3), (2, 1), drop=0.2),  # 5 x 320
            *blk(56, 64, 3, 1, gated=False),
        )
        self.gru1 = nn.GRU(64 * 5, hidden, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(2 * hidden, 2 * hidden)
        self.gru2 = nn.GRU(2 * hidden, hidden, bidirectional=True, batch_first=True)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(2 * hidden, n_classes)

    def forward(self, x):
        f = self.cnn(x)                          # B, 64, 5, 320
        B, C, H, W = f.shape
        f = f.permute(0, 3, 1, 2).reshape(B, W, C * H)
        f, _ = self.gru1(self.drop(f))
        f = self.fc1(f)
        f, _ = self.gru2(self.drop(f))
        return self.out(f).transpose(0, 1)


# --------------------------------------------------------------------------- HTR-VT
class BasicBlock(nn.Module):
    def __init__(self, i, o, stride=(1, 1)):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.sc = (nn.Sequential(nn.Conv2d(i, o, 1, stride, bias=False), nn.BatchNorm2d(o))
                   if stride != (1, 1) or i != o else nn.Identity())

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(y)) + self.sc(x))


class HTRVT(nn.Module):
    """HTR-VT (Li et al., Pattern Recognition 2025): extrator CNN (ResNet) + encoder ViT + CTC,
    com mascaramento de spans de tokens no treino (regularizador central do método para bases pequenas).
    Versão reduzida: d=256 (orig. 768), 4 camadas, 8 cabeças (orig. 6); máscara 0,4 com spans de 8 (como no original)."""

    def __init__(self, n_classes, d=256, layers=4, heads=8, mask_ratio=0.4, span=8, T=320):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            BasicBlock(32, 64, (2, 2)),    # 20 x 640
            BasicBlock(64, 128, (2, 2)),   # 10 x 320
            BasicBlock(128, 256, (2, 1)),  # 5 x 320
            BasicBlock(256, 256, (1, 1)),
        )
        self.proj = nn.Linear(256 * 5, d)
        # posição senoidal fixa, como no artigo original
        pe = torch.zeros(T, d); pos = torch.arange(T)[:, None].float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pos", pe[None], persistent=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d)); nn.init.trunc_normal_(self.mask_token, std=0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.1, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)
        self.mask_ratio, self.span = mask_ratio, span

    def span_mask(self, B, T, device):
        m = torch.zeros(B, T, dtype=torch.bool, device=device)
        n_spans = int(T * self.mask_ratio / self.span)
        for b in range(B):
            starts = torch.randint(0, T - self.span, (n_spans,))
            for s in starts.tolist():
                m[b, s:s + self.span] = True
        return m

    def forward(self, x):
        f = self.cnn(x)
        B, C, H, W = f.shape
        t = self.proj(f.permute(0, 3, 1, 2).reshape(B, W, C * H))
        if self.training and self.mask_ratio > 0:
            m = self.span_mask(B, W, x.device)
            t = torch.where(m[..., None], self.mask_token.expand(B, W, -1), t)
        t = self.norm(self.enc(t + self.pos[:, :W]))
        return self.head(t).transpose(0, 1)


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2021), usado pelo HTR-VT."""

    def __init__(self, params, base_optimizer, rho=0.05, **kw):
        defaults = dict(rho=rho, **kw)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kw)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        norm = torch.norm(torch.stack([p.grad.norm(2) for g in self.param_groups for p in g["params"] if p.grad is not None]), 2)
        for g in self.param_groups:
            scale = g["rho"] / (norm + 1e-12)
            for p in g["params"]:
                if p.grad is None:
                    continue
                e = p.grad * scale
                p.add_(e); self.state[p]["e_w"] = e
        self.zero_grad()

    @torch.no_grad()
    def second_step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if "e_w" in self.state[p]:
                    p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        self.zero_grad()


def build(name, n_classes):
    return {"crnn": CRNN, "flor": HTRFlor, "htrvt": HTRVT}[name](n_classes)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
