"""Métricas: CER/WER, CER por classe de caractere e fidelidade ortográfica (sobrecorreção)."""
import os
import re
import unicodedata
from collections import Counter
from functools import lru_cache

import Levenshtein as L

# ----------------------------------------------------------------------------- CER / WER

def cer(refs, hyps):
    e = sum(L.distance(r, h) for r, h in zip(refs, hyps))
    return e / max(1, sum(len(r) for r in refs))


def wer(refs, hyps):
    e = n = 0
    for r, h in zip(refs, hyps):
        rw, hw = r.split(), h.split()
        vocab = {w: chr(0x100 + i) for i, w in enumerate(set(rw) | set(hw))}
        e += L.distance("".join(vocab[w] for w in rw), "".join(vocab[w] for w in hw))
        n += len(rw)
    return e / max(1, n)


# ----------------------------------------------------------------------------- CER por classe
ACCENTED = set("áàâãéèêíóôõúüçÁÀÂÃÉÈÊÍÓÔÕÚÜÇ")


def char_class(c):
    if c == " ":
        return "espaço"
    if c in ACCENTED:
        return "acento/cedilha"
    if c.isalpha():
        return "letra s/ acento"
    if c.isdigit():
        return "dígito"
    return "pontuação/símbolo"


def per_class_cer(refs, hyps):
    """Erros (substituição + remoção) atribuídos à classe do caractere de referência / total da classe.
    Também conta, para acentos, quantas substituições trocaram só o diacrítico (ex.: ã -> a)."""
    tot, err = Counter(), Counter()
    diacritic_only = 0
    for r, h in zip(refs, hyps):
        for c in r:
            tot[char_class(c)] += 1
        for op, i, j in L.editops(r, h):
            if op in ("replace", "delete"):
                err[char_class(r[i])] += 1
                if op == "replace" and r[i] in ACCENTED and strip_acc(r[i]) == strip_acc(h[j]):
                    diacritic_only += 1
    out = {k: err[k] / tot[k] for k in tot}
    out["_diacritic_only_subs"] = diacritic_only
    out["_n"] = dict(tot)
    return out


def strip_acc(s):
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


# ----------------------------------------------------------------------------- Fidelidade ortográfica
_WORD = re.compile(r"[A-Za-zÀ-ÿ]+")


class Lexicon:
    """Léxico do português brasileiro: dicionário Hunspell pt_BR (VERO/LibreOffice), consultado com spylls.
    Uma palavra de referência é classificada como:
      IV  - reconhecida pelo dicionário;
      ERR - provável erro ortográfico do aluno: NÃO reconhecida, toda minúscula e com >= 3 letras
            (ex.: 'excessão', 'impecilho', 'porisso'). Nomes próprios (maiúscula) ficam de fora;
      OOV - demais casos, ignorados.
    Fragmentos de hifenização no início/fim da linha ('con-' / 'ceito') são descartados antes."""

    def __init__(self, path=None):
        from spylls.hunspell import Dictionary
        path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lexicon", "pt_BR")
        self.d = Dictionary.from_files(path)

    @lru_cache(maxsize=None)
    def known(self, w):
        return bool(self.d.lookup(w)) or bool(self.d.lookup(w.lower()))

    @lru_cache(maxsize=None)
    def kind(self, w):
        if self.known(w):
            return "IV"
        if len(w) >= 3 and w.islower():
            return "ERR"
        return "OOV"


def _words(s, drop_fragments=False):
    ws = _WORD.findall(s)
    if drop_fragments and ws:
        # 1a palavra pode ser continuação da linha anterior; última pode ser cortada por hífen
        if s.rstrip().endswith("-"):
            ws[-1] = None
        ws[0] = None
    return ws


def _is_correction(w, p, lex, min_sim=0.6):
    """A saída `p` é uma palavra válida e parecida com a referência `w` (similaridade de Levenshtein
    normalizada >= 0,6, ignorando caixa e acentos). Exclui desalinhamentos triviais como 'filosófo' -> 'o'."""
    if p is None or not lex.known(p):
        return False
    return L.ratio(strip_acc(w).lower(), strip_acc(p).lower()) >= min_sim


def spelling_fidelity(refs, hyps, lex=None, seen=None, min_sim=0.6):
    """Alinha palavras ref/hyp (Levenshtein) e mede, por tipo de palavra de referência:
      acc_IV   - fração de palavras corretas do léxico transcritas exatamente;
      acc_ERR  - fração de erros ortográficos do aluno preservados exatamente;
      overcorr - fração de erros do aluno que viraram uma palavra VÁLIDA e parecida (o modelo 'corrigiu');
    Um modelo fiel deveria ter acc_ERR próximo de acc_IV e overcorr próximo de zero.

    Controle (se `seen`, o vocabulário das transcrições de treino, for dado): palavras válidas que
    NUNCA aparecem no treino (IVu). Erros do aluno também são, quase sempre, inéditos; comparar ERR
    com IVu separa o efeito "palavra rara/inédita" do efeito "palavra fora da norma". Para cada grupo
    medimos ainda P(saída é palavra válida | transcrição errada), base de comparação da sobrecorreção.
    `min_sim` é o limiar de similaridade que define "parecida" (padrão 0,6; ver src/sensibilidade.py)."""
    lex = lex or Lexicon()
    c = Counter()
    examples = []
    err_flags = []
    for r, h in zip(refs, hyps):
        rw, hw = _words(r, drop_fragments=True), _words(h)
        vocab = {w: chr(0x100 + i) for i, w in enumerate(set(rw) | set(hw))}
        rs, hs = "".join(vocab[w] for w in rw), "".join(vocab[w] for w in hw)
        aligned = {i: None for i in range(len(rw))}
        # posições iguais e substituições a partir dos opcodes
        for tag, i1, i2, j1, j2 in L.opcodes(rs, hs):
            if tag == "equal":
                for k in range(i2 - i1):
                    aligned[i1 + k] = hw[j1 + k]
            elif tag == "replace":
                for k in range(min(i2 - i1, j2 - j1)):
                    aligned[i1 + k] = hw[j1 + k]
        for i, w in enumerate(rw):
            if w is None:
                continue
            k = lex.kind(w)
            if k == "OOV":
                continue
            p = aligned[i]
            groups = [k] + (["IVu"] if seen is not None and k == "IV" and w.lower() not in seen else [])
            for g in groups:
                c[f"n_{g}"] += 1
                if p == w:
                    c[f"ok_{g}"] += 1
                else:
                    c[f"wrong_{g}"] += 1
                    if _is_correction(w, p, lex, min_sim):
                        c[f"wrongvalid_{g}"] += 1
            if k == "ERR":
                err_flags.append(int(p != w and _is_correction(w, p, lex, min_sim)))  # mesma ordem em todos os modelos
            if k == "ERR" and p != w and _is_correction(w, p, lex, min_sim):
                c["overcorr"] += 1
                if len(examples) < 30:
                    examples.append((w, p))
    return {
        "acc_IV": c["ok_IV"] / max(1, c["n_IV"]),
        "acc_ERR": c["ok_ERR"] / max(1, c["n_ERR"]),
        "overcorr": c["overcorr"] / max(1, c["n_ERR"]),
        "n_IV": c["n_IV"], "n_ERR": c["n_ERR"],
        "acc_IVu": c["ok_IVu"] / max(1, c["n_IVu"]), "n_IVu": c["n_IVu"],
        "pvalid_wrong_ERR": c["wrongvalid_ERR"] / max(1, c["wrong_ERR"]),
        "pvalid_wrong_IVu": c["wrongvalid_IVu"] / max(1, c["wrong_IVu"]),
        "pvalid_wrong_IV": c["wrongvalid_IV"] / max(1, c["wrong_IV"]),
        "counts": dict(c),  # contagens brutas, para bootstrap por página
        "examples": examples,
        "err_overcorr_flags": err_flags,  # 0/1 por erro do aluno, para o teste de McNemar entre modelos
    }


def full_report(df, lex=None):
    """df com colunas text (ref), pred, has_strike. Retorna dicionário com todas as métricas."""
    refs, hyps = df.text.tolist(), df.pred.tolist()
    rep = {"CER": cer(refs, hyps), "WER": wer(refs, hyps)}
    for flag, name in ((0, "CER_sem_rasura"), (1, "CER_com_rasura")):
        sub = df[df.has_strike == flag]
        if len(sub):
            rep[name] = cer(sub.text.tolist(), sub.pred.tolist())
    rep["per_class"] = per_class_cer(refs, hyps)
    rep["spelling"] = spelling_fidelity(refs, hyps, lex)
    return rep
