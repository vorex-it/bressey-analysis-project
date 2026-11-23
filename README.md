# Transcrever ou corrigir? HTR em redações escolares brasileiras (BRESSAY)

Análise comparativa de quatro arquiteturas de reconhecimento de texto manuscrito (HTR) em linhas de
redações do [BRESSAY](https://zenodo.org/records/11637681), com uma métrica proposta de
**fidelidade ortográfica**: o modelo transcreve o que o aluno escreveu ou "corrige" o aluno?

Projeto da disciplina de Redes Neurais (mestrado, PPGIA/UFRPE). Artigo em [`artigo/`](artigo/).

## Resultados (teste, 5.863 linhas)

| Modelo | Parâm. | CER (%) [IC 95%] | WER (%) | Erros do aluno preservados (%) | Sobrecorreção (%) | CPU ms/linha |
|---|---|---|---|---|---|---|
| CRNN | 10,1M | **3,89** [3,44–4,35] | **15,67** | **26,9** | 30,9 | 289 |
| HTR-Flor | 1,0M | 5,76 [5,19–6,37] | 22,76 | 26,4 | 19,4 | **82** |
| HTR-VT (reduzido) | 5,9M | 11,11 [10,30–11,93] | 41,54 | 20,1 | **7,7** | 163 |
| TrOCR-small (fine-tuning) | 61,6M | 6,43 [5,86–7,02] | 18,38 | 16,3 | 37,6 | 2.109 |
| TrOCR-small (zero-shot) | 61,6M | 68,26 | 110,71 | 1,0 | 1,9 | 1.561 |

A coluna de sobrecorreção é uma taxa absoluta; o viés relativo ao grupo de controle (razão de viés) está no artigo e em `resultados/razao_vies.json`. Resultados de uma única semente por modelo; ver limitações no artigo.

Tabelas completas, figuras e exemplos em [`resultados/`](resultados/); métricas e histórico de treino por modelo em `resultados/runs/`. As predições linha a linha não são publicadas porque contêm as transcrições do BRESSAY; são regeneradas pelos scripts de treino.

## Modelos

| Modelo | Paradigma | Implementação |
|---|---|---|
| CRNN (Shi et al., 2017) | CNN + BiLSTM + CTC | `src/models.py` |
| HTR-Flor (Neto et al., 2020) | convoluções com porta + BiGRU + CTC | `src/models.py` (reimplementação em PyTorch) |
| HTR-VT (Li et al., 2025) | CNN + encoder ViT + span masking + SAM + CTC | `src/models.py` (versão reduzida, d=256) |
| TrOCR-small (Li et al., 2023) | encoder ViT + decoder Transformer pré-treinados | HuggingFace `microsoft/trocr-small-handwritten` |

## Estrutura

```
dataset_bressey/           bressay_lines.zip gerado localmente (ignorado pelo git; ver Dados)
prepare_data.py            gera bressay_lines.zip (linhas + manifest.csv) a partir do BRESSAY original
src/data.py                carregamento, normalização 40x1280, aumento de dados
src/models.py              CRNN, HTR-Flor, HTR-VT, otimizador SAM
src/metrics.py             CER, WER, CER por classe, fidelidade ortográfica (com controle de palavras inéditas)
src/train_ctc.py           treino/avaliação dos modelos CTC
src/train_trocr.py         fine-tuning/avaliação do TrOCR
src/analyze.py             tabelas, IC por bootstrap, teste pareado, figuras
src/bias_ratio.py          razão de viés lexical (IC por bootstrap de páginas) e McNemar entre modelos
src/fig_exemplos.py        figura de exemplos (linhas manuscritas, alvo e saída dos modelos)
src/curva_tabela.py        tabela da curva de aprendizado (CER e fidelidade por fração de treino)
src/sensibilidade.py       sensibilidade da sobrecorreção e da razão de viés ao limiar de similaridade
src/device.py              escolha automática de GPU (CUDA / Apple MPS) ou CPU
lexicon/pt_BR.{dic,aff}    dicionário Hunspell pt_BR (VERO/LibreOffice, LGPLv3/MPL)
notebooks/                 roteiro completo para o Google Colab
rodar_local.sh             roteiro para rodar localmente (Mac/Linux)
artigo/                    artigo em LaTeX (IEEEtran) + figuras
resultados/                tabelas, figuras, métricas e históricos dos experimentos
```

## Dados (`dataset_bressey/`)

> **O dataset não faz parte deste repositório.** A pasta `dataset_bressey/` está no `.gitignore`: quem clonar o
> projeto precisa baixar o BRESSAY na fonte oficial e gerar o pacote de linhas localmente antes de treinar ou avaliar.

### Como obter

1. Baixe o BRESSAY no [Zenodo](https://zenodo.org/records/11637681) (Neto et al., ICDAR 2024, CC BY 4.0) e
   descompacte-o, por exemplo, em `~/Downloads/bressay`.
2. Gere o pacote de linhas dentro de `dataset_bressey/`:
   ```
   mkdir -p dataset_bressey
   python prepare_data.py --bressay ~/Downloads/bressay --out dataset_bressey/bressay_lines.zip
   ```
3. Descompacte o pacote para uso local:
   ```
   unzip dataset_bressey/bressay_lines.zip -d dataset_bressey/bressay_lines
   ```

### Conteúdo

`bressay_lines.zip` (≈ 290 MB) contém 30.090 imagens PNG em `lines/` (uma por linha manuscrita, com o nome original do BRESSAY, sem
reamostragem) e um `manifest.csv` na raiz, com uma linha por imagem:

| Coluna | Conteúdo |
|---|---|
| `id` | identificador da linha (= nome do PNG sem extensão) |
| `page` | página/redação de origem (1.000 páginas) |
| `split` | `train`, `val` ou `test` (partição oficial do BRESSAY) |
| `text` | alvo usado no treino e na avaliação (texto final do aluno, com os erros ortográficos dele) |
| `raw` | transcrição original com as marcações do BRESSAY (`--...--` riscado, `##...##`/`$$...$$` inserções) |
| `has_strike`, `has_insert`, `illegible` | 1 se a linha tem trecho riscado, inserção ou trecho ilegível |
| `width`, `height` | dimensões da imagem em pixels (mediana 486 × 21) |

| Partição | Páginas | Linhas | Ilegíveis (excluídas) | Linhas usadas | Com riscado | Com inserção |
|---|---|---|---|---|---|---|
| Treino | 647 | 19.565 | 338 | 19.227 | 2.377 | 877 |
| Validação | 154 | 4.609 | 55 | 4.554 | 495 | 148 |
| Teste | 199 | 5.916 | 52 | 5.864 | 724 | 347 |
| **Total** | **1.000** | **30.090** | **445** | **29.645** | **3.596** | **1.372** |

As linhas marcadas como ilegíveis são descartadas por `src/data.py` (`load_manifest`). Os alvos têm em média
87 caracteres por linha.

Os scripts recebem a pasta descompactada por `DATA` (ou `--data`), por exemplo
`DATA=dataset_bressey/bressay_lines bash rodar_local.sh setup`.

O zip e a pasta descompactada não vão para o GitHub por dois motivos: o arquivo passa do limite de 100 MB por
arquivo, e o repositório não redistribui as imagens do BRESSAY (ver *Dados e licenças*).

## Como reproduzir

1. Obtenha o BRESSAY na fonte e gere `dataset_bressey/bressay_lines.zip` (ver *Dados → Como obter*).
2. **Colab:** envie `bressay_lines.zip` para `MyDrive/` e este repositório para `MyDrive/bressay-htr/`;
   abra `notebooks/projeto_bressay_htr.ipynb` com GPU e execute as células.
3. **Local:** descompacte o zip (ver *Dados*) e rode `bash rodar_local.sh setup`, depois
   `bash rodar_local.sh principal` e `bash rodar_local.sh analise`, com `DATA=dataset_bressey/bressay_lines`
   (o padrão do script é `~/Downloads/bressay_lines`).

### Curva de aprendizado

Treina cada modelo com 10%, 25% e 50% das páginas de treino (escritores inteiros, sorteio fixo) e
avalia no mesmo teste. A análise gera `curva_aprendizado.png` automaticamente.

- Colab: seção 3 do notebook (`FRACS` e `MODELS` controlam o que roda).
- Local: `bash rodar_local.sh curva` (ex.: `FRACS="0.1 0.5" MODELS="flor crnn" bash rodar_local.sh curva`).
- Depois: `python -m src.analyze --runs <pasta_dos_runs> --out <pasta>/_results --data <pasta_bressay_lines>`
  e `python -m src.curva_tabela --runs <pasta_dos_runs> --data <pasta_bressay_lines>` (CER, WER, sobrecorreção e razão de viés por fração; saída em `resultados/curva_tabela.csv`).

Resultado (uma semente por célula): TrOCR e CRNN não diferem de forma distinguível com 10% das páginas (CER 15,9% e 16,5%; p = 0,066), o CRNN lidera a partir de 25% (9,6% contra 11,4%), e a razão de viés do CRNN e do HTR-Flor cresce com o volume de dados (CRNN: 1,22 com 10% e 2,04 com 100%), o que o HTR-VT não repete. Detalhes na Seção IV-D do artigo.

## Protocolo

- Partição oficial do BRESSAY (treino 647 / validação 154 / teste 199 páginas; escritores disjuntos).
- Alvo = texto final do aluno: trechos riscados removidos, inserções mantidas, linhas ilegíveis excluídas;
  erros ortográficos do aluno preservados.
- Mesmo aumento de dados para todos; melhor checkpoint pela CER de validação; CTC: 40 épocas, decodificação
  gulosa sem modelo de linguagem; TrOCR: 8 épocas, beam search 4. GPU NVIDIA T4 (Colab).

## Dados e licenças

Código sob licença MIT (ver `LICENSE`). BRESSAY: Neto et al., ICDAR 2024 — disponível no [Zenodo](https://zenodo.org/records/11637681) sob licença CC BY 4.0; as imagens
não são redistribuídas aqui. Dicionário pt_BR: projeto VERO (LibreOffice), LGPLv3/MPL.

### Sensibilidade ao limiar

`python -m src.sensibilidade --runs <pasta_dos_runs> --data <pasta_bressay_lines>` recalcula a sobrecorreção e a razão de viés com limiares de similaridade de 0 a 0,8 e a diferença de razão entre pares de modelos (bootstrap pareado por página). A ordem da sobrecorreção absoluta não muda; a ordem entre CRNN e TrOCR na razão de viés muda com o limiar 0,8. Saídas em `resultados/sensibilidade*.csv`.
