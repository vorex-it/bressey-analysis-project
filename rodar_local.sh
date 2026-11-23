#!/usr/bin/env bash
# Roda o projeto inteiro localmente (Mac com Apple Silicon usa a GPU via MPS automaticamente).
#
#   bash rodar_local.sh setup        # cria o ambiente Python e instala dependências (1x)
#   bash rodar_local.sh bench        # mede o tempo por época de cada modelo (~10 min)
#   bash rodar_local.sh principal    # treina os 4 modelos (+ TrOCR zero-shot)
#   bash rodar_local.sh ctc          # só CRNN, HTR-Flor e HTR-VT
#   bash rodar_local.sh trocr        # só TrOCR (zero-shot + fine-tuning)
#   bash rodar_local.sh curva        # curva de aprendizado (10%, 25% e 50% do treino; mude com FRACS="0.1 0.5")
#   bash rodar_local.sh analise      # tabelas e figuras em $RUNS/_results
#
# Variáveis opcionais: FRACS, MODELS (curva), EPOCHS (CTC, padrão 30), TROCR_EPOCHS (padrão 6), DATA, RUNS, WORKERS
set -euo pipefail
cd "$(dirname "$0")"

DATA="${DATA:-$HOME/Downloads/bressay_lines}"
RUNS="${RUNS:-$HOME/Downloads/htr_runs}"
EPOCHS="${EPOCHS:-30}"
TROCR_EPOCHS="${TROCR_EPOCHS:-6}"
WORKERS="${WORKERS:-4}"
PY=".venv/bin/python"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$RUNS/logs"

# impede o Mac de dormir enquanto o script roda
if command -v caffeinate >/dev/null 2>&1 && [ -z "${CAFFEINATED:-}" ] && [ "${1:-}" != "setup" ]; then
  export CAFFEINATED=1; exec caffeinate -dims bash "$0" "$@"
fi

run() {  # run <nome_do_run> <comando...>
  local name="$1"; shift
  if [ -f "$RUNS/$name/report.json" ]; then echo "== $name: já concluído, pulando"; return; fi
  echo "== $name: iniciando ($(date '+%H:%M'))  log: $RUNS/logs/$name.log"
  if ! "$@" 2>&1 | grep --line-buffered -v -i -E "warning|warn\(|deprecat" | tee "$RUNS/logs/$name.log"; then
    echo "!! $name falhou; veja $RUNS/logs/$name.log. Seguindo para o próximo."
  fi
}

case "${1:-}" in
  setup)
    python3 -m venv .venv
    $PY -m pip install -q --upgrade pip
    $PY -m pip install -q -r requirements.txt
    $PY -c "import torch; print('torch', torch.__version__, '| MPS (GPU Apple):', torch.backends.mps.is_available(), '| CUDA:', torch.cuda.is_available())"
    [ -f "$DATA/manifest.csv" ] && echo "dados OK em $DATA" || echo "ATENÇÃO: descompacte bressay_lines.zip em $DATA"
    ;;
  bench)
    for m in flor crnn htrvt; do
      $PY -u -m src.train_ctc --model $m --data "$DATA" --out /tmp/htr_bench --epochs 1 --max_train 640 --max_eval 64 --workers "$WORKERS" 2>&1 | grep -E "params|ep |TESTE"
    done
    $PY -u -m src.train_trocr --data "$DATA" --out /tmp/htr_bench --epochs 1 --max_train 320 --max_eval 32 --val_subset 32 --beams 1 --workers "$WORKERS" 2>&1 | grep -E "params|ep |TESTE|Error|error"
    echo "Multiplique o tempo de 'ep 1' por ~30 (CTC: 19.224/640 linhas) e por ~60 (TrOCR: 19.224/320) para estimar 1 época completa."
    ;;
  ctc)
    for m in flor crnn htrvt; do
      run "${m}_frac1" $PY -u -m src.train_ctc --model $m --data "$DATA" --out "$RUNS" --epochs "$EPOCHS" --workers "$WORKERS"
    done
    ;;
  trocr)
    run trocr_zeroshot $PY -u -m src.train_trocr --data "$DATA" --out "$RUNS" --zero_shot --workers "$WORKERS"
    run trocr_frac1   $PY -u -m src.train_trocr --data "$DATA" --out "$RUNS" --epochs "$TROCR_EPOCHS" --workers "$WORKERS"
    ;;
  principal)
    for m in flor crnn htrvt; do
      run "${m}_frac1" $PY -u -m src.train_ctc --model $m --data "$DATA" --out "$RUNS" --epochs "$EPOCHS" --workers "$WORKERS"
    done
    run trocr_zeroshot $PY -u -m src.train_trocr --data "$DATA" --out "$RUNS" --zero_shot --workers "$WORKERS"
    run trocr_frac1   $PY -u -m src.train_trocr --data "$DATA" --out "$RUNS" --epochs "$TROCR_EPOCHS" --workers "$WORKERS"
    ;;
  curva)
    for f in ${FRACS:-0.1 0.25 0.5}; do
      for m in ${MODELS:-flor crnn htrvt trocr}; do
        [ "$m" = trocr ] && continue
        run "${m}_frac${f}" $PY -u -m src.train_ctc --model $m --data "$DATA" --out "$RUNS" --epochs "$EPOCHS" --frac $f --workers "$WORKERS"
      done
      if [[ " ${MODELS:-trocr} " == *" trocr "* ]]; then
        run "trocr_frac${f}" $PY -u -m src.train_trocr --data "$DATA" --out "$RUNS" --epochs "$TROCR_EPOCHS" --frac $f --workers "$WORKERS"
      fi
    done
    ;;
  analise)
    $PY -m src.analyze --runs "$RUNS" --out "$RUNS/_results" --data "$DATA"
    echo "Resultados em $RUNS/_results"
    ;;
  *)
    sed -n '2,10p' "$0"; exit 1 ;;
esac
