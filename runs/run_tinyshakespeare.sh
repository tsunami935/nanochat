export NANOCHAT_BASE_DIR=".cache/tinyshakespeare"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

python -m tinyshakespeare.tok_train --vocab-size=16384
python -m tinyshakespeare.tok_eval

python -m tinyshakespeare.base_train \
    --depth=4 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=256 \
    --device-batch-size=4 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=262144 \
    --sample-every=100 \
    --save-every=200 \
    --num-iterations=1600 \
    --dir=".cache/tinyshakespeare"
    