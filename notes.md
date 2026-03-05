### Remarks
The following commands in [runs/runcpu.sh](runs/runcpu.sh) are more important than I thought.
```
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
```

Tokenizer step took ~40s.

