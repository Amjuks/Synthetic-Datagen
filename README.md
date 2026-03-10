# Synthetic-Datagen (Magpie-style)

Lightweight, modular synthetic dataset generator for coding and math domains with:

- Domain-specialized seed sampling (not just a generic role prompt)
- Magpie-style two-stage generation (instruction -> response)
- Optional multi-turn and reasoning columns
- Automatic hardware detection (GPU/CPU)
- Automatic vLLM usage when installed, fallback to transformers
- Tagging and semantic deduplication
- CSV outputs with checkpointing + resume

## Quickstart

```bash
pip install -r requirements.txt
python run.py --config configs/coding_template.json --samples 100
```

## Output files

- `datasets/<domain>_dataset.csv`
- `datasets/<domain>_dataset_tagged.csv`
- `datasets/<domain>_dataset_deduplicated.csv`

## Notes

- Default models in config are Llama 3 8B Instruct. You can set 70B in config if your hardware supports it.
- For gated models, authenticate with Hugging Face (`huggingface-cli login`).
