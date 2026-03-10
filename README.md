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

Reference project inspiration: [Magpie](https://github.com/magpie-align/magpie)

## CLI Guide (Model, Turns, Reasoning)

You can keep domain settings in a config file and override runtime behavior from the command line.

### 1) Base command

```bash
python run.py --config <config_path> --samples <N>
```

Example:

```bash
python run.py --config configs/coding_template.json --samples 200
```

### 2) Model used

This repo is pinned to `Qwen/Qwen2.5-0.5B-Instruct` for generation.

### 3) Single-turn generation

Single-turn outputs `instruction`, `response` (and optional `reasoning`).

```bash
python run.py --config configs/coding_template.json --samples 100 --turn-mode single
```

### 4) Multi-turn generation (fixed turns)

Multi-turn outputs `turn_count`, `instruction_1..N`, `response_1..N` (and optional `reasoning_1..N`).

Exactly 2 turns:

```bash
python run.py --config configs/coding_template.json --samples 100 --turn-mode multi --min-turns 2 --max-turns 2
```

Exactly 4 turns:

```bash
python run.py --config configs/coding_template.json --samples 100 --turn-mode multi --min-turns 4 --max-turns 4
```

### 5) Multi-turn generation (range of turns)

Each row gets a random turn count in `[min_turns, max_turns]`.

```bash
python run.py --config configs/coding_template.json --samples 120 --turn-mode multi --min-turns 2 --max-turns 5
```

### 6) Reasoning on/off

Enable reasoning:

```bash
python run.py --config configs/math_template.json --samples 80 --reasoning
```

Disable reasoning:

```bash
python run.py --config configs/math_template.json --samples 80 --no-reasoning
```

Works with multi-turn too (adds `reasoning_1..N`):

```bash
python run.py --config configs/math_template.json --samples 60 --turn-mode multi --min-turns 2 --max-turns 3 --reasoning
```

### 7) Full example (all requested controls together)

```bash
python run.py --config configs/coding_template.json --samples 150 --turn-mode multi --min-turns 3 --max-turns 5 --reasoning --output-dir datasets
```

### 7.1) Custom output filename and folder

If `--output-name` contains folders, they are created automatically under `--output-dir`.

```bash
python run.py --config configs/coding_template.json --samples 100 --output-dir datasets --output-name coding/single/run_001
```

Artifacts created:

- `datasets/coding/single/run_001.csv`
- `datasets/coding/single/run_001_tagged.csv`
- `datasets/coding/single/run_001_deduplicated.csv`
- `datasets/coding/single/run_001_report.json`
- `datasets/coding/single/run_001_report.md`

### 8) Randomness and reproducibility

Different output every run (default behavior):

```bash
python run.py --config configs/coding_template.json --samples 120 --turn-mode multi --min-turns 2 --max-turns 5
```

Reproducible output with a fixed seed:

```bash
python run.py --config configs/coding_template.json --samples 120 --turn-mode multi --min-turns 2 --max-turns 5 --seed 42
```

### 9) Open-ended generation (no manual language/task/context list needed)

If `domain_structure` in config is `{}`, the model chooses language/task/context/difficulty naturally on its own.

```bash
python run.py --config configs/coding_template.json --samples 200
```

### 9.1) Batch examples for common scenarios

Coding, single-turn:

```bash
python run.py --config configs/coding_template.json --samples 100 --turn-mode single --output-name coding/single/s100
```

Coding, multi-turn 3-6:

```bash
python run.py --config configs/coding_template.json --samples 100 --turn-mode multi --min-turns 3 --max-turns 6 --output-name coding/multi/s100
```

Math, single-turn:

```bash
python run.py --config configs/math_template.json --samples 100 --turn-mode single --output-name math/single/s100
```

Math, multi-turn 3-6:

```bash
python run.py --config configs/math_template.json --samples 100 --turn-mode multi --min-turns 3 --max-turns 6 --output-name math/multi/s100
```

### 10) Multi-turn relation guarantee

Follow-up turns are generated from the full previous conversation history plus the same domain seed.  
That means each next user instruction is explicitly prompted to deepen/extend/debug/optimize/test the prior turns, rather than starting a new unrelated topic.

## Output files

- `datasets/<domain>_dataset.csv`
- `datasets/<domain>_dataset_tagged.csv`
- `datasets/<domain>_dataset_deduplicated.csv`
- `datasets/<domain>_dataset_report.json`
- `datasets/<domain>_dataset_report.md`

## Notes

- Model is fixed to Qwen 2.5 0.5B Instruct in runtime (`run.py`).
- For gated models, authenticate with Hugging Face (`huggingface-cli login`).
