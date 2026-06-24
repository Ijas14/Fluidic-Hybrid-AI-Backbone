# Fluidic ROCm Backbone

A compact hybrid language-model backbone that combines a parallel state-space memory path with closed-form continuous-time recurrence. The project targets memory-efficient sequence modeling and includes training, inference, and NF4 quantization tooling, with practical orientation toward AMD/ROCm-capable environments.

## Overview

Fluidic ROCm Backbone is a research-oriented PyTorch codebase for experimenting with a hybrid sequence architecture built from:

- a **Liquid-S4 style state-space recurrence** (`LiquidS4StateLoop`), and
- a **CfC (Closed-form Continuous-time) recurrent module** with DEQ-style training flow (`DEQCfCSequenceProcessor`).

The model is implemented in `model.py` and `core_modules.py`, trained via `train.py`, and sampled through `inference.py` / `test.py`. The repository also includes data preparation and quantization utilities under `tools/`.

## Key Features

- **Hybrid sequence backbone**: combines SSM-style memory and CfC dynamics in each block.
- **Stateful autoregressive inference**: maintains per-layer SSM/CfC states during generation.
- **Memory-mapped dataset pipeline**: `tools/prepare_data.py` compiles corpus data into `numpy.memmap` binaries.
- **Mixed-precision training path**: `train.py` uses AMP (`torch.cuda.amp`) and gradient accumulation.
- **NF4 quantization prototype**: `quantization.py` and `core_modules.py` include a Triton-based fused kernel path.
- **Distillation workflow scaffold**: `distill.py` provides teacher-logit caching and student distillation flow.

## Architecture

At a high level, token IDs pass through embedding, multiple hybrid blocks, final normalization, and an LM head:

1. **Embedding** (`FluidicHybridBackbone.embed`)
2. **Repeated `FluidicBlock` layers**
   - LayerNorm
   - `LiquidS4StateLoop` (parallel recurrence)
   - `DEQCfCSequenceProcessor` over `CfCCell`
   - Linear projection + residual connection
3. **Final LayerNorm + LM projection** (`lm_head`)

Core implementation files:

- `model.py`
- `core_modules.py`

## Repository Structure

```text
<repository-root>/
├── model.py
├── core_modules.py
├── quantization.py
├── train.py
├── inference.py
├── test.py
├── distill.py
├── data/
│   └── tokenizer.json
├── tools/
│   ├── prepare_data.py
│   ├── nf4_compress.py
│   └── assemble_quantized.py
└── proof/
    └── training/
```

## Requirements

Minimum practical requirements (based on current scripts):

- Python 3.10+
- PyTorch (CUDA/ROCm-enabled build recommended for GPU paths)
- NumPy
- `tokenizers`
- `datasets`
- `triton` (for NF4 kernel path)
- Optional for distillation/quantization helpers: `transformers`, `quark`

## Installation

```bash
git clone https://github.com/Ijas14/fluidic-rocm-backbone.git
cd fluidic-rocm-backbone

python -m venv .venv
source .venv/bin/activate
pip install torch numpy tokenizers datasets triton transformers
```

If the GitHub repository has not been renamed yet, use:

```bash
git clone https://github.com/Ijas14/Fluidic-Hybrid-AI-Backbone.git
cd Fluidic-Hybrid-AI-Backbone
```

## Training

1. Prepare dataset/tokenizer artifacts:

```bash
python tools/prepare_data.py
```

2. Move generated files into `data/` if needed so `train.py` can find:

- `data/train_corpus_16k.bin`
- `data/tokenizer.json`

3. Run training:

```bash
python train.py
```

By default, checkpoints are written under `models/v2_17.3M/checkpoints/` and final weights to `models/v2_17.3M/final.pt`.

## Inference

Run standard generation script:

```bash
python inference.py
```

Or use the interactive checkpoint tester:

```bash
python test.py --tokens 50 --temperature 0.8 --top_k 40
```

Both scripts expect a tokenizer at `data/tokenizer.json` and model weights in `models/`.

## Notes and Limitations

- This repository is a **research/prototype codebase**, not a production-serving stack.
- Several scripts assume GPU availability and call CUDA paths directly; CPU fallback is partial.
- The Triton NF4 path is experimental and should be validated on your hardware/software stack.
- Distillation and Quark tooling require additional dependencies and local model assets.
- Existing benchmark/proof artifacts are kept under `proof/` and may not reflect all environments.

## License

MIT License. See [LICENSE](LICENSE).
