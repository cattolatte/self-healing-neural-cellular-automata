# Self-Healing Neural Cellular Automata

> A PyTorch implementation of a Self-Healing Neural Cellular Automata (NCA) that learns morphogenesis, tissue regeneration, and damage recovery through decentralized local interactions.

## Overview

This project explores **Neural Cellular Automata (NCA)** — a class of trainable dynamical systems that replace hand-crafted cellular automata rules with a small neural network. Unlike traditional Cellular Automata (e.g., Conway's Game of Life), where every update rule is manually designed, Neural Cellular Automata learn local update rules through **gradient descent**.

The objective of this project is to demonstrate how a distributed collection of locally interacting cells can learn to:
- Grow from a single seed
- Form a stable biological tissue
- Maintain structural integrity over time
- Regenerate after physical damage (Self-Healing)

## Motivation

Biological organisms exhibit remarkable regenerative abilities without relying on a centralized controller. Every individual cell only communicates with its immediate neighbors, yet collectively they produce complex behaviors such as:
- Tissue growth
- Morphogenesis
- Homeostasis
- Wound healing

Neural Cellular Automata attempt to model these decentralized behaviors using trainable neural networks.

## Project Objectives

The model should learn to:
- Grow from a single seed cell
- Form a predefined tissue structure
- Preserve that structure over many generations
- Recover from simulated wounds
- Learn all behaviors through optimization rather than manually designed rules

## Core Architecture

Each cell in the simulation contains a **16-dimensional state vector** on a default **32 × 32** grid.

### Visible Channels

| Channel | Meaning |
|----------|----------|
| 0 | Epidermis |
| 1 | Dermis |
| 2 | Vasculature |

These three channels form the visible RGB target representation.

### Hidden Channels

Channels **3–15** store latent information learned during training. These hidden states act as internal communication signals that enable decentralized coordination between neighboring cells.

## Model Pipeline

```text
                  Seed Cell
                       ↓
Perception Layer (Identity + Sobel X + Sobel Y)
                       ↓
    Shared Neural Network (1x1 Convolutions)
                       ↓
           Stochastic Residual Update
                       ↓
                    Alive Mask
                       ↓
                Updated Cell State
```

Every cell executes the same neural network independently.

## Implemented Infrastructure

* YAML configuration loading with immutable nested values
* CPU/CUDA/Apple MPS device selection
* Python, NumPy, PyTorch, and CUDA seeding
* Reusable logging and filesystem utilities
* PNG target loading, RGB conversion, resizing, normalization, and validation
* Canonical seed/state initialization and state validation
* Identity, Sobel X, and Sobel Y local perception
* Shared 1x1 convolutional update rule
* Residual updates, stochastic asynchronous updates, and alive masking
* BPTT rollout training with MSE reconstruction loss
* Adam optimizer and optional learning-rate scheduling
* Sample pooling and checkpoint save/load
* **Visualization Suite:** RGB mapping, GIF generation, MP4 encoding (via ImageIO), and raw frame exports
* **Evaluation Pipeline:** Standardized quantitative benchmarks for Growth, Structural Persistence, and Self-Healing
* **CLI Orchestration:** Thin executable scripts for training, testing, and rendering

## Project Structure

```text
self-healing-neural-cellular-automata/
├── checkpoints/       # Ignored model artifacts
├── configs/           # YAML experiment configuration
├── docs/              # Project documentation
├── outputs/           # Ignored runtime artifacts
├── scripts/           # Command-line entry points
├── src/
│   ├── data/          # Target loading and preprocessing
│   ├── model/         # NCA computational core
│   ├── simulation/    # Seed/state initialization and rollout logic
│   ├── training/      # Training pipeline and sample pooling
│   ├── visualization/ # RGB Rendering and media exports
│   └── utils/         # Shared infrastructure
├── tests/             # Fast, isolated unit tests
├── PROJECT_CONTEXT.md # Canonical workspace-local specification
└── README.md

```

## Technology Stack

### Language

* Python 3.11+

### Deep Learning

* PyTorch

### Libraries

* NumPy
* Torchvision
* Pillow
* Matplotlib
* ImageIO
* OpenCV
* SciPy
* tqdm
* PyYAML

## Configuration

Use the YAML files in `configs/` as the source of truth for model shape, rollout length, batch size, optimizer, loss, scheduler, pool, paths, seed, and logging settings. Resolved configurations are preserved with experiment artifacts.

---

## Usage

This project exposes several command-line interfaces (CLIs) via the `scripts/` directory to run training, evaluation, and rendering tasks.

### 1. Training

To train the model from scratch using the default target and configuration:

```bash
python -m scripts.train --config configs/train.yaml --target assets/targets/your_target.png

```

### 2. Evaluation

To run the full diagnostic suite (Growth -> Persistence -> Healing) and output the `evaluation_metrics.json` report:

```bash
python -m scripts.evaluate \
    --config configs/eval.yaml \
    --checkpoint checkpoints/model_epoch_100.pt \
    --target assets/targets/your_target.png

```

### 3. Rendering Animations

You can export animations as either `.gif` or `.mp4` files. You must specify whether you want to visualize the `growth` or `healing` scenario.

**Render an MP4 Video:**

```bash
python -m scripts.render_video \
    --config configs/eval.yaml \
    --checkpoint checkpoints/model_epoch_100.pt \
    --scenario healing

```

**Render an Animated GIF:**

```bash
python -m scripts.render_gif \
    --config configs/eval.yaml \
    --checkpoint checkpoints/model_epoch_100.pt \
    --scenario growth

```

### 4. Generating Figures

To export publication-ready matplotlib figures evaluating the model's structural recovery capabilities:

```bash
python -m scripts.export_figures \
    --config configs/eval.yaml \
    --checkpoint checkpoints/model_epoch_100.pt \
    --target assets/targets/your_target.png

```

### Running Tests

To run the isolated unit testing suite to verify architectural invariants:

```bash
pytest tests/

```

---

## Expected Results

The final model successfully demonstrates:

- Growth from a single seed into a structured pattern
- Stable tissue persistence over long unrolled horizons (1,000+ steps)
- Recovery and regeneration after simulated geometric injuries (circle, square, half-grid masks)
- Smooth decentralized regeneration
- Decreasing reconstruction loss during BPTT training

## Future Improvements

Potential extensions include:

- Multiple tissue types
- Higher-resolution grids
- Learned perception filters
- 3D Neural Cellular Automata
- Interactive simulation interface
- Advanced evaluation metrics

## References

- Mordvintsev, A., Randazzo, E., Niklasson, E., & Levin, M. *Growing Neural Cellular Automata*.
- Distill: *Growing Neural Cellular Automata*
- PyTorch Documentation

## License

This project is intended for educational and research purposes.

## Status

**Complete.** Milestones 1–11 are fully implemented, establishing a robust end-to-end framework encompassing configuration, data processing, model dynamics, BPTT training, media visualization, quantitative evaluation, and hermetic unit testing.
