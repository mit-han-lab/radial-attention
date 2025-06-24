# Radial Attention: $\mathcal{O}(n\log n)$ Sparse Attention with Energy Decay for Long Video Generation

## 🔥News🔥

- [2025/06] Radial Attention is open-sourced! Wan2.1-14B, HunyuanVideo, and Mochi-1 are supported for fast video generation with high quality under 1-4⨉ video length.

## 📖About

**Radial Attention** is a **scalable sparse attention mechanism** for video diffusion models that translates **Spatiotemporal Energy Decay**—observed in attention score distributions—into exponentially decaying compute density. Unlike dense attention ($\mathcal{O}(n^2)$) or linear approximations, Radial Attention achieves **$\mathcal{O}(n \log n)$ complexity** while preserving expressive power for long videos. Here are our core contributions.

- **Physics-Inspired Sparsity**: Static masks enforce *spatially local* and *temporally decaying* attention, mirroring energy dissipation in physical systems.
- **Efficient Length Extension**: Pre-trained models (e.g., Wan2.1-14B, HunyuanVideo) scale to **4× longer videos** via lightweight LoRA tuning, avoiding full-model retraining. 

## 🔧Installation

We start with cloning the repository:

```bash
git clone git@github.com:mit-han-lab/radial-attention --recursive
cd radial-attention
```

We recommend using CUDA versions 12.4 + Pytorch versions 2.5.1

```bash
# 1. Create and activate conda environment
conda create -n radial python==3.12 -y
conda activate radial

# 2. Install PyTorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Install pip dependencies from CogVideoX and HunyuanVideo
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# 4. Install FlashInfer for fast and hardware-friendly inference
pip install flashinfer-python -i https://flashinfer.ai/whl/cu124/torch2.5/
```

## 🚀Inference Examples

### Wan2.1-14B

We support Text-to-Video inference of Wan2.1-14B. The running script is:

```bash
bash scripts/wan_t2v_inference.sh
```

### HunyuanVideo

We support Text-to-Video inference of HunyuanVideo. The running script is:

```bash
bash scripts/hunyuan_t2v_inference.sh
```

📕Open-source Plan

- [ ] Support Mochi-1
- [ ] Support Multi-GPU inference
- [ ] Release LoRA checkpoints for longer-video generation

## Acknowledgements

We thank [Sparse-VideoGen](https://github.com/svg-project/Sparse-VideoGen/tree/main) for insights on code design.