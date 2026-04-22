# ADR-0005: Use 8x H100 Cluster for Training Infrastructure

**Date**: 2026-04-22
**Status**: accepted
**Deciders**: SSPO Reproduction Team

## Context

The SSPO paper uses 8x A100 (40GB) or 8x H100 (80GB) for training. Consumer GPUs (GTX 1660 SUPER with 6GB VRAM) cannot train 7B models. UHD Graphics 770 (integrated graphics) lacks CUDA support. An 8x H100 cluster (80GB per GPU) is available for training. The paper's compute estimate is ~2 hours per experiment on 8x A100.

## Decision

We use the 8x H100 cluster for all training experiments. Single-node multi-GPU training with data parallelism.

**Compute Allocation (from paper page 7)**:
| Resource | Specification |
|----------|----------------|
| GPUs | 8x H100 (80GB) or 8x A100 (40GB) |
| Per GPU memory | 80GB (H100) or 40GB (A100) |
| Total memory | 640GB (H100) or 320GB (A100) |
| Training time | ~2 hours per experiment |

**Training Configuration (from paper page 7)**:
- Per device batch size: 4
- Gradient accumulation: 16
- Total batch size: 64 (per node)
- Mixed precision: bf16
- Training steps: ~500 steps (1 epoch)

**Memory Budget**:
| Component | Memory (GB) |
|-----------|-------------|
| Model (7B bf16) | ~14 |
| LoRA adapters | ~1 |
| Optimizer states | ~8 |
| Activations (batch 4) | ~4 |
| **Total per GPU** | **~27** (leaves headroom) |

## Alternatives Considered

### Alternative 1: Consumer GPU with QLoRA (GTX 1660 SUPER)
- **Pros**: Local, no cloud cost
- **Cons**: 6GB VRAM insufficient, would need aggressive quantization
- **Why not**: Cannot meet paper's training configuration

### Alternative 2: Single A100 40GB
- **Pros**: Lower cost
- **Cons**: Must reduce batch size to 2, slower training
- **Why not**: 8x H100 provides optimal throughput for reproduction

### Alternative 3: UHD Graphics 770 (iGPU)
- **Pros**: Available on some Intel CPUs
- **Cons**: No CUDA, ~1.5 TFLOPS FP32, CPU-only training
- **Why not**: Training 7B model would take months

## Training Scripts

```bash
# Single node 8x H100
torchrun --nnodes=1 --nproc_per_node=8 \
    examples/train/train.py \
    --config configs/mistral-7b-sspo.yaml

# SLURM job script
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:8
#SBATCH --time=24:00:00
```

## Consequences

### Positive
- Sufficient VRAM: 80GB per GPU allows full LoRA training without offloading
- Fast iteration: Multi-GPU data parallelism reduces training time to ~2 hours
- Flexibility: Can run multiple experiments in parallel

### Negative
- Cluster scheduling required: Need to request job allocation
- Cost: H100 cluster has associated compute cost

### Risks
- **Risk**: Cluster availability conflicts
- **Mitigation**: Use SLURM job scheduling, plan experiments in advance
- **Risk**: Different H100 vs A100 performance
- **Mitigation**: Paper uses A100, H100 is faster so no issue
