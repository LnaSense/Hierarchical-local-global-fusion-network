# Hierarchical Local-Global Fusion Network for Large-Range Physically Consistent Force-Map Reconstruction

This repository contains the official implementation of the paper:

**Physically consistent force-map reconstruction with the fusion of local estimation and global-constrained refinement in large-range contact-rich manipulation**

## Overview

Reconstructing physically consistent force maps from tactile observations is challenging, especially under large-range loading conditions, where nonlinear deformation and contact ambiguity become more significant. To address this problem, we propose a hierarchical local-global fusion framework for reconstructing large-range physically consistent force maps.

The proposed framework combines:
- local estimation that captures and estimates fine-scale tactile deformation cues
- global-constrained refinement that improves the physical consistency of the reconstructed force map under wide-range contact conditions.

This repository includes:
- code for the forward MLP model,
- code for FMR-Net,
- access information for the LaraS2R dataset.

This project is developed in a Conda environment, using Python 3.10, PyTorch 2.9.1, and CUDA 12.6. Detailed environment configuration can be found in `environment.yml`.

## Code

- `Forward_MLP.py`: script for training the forward MLP model.
- `FMR-Net.py`: script for training the proposed force-map refinement network (FMR-Net).

## Dataset

We also release LaraS2R, a large-range Sim2Real-paired dataset constructed for force-map reconstruction. This dataset supports:

- training of the proposed hierarchical framework,
- quantitative evaluation,
- comparison with baseline methods,
- future research on large-range physically consistent force-map reconstruction.

The LaraS2R dataset is available at:

https://github.com/LnaSense/Hierarchical-local-global-fusion-network/releases/tag/v1.0

## Citation

If you find this repository useful in your research, please cite the corresponding paper.
