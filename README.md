# Hierarchical Local-Global Fusion Network for Large-Range Physically Consistent Force-Map Reconstruction

This repository contains the official implementation of the paper:

**Physically consistent force-map reconstruction with the fusion of local estimation and global-constrained refinement in large-range contact-rich manipulation**

## Overview

Reconstructing physically consistent force maps from tactile observations is challenging, especially under large-range loading conditions, where nonlinear deformation and contact ambiguity become more significant. To address this problem, we propose a hierarchical local-global fusion framework for reconstructing large-range physically consistent force maps.

The proposed framework combines:
- local estimation that captures and estimates fine-scale tactile deformation cues
- global-constrained refinement that improves the physical consistency of the reconstructed force map under wide-range contact conditions.

## Code

- `Forward_MLP.py`: script for training the forward MLP model.
- `FMR-Net.py`: script for training the proposed force-map refinement network (FMR-Net).

## Dataset

We also release LaraS2R, a large-range Sim2Real-paired dataset constructed for force-map reconstruction. The dataset is available on request.

## Citation

If you find this repository useful in your research, please cite the corresponding paper.
