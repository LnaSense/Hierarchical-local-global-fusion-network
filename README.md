The repository of a hierarchical local-global fusion network for large-range physically consistent force-map reconstruction.

This repository provides the open source code for the article "Physically consistent force-map reconstruction with the fusion of local estimation and global-constrained refinement in large-range contact-rich manipulation". In this article, we propose a hierarchical local-global fusion network, in the aim of addressing the limitation of reconstructing large-range force maps in tactile sensing. You can obtain the training code of the forward MLP model and the well-designed force-map refinement network (FMR-Net).

Forward_MLP.py: The code to train the forward MLP.
FMR-Net.py: The code to train FMR-Net.

Furthermore, we construct a large-range Sim2Real-paired dataset (LaraS2R) for the training and evaluation of the proposed framework, as well as the comparison of relevant baselines and development of future studies.

The dataset LaraS2R is available at: https://github.com/LnaSense/Hierarchical-local-global-fusion-network/releases/tag/v1.0
