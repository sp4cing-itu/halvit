# HaLViT: Half of the Weights are Enough

**CVPR 2024 Workshop on Efficient Computer Vision (ECV) - Open Access**

This repository contains the official introduction and summary of the paper "HaLViT: Half of the Weights are Enough," accepted and published at the CVPR 2024 Workshop.

## Paper Details

* **Title:** HaLViT: Half of the Weights are Enough
* **Authors:** Onur Can Koyun, Behçet Uğur Töreyin
* [cite_start]**Affiliation:** Informatics Institute, Signal Processing for Computational Intelligence Research Group (SP4CING), Dept. of Artificial Intelligence and Data Engineering, İstanbul Technical University (ITU), İstanbul, Türkiye [cite: 6, 7, 8, 9]

## Abstract

Deep learning architectures like Transformers and Convolutional Neural Networks (CNNs) have led to groundbreaking advances across numerous fields. [cite_start]However, their extensive need for parameters poses challenges for implementation in environments with limited resources[cite: 12, 13].

[cite_start]In this research, we propose a strategy that focuses on the utilization of the column and row spaces of weight matrices, significantly reducing the number of required model parameters without substantially affecting performance[cite: 14]. [cite_start]This technique is applied to both Bottleneck (ResNet) and Attention (ViT) layers, achieving a notable reduction in parameters with minimal impact on model efficacy[cite: 15].

Our proposed model, HaLViT, exemplifies a parameter-efficient Vision Transformer. [cite_start]Through rigorous experiments on the ImageNet dataset and COCO dataset, HaLViT's performance validates the effectiveness of our method, offering results comparable to those of conventional models[cite: 16].

## Methodology

[cite_start]The core intuition of HaLViT is that with the application of a nonlinear function $\mathcal{F}(\cdot)$, the resulting vector $y=\mathcal{F}(Wx)$ no longer confines itself to the column space of $W$[cite: 130]. [cite_start]Consequently, the row and column spaces of matrix $W$ can be utilized independently in each layer to reduce the number of parameters[cite: 131].

We reformulate the operations to use a single weight matrix $W$ and its transpose $W^T$ for both projection and back-projection tasks, effectively halving the parameter count in specific layers.

### 1. Transformer Encoder Layer
* **Feed Forward Network (FFN):** Instead of two distinct matrices $W_1$ and $W_2$, we use $W$ and $W^T$.
    [cite_start]$$FFN(x) = W^T \mathcal{F}(Wx + b_1) + b_2$$ [cite: 165]
* [cite_start]**Multi-Head Attention (MHA):** We project input feature vectors into query, key, and value spaces using shared matrices, significantly reducing parameters[cite: 143].

### 2. Bottleneck Layer (CNN)
* [cite_start]Applied to ResNet architectures (specifically stages 3 and 4 of ResNet50), reusing weights for $1\times1$ convolutions[cite: 239].

## Experimental Results

### ImageNet-1k Classification
[cite_start]HaLViT demonstrates competitive performance with significantly fewer parameters compared to standard ViT and DeiT models[cite: 238, 241].

| Method | Parameters (M) | Top-1 Accuracy (%) |
| :--- | :---: | :---: |
| DeiT-T/16 | 5.7 | 72.2 |
| PVT-T | 13.2 | 75.1 |
| **HaLViT-T (Ours)** | **11.1** | **78.8** |
| ViT-Base | 86 | 77.9 |
| **HaLViT-Small (Ours)** | **43** | **81.3** |

### COCO Object Detection & Instance Segmentation
Evaluated on COCO val2017 using Mask R-CNN. [cite_start]HaLViT-M surpasses other models in its category[cite: 260, 262].

| Backbone | Parameters (M) | Box AP | Mask AP |
| :--- | :---: | :---: | :---: |
| ResNet18 | 31.2 | 34.0 | 31.2 |
| PVT-T | 32.9 | 36.7 | 35.1 |
| **HaLViT-T** | **30.8** | **35.3** | **33.3** |
| ResNet101 | 63.2 | 40.4 | 36.4 |
| PVT-M | 63.9 | 42.0 | 39.0 |
| **HaLViT-M** | **63.0** | **42.3** | **39.2** |

## Acknowledgments

This work was supported by:
* [cite_start]The Scientific and Technological Research Council of Türkiye (TUBITAK) with 1515 Frontier R&D Laboratories Support Program for BTS Advanced AI Hub (Project 5239903, Grant 121E378)[cite: 307, 308].
* [cite_start]Scientific Research Projects Coordination Department (BAP), Istanbul Technical University (Projects ITU-BAP MGA-2024-45372 and HIZDEP)[cite: 308].
* [cite_start]National Center for High Performance Computing (UHEM) (Grant numbers 1016682023 and 4016562023)[cite: 309].

## Citation

If you use this work or code, please cite our paper:

```bibtex
@inproceedings{koyun2024halvit,
  title={HaLViT: Half of the Weights are Enough},
  author={Koyun, Onur Can and T{\"o}reyin, Beh{\c{c}}et U{\u{g}}ur},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  pages={3669--3678},
  year={2024},
  organization={Computer Vision Foundation}
}
