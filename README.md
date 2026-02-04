# HaLViT: Half of the Weights are Enough

**CVPR Workshop Paper - Open Access**

This repository contains the summary and introduction of the paper "HaLViT: Half of the Weights are Enough," which was accepted and published as a CVPR Workshop paper.

## General Overview

Deep learning architectures, such as Transformers and Convolutional Neural Networks (CNNs), have led to groundbreaking advances across numerous fields. However, their extensive need for parameters poses challenges for implementation in environments with limited resources, such as mobile devices or edge computing platforms.

In this research, we propose a strategy that focuses on the utilization of the column and row spaces of weight matrices. This approach significantly reduces the number of required model parameters without substantially affecting performance. The technique is applied to both Bottleneck layers (in ResNets) and Attention layers (in Vision Transformers), achieving a notable reduction in parameters with minimal impact on model efficacy.

## Method Application

The main idea behind our method is simple: **Reuse the weights.**

In standard deep learning models, layers typically use separate, distinct weight matrices for every step. We discovered that we can use a single weight matrix ($W$) for the first operation, and then reuse its transpose ($W^T$) for the subsequent operation. Because these steps are separated by a non-linear activation function, the model learns effectively while cutting the number of parameters in half.

### HaLViT Transformer
We apply this "recycling" logic to the two most parameter-heavy parts of a Vision Transformer:

* **Feed Forward Network (FFN):** A standard FFN normally requires two separate big matrices (one to expand the features, one to reduce them back). We replace these with just **one matrix**. We use $W$ to expand, and reuse $W^T$ to reduce.
* **Multi-Head Attention (MHA):** We merge the weights used for attention. Instead of separate matrices for everything, Keys and Values share a single matrix ($W_{kv}$). Queries use a separate matrix ($W_q$), but we reuse $W_q$ (transposed) again for the final output projection.

### HaLViT CNN (ResNet)
We apply the same concept to Convolutional Neural Networks, specifically the "Bottleneck" blocks in ResNet:

* **Reusing Weights:** Inside a bottleneck block, we use matrix $W$ for the first convolution layer and $W^T$ for the last convolution layer.
* **Sharing Across Blocks:** We adopt a strategy of weight sharing where the same set of parameters is reused across multiple bottleneck blocks within the same stage of the network to decrease the parameter count.

## Authors and Affiliation

* **Authors:** Onur Can Koyun, Behçet Uğur Töreyin
* **Affiliation:** Informatics Institute, Signal Processing for Computational Intelligence Research Group (SP4CING), Dept. of Artificial Intelligence and Data Engineering, İstanbul Technical University, İstanbul, Türkiye

## Key Contributions and Results

* **Parameter Efficiency:** The primary contribution of this approach lies in its ability to offer a parameter-efficient solution for deep learning models. It effectively reduces the parameter count of Vision Transformers (ViTs) and ResNets by approximately half.
* **Method Application:** The method leverages the column and row spaces of weight matrices to independently utilize them in each layer. It reuses the single weight matrix and its transpose for projection and back-projection tasks, separated by nonlinear activation functions.
* **Performance:**
    * **ImageNet Classification:** HaLViT-Tiny achieved a top-1 accuracy of 78.8% with only 11.1M parameters. HaLViT-M attained a top-1 accuracy of 81.3% with 43.0M parameters. These results are competitive with or superior to other efficient models like PVT and DeiT.
    * **Object Detection (COCO):** When integrated with Mask R-CNN, HaLViT-M (63.0M parameters) demonstrated superior performance metrics compared to models like PVT-M and ResNet101.

## Acknowledgments

This work was supported by:
* The Scientific and Technological Research Council of Türkiye (TUBITAK) with 1515 Frontier R&D Laboratories Support Program for BTS Advanced AI Hub.
* Scientific Research Projects Coordination Department (BAP), Istanbul Technical University.
* National Center for High Performance Computing (UHEM).

## Citation

```bibtex
@InProceedings{Koyun_2024_CVPR,
    author    = {Koyun, Onur Can and T\"oreyin, Beh\c{c}et U\u{g}ur},
    title     = {HaLViT: Half of the Weights are Enough},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {3669-3678}
}
