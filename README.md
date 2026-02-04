
# HaLViT: Half of the Weights are Enough

This repository contains the introduction and implementation details for the paper **"HaLViT: Half of the Weights are Enough"**, published at the **CVPR 2024 Workshop**.

## Authors and Affiliation

**Authors:** Onur Can Koyun, Behçet Uğur Töreyin

**Affiliation:**
* Istanbul Technical University (ITU), Türkiye
* Informatics Institute
* Signal Processing for Computational Intelligence Research Group (SP4CING)
* Dept. of Artificial Intelligence and Data Engineering

**Contact:** `{okoyun, toreyin}@itu.edu.tr`

## Web

* https://spacing.itu.edu.tr/halvit.html
* https://openaccess.thecvf.com/content/CVPR2024W/ELVM/html/Koyun_HaLViT_Half_of_the_Weights_are_Enough_CVPRW_2024_paper.html

## Introduction

Deep learning architectures like Transformers and Convolutional Neural Networks (CNNs) have led to advances but require extensive parameters.

**HaLViT** proposes a strategy that focuses on the utilization of the column and row spaces of weight matrices. The method utilizes the property that the output of a nonlinear activation function does not confine itself to the column space of the weight matrix. This allows the row and column spaces of matrix W to be independently utilized in each layer to reduce the number of parameters.

## Method

### 1. Vision Transformers (ViT)

The method is applied to the **Transformer encoder layers**, specifically the Multi-Head Attention and Feed Forward Network.

**Multi-Head Attention (MHA):**
The method projects the input feature vector $x$ into the query, key, and value spaces using a shared matrix for keys and values ($W_{kv}$) and a separate matrix for queries ($W_{q}$).
* **MHA Equation:**
    $$\hat{x}=MHA(Q,K,V)=MHA(W_{q}x,W_{kv}x,W_{kv}^{T}x)$$
* **Projection Equation:**
    $$Proj(\hat{x},W_{q}^{T})=W_{q}^{T}\hat{x}$$

**Feed Forward Network (FFN):**
Instead of utilizing two distinct weight matrices ($W_{1}$ and $W_{2}$), the approach leverages the column space and row space of a single weight matrix $W$.
* **Equation:**
    $$FFN(x)=W^{T}\mathcal{F}(Wx+b_{1})+b_{2}$$

### 2. Convolutional Neural Networks (CNNs)

The method is applied to **Bottleneck layers** in Residual Networks (ResNets).

**Bottleneck Layer:**
Let $\mathcal{G}$ be a function consisting of $3\times3$ convolution, normalization, and nonlinear activation function. The method utilizes the column and row spaces of weight matrix $W$ for the $1\times1$ convolutions.
* **Equation:**
    $$Bottleneck(x) = W^T \mathcal{G}(Wx)$$

**Implementation Details (ResNet50):**
* **Stages Applied:** The method is applied exclusively to **stages 3 and 4**.
* **Weight Sharing:** The method utilizes the same weight matrix $W$ and $W^{T}$ across all $1\times1$ convolutions within each bottleneck layer at the same stages.
* **Equation with Sharing:**
    $$Bottleneck(x_{n})=W^{T}\mathcal{G}_{n}(Wx_{n})+x_{n}$$

## Performance Results

### Vision Transformers (ViT) Results
Models trained on ImageNet-1k.

| Model | Epochs | Params (M) | Top-1 Acc. (%) |
| :--- | :--- | :--- | :--- |
| **HaLViT-Tiny** | 300 | 11.1 | 77.3 |
| **HaLViT-Tiny** | 600 | 11.1 | 78.8 |
| **HaLViT-Small** | 300 | 43.0 | 81.3 |

* **Comparison:** HaLViT-M (43.0M Params) achieves 81.3% accuracy, comparable to ViT-Small/16 (48.8M Params, 80.8% accuracy).

### Convolutional Neural Networks (CNN) Results
Models trained on ImageNet-1k.

| Model | Params (M) | Top-1 Acc. (%) |
| :--- | :--- | :--- |
| Standard ResNet18 | 11.7 | 69.7 |
| Standard ResNet50 | 25.6 | 76.1 |
| **ResNet50 (HaLViT)** | **13.4** | **75.1** |

* **Note:** ResNet50 with HaLViT applied to stages 3 and 4 maintains a parameter count comparable to ResNet18 while achieving higher accuracy.

### Object Detection (COCO) Results
Evaluated using Mask R-CNN.

| Backbone | Params (M) | Box AP ($AP^{b}$) |
| :--- | :--- | :--- |
| PVT-T | 32.9 | 36.7 |
| **HaLViT-T** | **30.8** | **35.3** |
| PVT-M | 63.9 | 42.0 |
| **HaLViT-M** | **63.0** | **42.3** |

## Acknowledgments

This work was supported by:
* The Scientific and Technological Research Council of Türkiye (TUBITAK) - Project 121E378.
* Istanbul Technical University (ITU) Scientific Research Projects (BAP).
* National Center for High Performance Computing (UHEM).
