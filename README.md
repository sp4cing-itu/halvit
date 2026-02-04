
# HaLViT: Half of the Weights are Enough

This repository contains the introduction and implementation details for the paper **"HaLViT: Half of the Weights are Enough"**, published at the **CVPR 2024 Workshop** (Open Access version provided by the Computer Vision Foundation).

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
* 
## Introduction

**HaLViT** reduces the number of parameters in deep learning models.

**What the method does:**
Instead of using two separate weight matrices ($W_1$ and $W_2$) for consecutive linear transformations in a layer, the method uses a **single weight matrix ($W$)**. It uses the matrix $W$ for the first transformation and its **transpose ($W^T$)** for the second transformation.

## Method

The method applies this weight-sharing strategy to both Vision Transformers and Convolutional Neural Networks.

### 1. Vision Transformers (ViT)

**Feed Forward Network (FFN):**
The method replaces the two distinct matrices typically used in FFNs with a single matrix $W$.
1.  **First:** The input $x$ is multiplied by $W$.
2.  **Second:** The nonlinear activation $\mathcal{F}$ is applied.
3.  **Third:** The result is multiplied by the transpose $W^T$.

* **Equation:**
    $$FFN(x)=W^{T}\mathcal{F}(Wx+b_{1})+b_{2}$$

**Multi-Head Attention (MHA):**
The method shares weights between Key/Value projections and uses the transpose for the final projection.
1.  **Keys & Values:** Uses a shared matrix $W_{kv}$. Keys = $W_{kv}x$, Values = $W_{kv}^T x$.
2.  **Queries:** Uses a separate matrix $W_{q}$.
3.  **Final Projection:** Uses the transpose of the query matrix, $W_{q}^{T}$.

* **MHA Equation:**
    $$\hat{x}=MHA(Q,K,V)=MHA(W_{q}x,W_{kv}x,W_{kv}^{T}x)$$
* **Projection Equation:**
    $$Proj(\hat{x},W_{q}^{T})=W_{q}^{T}\hat{x}$$

### 2. Convolutional Neural Networks (CNNs)

**Bottleneck Layer:**
The method is applied to the $1\times1$ convolutions in the bottleneck block.
1.  The first $1\times1$ convolution uses $W$.
2.  The intermediate $3\times3$ convolution ($\mathcal{G}$) is applied.
3.  The second $1\times1$ convolution uses the transpose $W^T$.

* **Equation:**
    $$Bottleneck(x) = W^T \mathcal{G}(Wx)$$

**Implementation Details (ResNet50):**
* **Stages:** Applied only to **stages 3 and 4**.
* **Weight Sharing:** The same weight matrix $W$ is reused across multiple bottleneck layers within the same stage.
* **Equation with Sharing:**
    $$Bottleneck(x_{n})=W^{T}\mathcal{G}_{n}(Wx_{n})+x_{n}$$

## Performance Results

### 1. Image Classification (ImageNet-1k)

**Transformer Models (ViT)**

| Model | Epochs | Params (M) | Top-1 Acc. (%) |
| :--- | :--- | :--- | :--- |
| **HaLViT-Tiny** | 300 | 11.1 | 77.3 |
| **HaLViT-Tiny** | 600 | 11.1 | 78.8 |
| **HaLViT-Small** | 300 | 43.0 | 81.3 |

* **Comparison:** HaLViT-M (43.0M) achieves 81.3% accuracy, comparable to ViT-Small/16 (48.8M, 80.8% accuracy).

**CNN Models (ResNet)**

| Model | Params (M) | Top-1 Acc. (%) |
| :--- | :--- | :--- |
| Standard ResNet18 | 11.7 | 69.7 |
| Standard ResNet50 | 25.6 | 76.1 |
| **ResNet50 (HaLViT)** | **13.4** | **75.1** |

* **Comparison:** ResNet50 with HaLViT has a parameter count comparable to ResNet18 (13.4M vs 11.7M) but achieves higher accuracy (75.1% vs 69.7%).

### 2. Object Detection (COCO)

Evaluated on COCO val2017 using Mask R-CNN.

| Backbone | Params (M) | Box AP ($AP^{b}$) |
| :--- | :--- | :--- |
| PVT-T | 32.9 | 36.7 |
| **HaLViT-T** | **30.8** | **35.3** |
| PVT-M | 63.9 | 42.0 |
| **HaLViT-M** | **63.0** | **42.3** |

* **Comparison:** HaLViT-M outperforms PVT-M (42.0 AP) with fewer parameters.

## Acknowledgments

This work was supported by:
* The Scientific and Technological Research Council of Türkiye (TUBITAK) - Project 121E378.
* Istanbul Technical University (ITU) Scientific Research Projects (BAP).
* National Center for High Performance Computing (UHEM).
