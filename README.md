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

https://spacing.itu.edu.tr/halvit.html
https://openaccess.thecvf.com/content/CVPR2024W/ELVM/html/Koyun_HaLViT_Half_of_the_Weights_are_Enough_CVPRW_2024_paper.html

## Introduction

Deep learning models like Transformers and CNNs typically use separate weight matrices for consecutive linear transformations, resulting in a large number of parameters.

**HaLViT** reduces the number of parameters by **reusing the same weight matrix** for two different operations within a layer. Instead of using two distinct matrices (e.g., $W_1$ and $W_2$), the model uses a matrix $W$ for the first transformation and its transpose ($W^T$) for the second transformation. 

## Method

The method replaces distinct weight matrices with a single shared matrix in both Transformer and Convolutional layers.

### 1. Vision Transformers (ViT)

The method is applied to the Feed Forward Network (FFN) and Multi-Head Self-Attention (MHSA).

**Feed Forward Network (FFN):**
A standard FFN uses two distinct matrices ($W_1, W_2$). HaLViT performs the operation using a single matrix $W$:
1.  **First Projection:** Multiply input $x$ by $W$.
2.  **Activation:** Apply the nonlinear function $\mathcal{F}$ (e.g., GELU/ReLU).
3.  **Second Projection:** Multiply the result by the transpose of $W$ ($W^T$).

**Equation:**
$$FFN(x) = W^T \mathcal{F}(Wx + b_1) + b_2$$

**Multi-Head Attention (MHA):**
Standard MHA uses three separate matrices for Queries ($W_q$), Keys ($W_k$), and Values ($W_v$). HaLViT reduces this to two matrices:
1.  **Query:** Uses a separate matrix $W_q$.
2.  **Key and Value:** Uses a shared matrix $W_{kv}$.
    * Keys are generated using $W_{kv}x$.
    * Values are generated using the transpose $W_{kv}^T x$.
3.  **Final Projection:** The final output is projected using $W_q^T$.

**Equation:**
$$\hat{x} = MHA(W_q x, W_{kv} x, W_{kv}^T x)$$
$$Proj(\hat{x}) = W_q^T \hat{x}$$

### 2. Convolutional Neural Networks (CNNs)

The method is applied to **Bottleneck layers**, commonly found in ResNet architectures.

**Bottleneck Layer Optimization:**
* **Standard Bottleneck:** A traditional bottleneck block uses two independent weight matrices ($W_1$ and $W_2$) for the $1 \times 1$ convolutions that reduce and expand dimensions.
* **HaLViT Bottleneck:** We use a single weight matrix $W$ for both operations.
    1.  The first $1 \times 1$ convolution uses $W$.
    2.  The intermediate $3 \times 3$ convolution and activation ($\mathcal{G}$) are applied.
    3.  The final $1 \times 1$ convolution uses the transpose $W^T$.

**Equation:**
$$Bottleneck(x) = W^T \mathcal{G}(Wx)$$

**Weight Sharing Strategy:**
To further reduce parameters in ResNets, HaLViT shares these weights across multiple bottleneck layers within the same stage.
$$Bottleneck(x_n) = W^T \mathcal{G}_n(Wx_n) + x_n$$

## Performance Results

### Vision Transformers (ViT)
Evaluation performed on ImageNet-1k (Classification) and COCO (Object Detection).

* **ImageNet-1k Classification:**
    * **HaLViT-Tiny (300 epochs):** 11.1M parameters, **77.3%** Top-1 Accuracy.
    * **HaLViT-Tiny (600 epochs):** 11.1M parameters, **78.8%** Top-1 Accuracy.
    * **HaLViT-Small:** 43.0M parameters, **81.3%** Top-1 Accuracy.
    * *Comparison:* HaLViT-M matches the performance of ViT-Small/16 (80.8%) but with fewer parameters (43M vs 48.8M).

* **COCO Object Detection (Mask R-CNN):**
    * **HaLViT-T:** 30.8M parameters, **35.3** Box AP.
    * **HaLViT-M:** 63.0M parameters, **42.3** Box AP.
    * *Comparison:* HaLViT-M outperforms PVT-M (42.0 Box AP) while using fewer parameters.

### Convolutional Neural Networks (CNN)
Evaluation performed on ImageNet-1k using the ResNet50 architecture.

* **ImageNet-1k Classification:**
    * **ResNet50 (HaLViT):** 13.4M parameters, **75.1%** Top-1 Accuracy.
    * **Standard ResNet50:** 25.6M parameters, **76.1%** Top-1 Accuracy.
    * **Standard ResNet18:** 11.7M parameters, **69.7%** Top-1 Accuracy.
    * *Observation:* The proposed method allows ResNet50 to operate with a parameter count comparable to ResNet18 (13.4M vs 11.7M) while achieving significantly higher accuracy (75.1% vs 69.7%).

## Acknowledgments

This work was supported by:
* The Scientific and Technological Research Council of Türkiye (TUBITAK) - Project 121E378.
* Istanbul Technical University (ITU) Scientific Research Projects (BAP) - Projects ITU-BAP MGA-2024-45372 and HIZDEP.
* National Center for High Performance Computing (UHEM).
