# HaLViT: Half of the Weights are Enough

**CVPR Workshop Paper - Open Access**

[cite_start]This repository contains the summary and introduction of the paper "HaLViT: Half of the Weights are Enough," which was accepted and published as a CVPR Workshop paper[cite: 2, 5].

## General Overview

[cite_start]Deep learning architectures, such as Transformers and Convolutional Neural Networks (CNNs), have led to groundbreaking advances across numerous fields[cite: 12]. [cite_start]However, their extensive need for parameters poses challenges for implementation in environments with limited resources, such as mobile devices or edge computing platforms[cite: 13, 37].

[cite_start]In this research, we propose a strategy that focuses on the utilization of the column and row spaces of weight matrices[cite: 14]. [cite_start]This approach significantly reduces the number of required model parameters without substantially affecting performance[cite: 14]. [cite_start]The technique is applied to both Bottleneck layers (in ResNets) and Attention layers (in Vision Transformers), achieving a notable reduction in parameters with minimal impact on model efficacy[cite: 15].

[cite_start]The proposed model, HaLViT, exemplifies a parameter-efficient Vision Transformer[cite: 16]. [cite_start]Through rigorous experiments on the ImageNet dataset and COCO dataset, HaLViT's performance validates the effectiveness of the method, offering results comparable to those of conventional models while using significantly fewer parameters[cite: 16].

## Method Application

[cite_start]The core of our method leverages the row and column spaces of weight matrices to utilize them independently in each layer[cite: 131]. We apply this concept distinctively to Transformers and CNNs.

### HaLViT Transformer
[cite_start]In Vision Transformers (ViTs), our approach effectively reduces the parameter count by half[cite: 137].

* [cite_start]**Multi-Head Attention (MHA):** Instead of using separate matrices for queries, keys, and values, we project the input feature vector using a shared matrix for keys and values ($W_{kv}$) and a separate matrix for queries ($W_{q}$)[cite: 143]. [cite_start]This significantly reduces the number of parameters[cite: 143].
* [cite_start]**Feed Forward Network (FFN):** Instead of utilizing two distinct weight matrices ($W_{1}$ and $W_{2}$), our approach leverages the column space and row space of a single weight matrix $W$[cite: 163]. [cite_start]The operation is defined as $FFN(x)=W^{T}\mathcal{F}(Wx+b_{1})+b_{2}$[cite: 165].

### HaLViT CNN (ResNet Bottleneck)
[cite_start]We implement the method in the Bottleneck layers of Residual Networks (ResNets) to increase network efficiency[cite: 174].

* [cite_start]**Bottleneck Layer:** The proposed bottleneck operation uses the transpose of the weight matrix for the final projection: $Bottleneck(x) = W^T G(Wx)$, where $G$ consists of the $3\times3$ convolution and non-linearities[cite: 175, 181].
* [cite_start]**Weight Sharing:** Implementing the method in the Bottleneck layer reduces parameters less effectively than in the Transformer layer[cite: 183]. [cite_start]Consequently, we adopted a strategy of weight sharing in each stage in ResNets to further decrease the parameter count without adversely affecting performance[cite: 184].

## Authors and Affiliation

* [cite_start]**Authors:** Onur Can Koyun, Behçet Uğur Töreyin[cite: 6].
* [cite_start]**Affiliation:** Informatics Institute, Signal Processing for Computational Intelligence Research Group (SP4CING), Dept. of Artificial Intelligence and Data Engineering, İstanbul Technical University, İstanbul, Türkiye[cite: 7, 8, 9].

## Key Contributions and Results

* [cite_start]**Parameter Efficiency:** The primary contribution of this approach lies in its ability to offer a parameter-efficient solution for deep learning models[cite: 43]. [cite_start]It effectively reduces the parameter count of Vision Transformers (ViTs) and ResNets by approximately half[cite: 40].
* [cite_start]**Method Application:** The method leverages the column and row spaces of weight matrices to independently utilize them in each layer[cite: 131]. [cite_start]It reuses the single weight matrix and its transpose for projection and back-projection tasks, separated by nonlinear activation functions[cite: 167].
* **Performance:**
    * [cite_start]**ImageNet Classification:** HaLViT-Tiny achieved a top-1 accuracy of 78.8% with only 11.1M parameters[cite: 222]. [cite_start]HaLViT-M attained a top-1 accuracy of 81.3% with 43.0M parameters[cite: 224]. [cite_start]These results are competitive with or superior to other efficient models like PVT and DeiT[cite: 223, 241].
    * [cite_start]**Object Detection (COCO):** When integrated with Mask R-CNN, HaLViT-M (63.0M parameters) demonstrated superior performance metrics compared to models like PVT-M and ResNet101[cite: 262].

## Acknowledgments

This work was supported by:
* [cite_start]The Scientific and Technological Research Council of Türkiye (TUBITAK) with 1515 Frontier R&D Laboratories Support Program for BTS Advanced AI Hub[cite: 307].
* [cite_start]Scientific Research Projects Coordination Department (BAP), Istanbul Technical University[cite: 308].
* [cite_start]National Center for High Performance Computing (UHEM)[cite: 309].

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
