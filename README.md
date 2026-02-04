# HaLViT: Half of the Weights are Enough

**CVPR Workshop Paper - Open Access**

[cite_start]This repository contains the summary and introduction of the paper "HaLViT: Half of the Weights are Enough," which was accepted and published as a CVPR Workshop paper[cite: 2, 5].

## General Overview

[cite_start]Deep learning architectures, such as Transformers and Convolutional Neural Networks (CNNs), have led to groundbreaking advances across numerous fields[cite: 12]. [cite_start]However, their extensive need for parameters poses challenges for implementation in environments with limited resources[cite: 13].

[cite_start]In this research, we propose a strategy that focuses on the utilization of the column and row spaces of weight matrices[cite: 14]. [cite_start]This approach significantly reduces the number of required model parameters without substantially affecting performance[cite: 14]. [cite_start]The technique is applied to both Bottleneck and Attention layers, achieving a notable reduction in parameters with minimal impact on model efficacy[cite: 15].

[cite_start]The proposed model, HaLViT, exemplifies a parameter-efficient Vision Transformer[cite: 16]. [cite_start]Through rigorous experiments on the ImageNet dataset and COCO dataset, HaLViT's performance validates the effectiveness of the method, offering results comparable to those of conventional models[cite: 16].

## Authors and Affiliation

* [cite_start]**Authors:** Onur Can Koyun, Behçet Uğur Töreyin [cite: 6]
* [cite_start]**Affiliation:** Informatics Institute, Signal Processing for Computational Intelligence Research Group (SP4CING), Dept. of Artificial Intelligence and Data Engineering, İstanbul Technical University, İstanbul, Türkiye [cite: 7, 8, 9]

## Key Contributions and Results

* [cite_start]**Parameter Efficiency:** The primary contribution of this approach lies in its ability to offer a parameter-efficient solution for deep learning models, particularly beneficial in environments constrained by limited memory resources[cite: 43].
* [cite_start]**Method Application:** The method leverages the column and row spaces of weight matrices to independently utilize them in each layer, applied to Transformer encoder layers in Vision Transformers (ViTs) and bottleneck layers in Residual Networks (ResNets)[cite: 131, 136].
* **Performance:**
    * [cite_start]**ImageNet:** HaLViT-Tiny achieved a top-1 accuracy of 77.3% (300 epochs) and 78.8% (600 epochs) with only 11.1M parameters[cite: 221, 222]. [cite_start]HaLViT-M attained a top-1 accuracy of 81.3% with 43.0M parameters[cite: 224].
    * [cite_start]**Object Detection (COCO):** HaLViT-M surpasses other models in its category (such as PVT-M) in object detection tasks[cite: 262].

## Acknowledgments

This work was supported by:
* [cite_start]The Scientific and Technological Research Council of Türkiye (TUBITAK) with 1515 Frontier R&D Laboratories Support Program for BTS Advanced AI Hub[cite: 307].
* [cite_start]Scientific Research Projects Coordination Department (BAP), Istanbul Technical University (Projects ITU-BAP MGA-2024-45372 and HIZDEP)[cite: 308].
* [cite_start]National Center for High Performance Computing (UHEM)[cite: 309].

## Citation

```bibtex
@inproceedings{koyun2024halvit,
  title={HaLViT: Half of the Weights are Enough},
  author={Koyun, Onur Can and T{\"o}reyin, Beh{\c{c}}et U{\u{g}}ur},
  booktitle={CVPR Workshop},
  year={2024}
}
