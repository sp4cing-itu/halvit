
# HaLViT: Half of the Weights are Enough

This repository contains the introduction and implementation details for the paper **"HaLViT: Half of the Weights are Enough"**, published at the **CVPR 2024 Workshop**.

## Abstract

Deep learning architectures like Transformers and Convolutional Neural Networks (CNNs) have led to ground breaking advances across numerous fields. However, their extensive need for parameters poses challenges for implementation in environments with limited resources. In our research, we propose a strategy that focuses on the utilization of the column and row spaces of weight matrices, significantly reducing the number of required model parameters without substantially affecting performance. This technique is applied to both Bottleneck and Attention layers, achieving a notable reduction in parameters with minimal impact on model efficacy. Our proposed model, HaLViT, exemplifies a parameter-efficient Vision Transformer. Through rigorous experiments on the ImageNet dataset and COCO dataset, HaLViT’s performance validates the effectiveness of our method, offering results comparable to those of conventional models.

## Web

* https://spacing.itu.edu.tr/halvit.html
* https://openaccess.thecvf.com/content/CVPR2024W/ELVM/html/Koyun_HaLViT_Half_of_the_Weights_are_Enough_CVPRW_2024_paper.html

## Introduction

**What the method does:**
Instead of using two separate weight matrices ($W_1$ and $W_2$) for consecutive linear transformations in a layer, the method uses a **single weight matrix ($W$)**. It uses the matrix $W$ for the first transformation and its **transpose ($W^T$)** for the second transformation.

## Method

The method applies this weight-sharing strategy to both **Vision Transformers** and **Convolutional Neural Networks**.

## Acknowledgments

This work was supported by:
* The Scientific and Technological Research Council of Türkiye (TUBITAK) - Project 121E378.
* Istanbul Technical University (ITU) Scientific Research Projects (BAP).
* National Center for High Performance Computing (UHEM).
