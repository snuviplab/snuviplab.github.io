---
layout: post
title: “U-Net" 
modified:
categories: paper_review
tags: [Object Detection, Object Segmentation, Medical Data, U-Net] 
image:
  feature:
date: 2021-10-14T16:42:00-00:00 
comments : true
---


# U-Net

30 July 2021

## U-Net: Convolutional Networks for Biomedical Image Segmentation (CVPR, 2015)

[https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

- Semantic segmentation model developed specifically for biomedical images

### Motivation

- use patches to replace the sliding windows
    - reduce redundancy due to overlapping patches
- large patches → more max-pooling → lower localization accuracy
- small patches → see only little context

### Fully Convolutional Networks

![Untitled](https://user-images.githubusercontent.com/46922219/137266160-d676f455-6f06-4815-9d9d-26d7ea259427.png)

- uses sliding windows
- **remove fully connected layers** and replace with $1\times1$ convolution
    - no need to fix the input image shape
    - preserve spatial information
- **skip layers** to combine high resolution freatures from the encoder with the upsampled output
    - successive convolution to learn a more precise localization output

### Approach

- upsampling operators with **large number of feature channels**
    - increase the resolution of the output
    - propagate context information to higher resolution layers
    
      → **U-shape**d architecture
    
- use **patches** of the input instead of sliding windows
    - remove redundant computations to reduce the training cost
- excessive data augmentation
    - learn invariance to deformation of the image corpus
    - learn even with small training set

### Network Architecture

![Untitled 1](https://user-images.githubusercontent.com/46922219/137266130-2efb051f-7ac4-4e34-a962-d0c9dd39b5c2.png)

- **Contracting Path** (left half) - **encoder**
    - repeated $3\times3$ convolutions + ReLU + $2\times2$ max pooling [**downsampling**]
        - VGG based architecture
    - downsampling: double the number of feature channels
    - convolutions: unpadded
    
    : extract spatial context using convolutions
    
- **Expansive Path** (right half) - **decoder**
    - [**upsampling**] $2\times2$ up-convolution + concat with cropped feature map + $3\times3$ convolutions + ReLU
    - upsampling: halve the number of feature channels
    - concatenation: skip connection between encoder and decoder → combine context information to localization step to enhance pixel segmentation
        - cropping: necessary due to the loss of border pixels in every convolution
            - unpadded convolution → different input & output size (572*572 & 388*388): missing border pixels
            - solution 1) **mirror extrapolation**
                - extrapolate the missing context by mirroring the input image instead of zero-padding
                
                ![Untitled 2](https://user-images.githubusercontent.com/46922219/137266135-e6195542-7c74-4c3c-bd05-24f1da4bf04d.png)
                
            - solution 2) **overlap-tile strategy**
                - generate seamless segmentation that only pixels for which the full context is available from the input image
                - important to apply to large images (by separating the original input into overlapping tiles)
                
                ![Untitled 3](https://user-images.githubusercontent.com/46922219/137266138-3a65621d-86ee-459c-a4de-ed7210994ee7.png)
                
    
    : increase localization accuracy by using both upsampled information and context feature map
    
- final layer
    - $1\times1$ convolution: map each component feature vector to number of classes
        - classification without FC layer to preserve spatial information
    

### Training

- implementation: SGD of Caffe
- to minimize overhead: large input tiles over a large batch size → reduce batch to a single image
- loss: pixel-wise soft-max function combined with cross entropy
    - pixel-wise soft-max
        - $p_k(x):\text{ probability that pixel }x \text{ belongs to class }k$
        - $a_k(x): \text{logit that pixel }x \text{ belongs to class }k$ (model output)
    
    $$p_k(x)=\exp(a_k(x))/(\Sigma_{k'=1}^K\exp(a_{k'}(x))$$
    
    - cross entropy loss
        - penalize each pixel if $p_{l(x)}(x)$  deviates from 1
            - $p_{l(x)}(x) =1 \text{ if pixel }x \text{ belongs to class }l(x)$
            - $l(x): \text{GT label of pixel }x$
    
    $$E=\sum_{x\in\Omega}w(x)\log(p_{l(x)}(x))$$
    
    - **weight map**
        
        ![Untitled 4](https://user-images.githubusercontent.com/46922219/137266143-2f8137c7-85c1-4860-aa93-f902ee32d9ea.png)
        
        - large if pixel x is close to the borders → **more weight for border adjacent pixels**
        - $w_0, \sigma: \text{hyperparameters; by defualt: } 10, 5$
        - $w_c(x): \text{weight map to balance the class frequencies}$
            - more weight to less frequent labels → learn even the segments of small area
        - $d_1(x): \text{pixel }x's \text{ distance to the border of the nearest cell}$
        - $d_2(x): \text{pixel }x's \text{ distance to the border of the second nearest cell}$
    
    $$w(x)=w_c(x)+w_0 \cdot\exp(-\frac{(d_1(x)+d_2(x))^2}{2\sigma^2})$$
    
    - pre-computed weight map for each ground truth segmentation: compensate different frequency of pixels from a certain class in the training sset → force the network to learn small separation borders
- **Why weight loss?**
    
    → proportion of **border pixels** is low: without weight loss, borders between different objects that belong to the same class might be ignored and displayed as a single object
    

### Experiments

1. segmentation of neuronal structures [EM segmentation challenge]
    - warping error: segmentation metric, cost function for learning boundary detection
    - rand error: defined as 1 - the maximal F-score of the foreground-restricted Rand index, measure of similarity between two clusters or segmentations
    - pixel error: squared Euclidean distance between the original and the result labels
    
    ![Untitled 5](https://user-images.githubusercontent.com/46922219/137266147-8b6f4e9c-eb24-4187-93a5-34c00e74c7da.png)
    
2. light microscopic images [ISBI cell tracking challenges]
    - IOU results (intersection over union)
    
    ![Untitled 6](https://user-images.githubusercontent.com/46922219/137266149-1e8c3c10-2dd7-4305-8050-65593b47e47c.png)
    
    - segmentation results
    
    ![Untitled 7](https://user-images.githubusercontent.com/46922219/137266152-24b53e89-8606-480b-b482-003c77536915.png)
    

### Contributions

- faster training speed
- optimize trade-off between good localization and the use of context

### References

1.  [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
2. [https://joungheekim.github.io/2020/09/28/paper-review/](https://joungheekim.github.io/2020/09/28/paper-review/)
3. [https://kuklife.tistory.com/119](https://kuklife.tistory.com/119)
