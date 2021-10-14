---
layout: post
title: “VGG Net" 
modified:
categories: paper_review
tags: [CNN, VGG, VGGNet, Object Detection] 
image:
  feature:
date: 2021-10-14T16:50:00-00:00 
comments : true
---


# VGG

5 July 2021

## Very Deep Convolutional Networks for Large-Scale Image Recognition (CVPR, 2014)

[https://arxiv.org/abs/1409.1556](https://arxiv.org/abs/1409.1556)

- 층을 깊게 쌓는 deep neural network의 시작
- 크기가 작은 필터 (= 적은 파라미터 수)를 가지고도 높은 정확도를 보임

### configurations

- Architectures
    - Convolution: layer depth의 영향을 확인하기 위해 kernel size는 3*3 (최소한)으로 고정
    - Pooling: 2*2, stride 2 max-pool layers for 5 convolution layers
    - stack of convolutional layers + FC layers + softmax layer
    - Activation: ReLU
- Configurations
    
    ![Untitled](https://user-images.githubusercontent.com/46922219/137274168-814d6ea1-b78c-47d2-b9a4-873629a55dc6.png)
    

D: VGG16, E: VGG19

- LRN: Local Response Normalization - 특히 높은 pixel 값을 normalize 해주는 것
    
    increased memory consumption and computation time
    
- single 7*7 layer vs three 3*3 layers
    1. multiple activation layers → more discriminative decision function
    2. decreased number of parameters
- 1*1 layer
    
    → affect of increased non-linearity
    
    → dimensionality reduction (decreased channel size)
    

### classification framework

batch size: 256, momentum: 0.9 (SGD)

regularization: weight decay, dropout 0.5 for FC layers

learning rate: 0.01, decreased by 0.1 if validation accracy stopped increasing

- required less epochs to converge due to
    1. implicit regularization by greater depth and smaller convolution
    2. pre-initialization of some layers using shallower network (VGG-A)

![Untitled 1](https://user-images.githubusercontent.com/46922219/137274150-e6a386ff-84bf-4934-8b41-ff0aec77dc4a.png)

### experiments

1. dataset: ILSVRC-2012 (1000 classes, 1.3M training / 50K validation / 100K testing images)
2. performance: top-1 error, top-5 error
3. result

![Untitled 2](https://user-images.githubusercontent.com/46922219/137274164-ac110e4a-f06d-40cd-b093-6d366bc4a42e.png)

### conclusion

1. 간단한 구조로 좋은 성능을 구현
2. 특징: 작은 kernel size로 반복되는 channel size를 여러개 쌓는 구조
3. VGG16, VGG19를 주로 사용
4. 매우 깊은 인공신경망 구조의 시작
