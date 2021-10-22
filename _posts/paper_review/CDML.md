---
layout: post
title: Collaborative Deep Metric Learning for Video Understanding
modified:
categories: paper_review
tags: [CDML, Collaborative Deep Metric Learning for Video Understanding, Recommender System, Metric Learning] 
image:
  feature:
date: 2021-10-22T19:11:00-00:00 
comments : true
---

# CDML

19 August 2021

## Collaborative Deep Metric Learning for Video Understanding (KDD 2018)

[https://dl.acm.org/doi/pdf/10.1145/3219819.3219856](https://dl.acm.org/doi/pdf/10.1145/3219819.3219856)

### Motivation

- challenges in video understanding
    1. large video files, prohibited downlad and store
    2. computationally expensive
    3. costly labels
    
    → can be tackeld with metric learning
    
- goal
    - learn **embedding function** → project video onto a low-dimensional space
        - related videos: close to each other
        - unrelated videos: far apart from each other
    - embedding function should generalize well across different video understanding tasks
    

### Approach

- source of information
    1. raw video content
        - extract image and audio features using SOTA deep neural networks
    2. user  behavior - collaborative filtering information
        - construct a graph - edge if co-watched by many users
- **content-aware embedding: metric space of video content embedding trained to reconstruct the CF information**
    - mapping from video content to CF signals
    - capture high-level semantic relationship between videos
- semi-supervised

### Methodology

1. Video Features
    - extract visual / audio features using pre-trained models
    
    ![Untitled](https://user-images.githubusercontent.com/46922219/138436319-35781cec-2408-4639-91be-db18084fab62.png)
    
    - video features
        - 1 FPS using Inception-v3 trained on JFT dataset
        - apply PCA to last hidden layer
        - frame level features → video level : average pooling
    - audio features
        - VGG-inspired model
        - non-overlapping 960 ms frames
        - average pooled into video level
2. Collaborative Deep Metric Learning
    - construct a graph
        - nodes: videos
        - edges: co-watched, weight: co-watch frequency
    - objective: co-watched videos → close in the embedding space
        - ranking triplet loss: training data point - triplet of three videos
            - anchor: more relevant to positive than negative
            - positive
            - negative
        - loss: hinge loss ← minimize
            
            $$\mathcal{L}_{hinge}(f_\theta(x_i^a),f_\theta(x_i^p),f_\theta(x_i^n))= [||f_\theta(x_i^a)-f_\theta(x_i^p)||^2_2 - ||f_\theta(x_i^a)-f_\theta(x_i^n)||^2_2 + \alpha]_+$$
            
        - minimize the distance between anchor and positive, maximize the distance between anchor and negative
        - $\alpha$: margin paramter, $=0$ in this experiment
    - extracted features → embedding network
        
        ![Untitled 1](https://user-images.githubusercontent.com/46922219/138436216-c0196039-14d4-490b-85b0-971990357c6d.png)
        
        1. early fusion
            - concatenate input features → FC
        2. late fusion
            - each input feature → FC layers → element-wise product

### Experiments

- semi-hard negative mining within mini-batch of 7200 triplets
    - re-sample negative from the mini-batch
        - negative not too far from the anchor
        - closest one further from the positive

**Task 1) Video Retrieval**

1. How?
    - compute similarity score between two candidates (cosine similarity)
    - query: video
    - candidates: co-watched videos
2. Dataset
    - YouTube-8M dataset: 278M videos with 1000 or more views
3. Evaluation
    - NDCG
    - MAP
4. Results
    
    ![Untitled 2](https://user-images.githubusercontent.com/46922219/138436220-e7ad752b-3fb5-4112-ac7a-4b4ba5b9f34a.png)
    

**Task 2) Video Recommendation**

1. How?
    - query: user - represented as recent watch history
    - candidate video → compute similarity scores (cosine similarity) → recommend the video with highest arithmetic mean
2. Dataset
    - MovieLens 100K + YouTube trailers
        - 25141 trailers for 26733 unique movies (94%)
        - users less than 10 test rating excluded
3. Evaluation
    - NDCG@10
    - MAP
4. Results
    
    ![Untitled 3](https://user-images.githubusercontent.com/46922219/138436225-4b64ff72-c816-4bd0-a47c-ee47fe92b5c8.png)
    

**Task 3) Video Annotation / Classification**

1. How?
    - multi-labeled classification problem
    - feature vector of the video → binary vector * number of classes
2. Dataset
    - YouTube-8M (large-scale video classification challenge)
    - MovieLens-20M (movie trailer → movie tag classification)
3. Evaluation
    - GAP (Global Average Precision): average precision based on the top 20 predictions per example
    - MAP
4. Results
    
    ![Untitled 4](https://user-images.githubusercontent.com/46922219/138436232-5acc84c4-0138-49a9-9e46-c723b901c2f1.png)
    
    ![Untitled 5](https://user-images.githubusercontent.com/46922219/138436237-ce1d1151-ddb5-462d-96da-d88afe5b7892.png)
    

### References

1. [https://dl.acm.org/doi/pdf/10.1145/3219819.3219856](https://dl.acm.org/doi/pdf/10.1145/3219819.3219856)
2. 이준석 교수님 MLVU 강의노트
