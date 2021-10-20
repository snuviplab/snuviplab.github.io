---
layout: post
title: "Faster R-CNN (NIPS 2015)"
modified:
categories: paper_review
excerpt:
tags: [Object Detection]
image:
 feature:
date: 2021-10-20T03:07:00-00:00
comments : true
---

# Faster R-CNN (NIPS 2015)

## Main Idea

- Fast R-CNN drawbacks
    - region proposal이 차지하는 시간이 매우 길다 (test time:2.3sec, proposal time:2 sec)
    - real-time 으로는 부적합
- Region Proposal Network(RPN)
    - Selective Search를 사용하지 않고 RPN을 통해서 RoI를 계산함
    - GPU를 통한 RoI 연산이 가능해짐
    - 이 외에는 Fast R-CNN과 동일한 구조 (⇒ RoI pooling → classification & bbox regression)

## Faster R-CNN architecture

![/images/Fast-RCNN/2021-07-27_5.34.38.png](/images/Fast-RCNN/2021-07-27_5.34.38.png)

### Region Proposal Network(RPN) module

RPN의 목적

- 생성한 Region proposal이 Object 여부와 bbox의 좌표를 예측하면서 적절한 bbox를 생성하도록 학습하는 것

```markdown
💥 input : feature map
💥 output : object proposals
```

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled 5.png)

[https://oi.readthedocs.io/en/latest/computer_vision/object_detection/faster_r-cnn.html](https://oi.readthedocs.io/en/latest/computer_vision/object_detection/faster_r-cnn.html)

![/images/Fast-RCNN/Untitled 6.png](/images/Fast-RCNN/Untitled 6.png)


1. CNN을 통해 뽑아낸 feature map을 input으로 받음. 
2. feature map에 3x3 convolution을 256 channel (or 512 channel)만큼 수행함. (→ 위 그림의 중간 단계)
    - ZFNet → 256
    - VGGNet → 512
3. 2번째 feature map을 input으로 받아 classification(*cls* layer)과 bbox regression(*reg* layer) 예측 값을 계산함.
    - cls → obj. / no obj.
    - bbox regression → box locations
4. CLS layer
    - Anchor Targeting
    - Anchor : 각 슬라이딩 윈도우에서 bounding box의 후보로 사용되는 box
    - feature 맵에 sliding window 방법으로 중심 좌표를 중심으로 k개의 anchor box를 만듦
        ![/images/Fast-RCNN/Untitled 6.png](/images/Fast-RCNN/Untitled 6.png)
        
    - k=9 ⇒ 3개의 scale(8,16,32) X 3개의 ratio(0.5,1,2)
    - `1x1 conv` 를 `(2(obj/No obj.) x 9(Anchor 개수 = k)) 채널 수` 만큼 수행하며, 그 결과로 H*W*18(2x9) 크기의 feature map을 얻음.
    - 18개의 채널 → 앵커박스 각각에 대한 obj, no.obj에 예측값
        - G.T Label → 만들어진 모든 Anchor들과 G.T Box의 IoU를 계산하여 결정
            - IoU > 0.7 : 1(Positive), G.T Box마다 IoU가 가장 높은 Anchor : 1(Positive)
            IoU <0.3 : 0(Negative)

    **⇒ 한번의 1x1 컨볼루션으로 Anchor들에 대한 예측을 모두 수행할 수 있다.**

5. bbox Regression 
    - `1x1 conv`을 `(4_bbox x 9_anchorBox) 채널 수` 만큼 수행
6. 4번 5번의 Classification의 값과 bbox Regression의 값들을 통해 RoI를 계산함.
    - Classification을 통해서 얻은 obj 확률 값들을 정렬하여 높은 순으로 K개의 앵커를 고름. 
    → K개의 앵커들에 각각 bbox regression을 적용
    → Non-Max Suppression(NMS)을 적용하여 특정 개수의 RoI만큼 샘플링
        - RoI Sampling
            - 보통 Training시에 NMS를 거치면 2000개의 RoI가 남음.
            - Positive:Negative 비율이 1:1이 되도록 RoI를 Sampling
                - 만약 1:1이 안된다면, IoU 값이 가장 높은 box를 positive anchor로 사용함.

#### Loss for RPN
<img src="/images/Fast-RCNN/Untitled 7.png" width = "40%">

- 전체 Anchor 개수 중 i번째 anchor에 대한 Loss
- $L_{cls}$ : Log loss
    - G.T label = 0(==no obj)라면 뒤의 regression term은 날아감.
- $L_{reg}$ : Smooth L1 Loss
    - R-CNN과 동일

## Experiments

### Training

- 4-step alternating training strategy

![/images/Fast-RCNN/Untitled 8.png](/images/Fast-RCNN/Untitled 8.png)

cs231n

- Step 1:Train RPNs
    
    > ImageNet으로 Pre-train된 ConvNet을 이용하여 RPN을 end-to-end로 학습함
    > 
- Step 2: Train Fast R-CNN using the proposals from RPNs
    
    > ImageNet으로 Pre-train된 ConvNet과 Step 1에서 학습된 RPN을 이용하여 Fast R-CNN 모델을 학습시킴
    > 
- Step 3: Fix the shared convolutional layers and fine-tune unique layers to RPN
    
    > 앞서 학습시킨 Fast RCNN과 RPN을 불러온 다음, 다른 weight들은 고정하고 RPN에 해당하는 레이어들만 fine-tuning
    여기서부터 RPN과 Fast RCNN이 convolution weight를 **Share**함
    > 
- Step 4: Fine-tune unique layers to Fast R-CNN
    
    > 공유하는 CNN과 RPN은 고정시킨고,  Fast R-CNN에 해당하는 레이어만 fine tune 시킴
    > 

![/images/Fast-RCNN/2021-08-02_4.04.11.png](/images/Fast-RCNN/2021-08-02_4.04.11.png)

[https://www.youtube.com/watch?v=kcPAGIgBGRs&ab_channel=JinWonLee](https://www.youtube.com/watch?v=kcPAGIgBGRs&ab_channel=JinWonLee)


### Dataset

- PASCAL VOC 2007, 2012
    
    ![/images/Fast-RCNN/2021-07-28_9.46.28.png](/images/Fast-RCNN/2021-07-28_9.46.28.png)
    
    ![/images/Fast-RCNN/2021-07-28_9.46.45.png](/images/Fast-RCNN/2021-07-28_9.46.45.png)
    
    ![/images/Fast-RCNN/2021-08-01_11.11.02.png](/images/Fast-RCNN/2021-08-01_11.11.02.png)
    
- MS COCO
    
    ![/images/Fast-RCNN/2021-07-28_9.47.26.png](/images/Fast-RCNN/2021-07-28_9.47.26.png)
    

## Conclusion

- R-CNN, Fast R-CNN, Faster R-CNN 비교 사진

![/images/Fast-RCNN/Untitled 9.png](/images/Fast-RCNN/Untitled 9.png)

[https://brunch.co.kr/@kakao-it/66](https://brunch.co.kr/@kakao-it/66) Kakao AI Report

- Region Proposal이 Selective Search ⇒  RPN
- 이전보다 빠르고, 성능 Good
- 복잡한 트레이닝 과정
- 그래도 아직 real-time까지는...
