---
layout: post
title: "Fast R-CNN, Faster R-CNN"
modified:
categories: paper_review
excerpt:
tags: [Object Detection]
image:
  feature:
date: 2021-08-02T21:38:00-00:00
comments : true
---

| **Fast R-CNN (ICCV 2015)**
    
# Main Idea

- R-CNN : multi-stage(proposal, detection, bbox) ⇒ Fast R-CNN : single-stage
    - using multi-task learning
- 전체 이미지를 CNN에 넣음 (1 CNN on the entire image)
- 이전에는 Selective Search로 뽑아낸 모든 region proposal 후보들을 CNN 모델에 넣었음. (1 CNN per region)
- **2000 → 1**

# Fast R-CNN architecture

![/images/Fast-RCNN/스크린샷_2021-07-27_오후_3.26.36.png](/images/Fast-RCNN/스크린샷_2021-07-27_오후_3.26.36.png)

### Process

1. 전체 이미지를 CNN model에 통과시켜 Feature map 추출
2. Selective Search를 통해 찾아낸 RoI들을 피쳐맵에 projection시키고, 각각의 RoI에 대해서 RoI pooling을 진행 ⇒ 고정된 크기의 feature vector 추출
3. feature vector를 2개의 브랜치로 나눠 Fully Conneted layer에 통과
    1. [1번 브랜치] classification : softmax를 통과하여 해당 RoI가 어떤 오브젝트인지 분류 
    💥 output : softmax value → class
    2. [2번 브랜치] bbox regression : regression을 통해 selective search로 찾은 바운딩 박스 조정⇒ 올바른 bbox 좌표 예측 
    💥 output : bounding-box regression offsets

---

## (1) Deep ConvNet & RoI Projection

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)

[https://nuggy875.tistory.com/33](https://nuggy875.tistory.com/33)

![/images/Fast-RCNN/스크린샷_2021-07-27_오후_4.03.57.png](/images/Fast-RCNN/스크린샷_2021-07-27_오후_4.03.57.png)

                R-CNN 방식                                                                  Fast R-CNN 방식

- R-CNN → Selective Search로 뽑아낸 Region Proposal 후보들을 모두 CNN에 넣어 시간이 오래 걸림
- Fast R-CNN → 1 CNN on the entire image
    - Selective Search를 통해 Region Proposal은 함. R-CNN과 다르게 뽑아낸 영역을 그대로 가지고 있는 채로 전체 이미지를 CNN 모델에 넣음
    - 🐳 CNN model의 ouput인 Feature Map에 RoI projection
        
        > 🔍 RoI Projection : 인풋 사이즈와 피쳐맵 사이즈가 동일하다면 앞서 구한 RoI들의 좌표를 피쳐맵에 그대로 사용하면 되지만, 만약 다르다면 각 RoI들은 모델을 통과하여 줄어든 크기의 비율을 따져서 좌표를 변경시킴.
        > 

## (2) RoI Pooling

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)

> RoI들은 input 이미지에서 Selective Search를 통해 찾은 것을 feature map에 적용한 것이다.
> 
- Selective Search로 통해 구한 RoI 영역을 각기 다른 크기를 가지고 있음
- FC : 고정된 크기의 인풋 필요
- ⛳️ GOAL:  다양한 크기의 RoI → 고정된 크기로 축소
    - **이 RoI는 어느 시기의 피쳐맵을 사용?**

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)

8x8 input feature map에서 Selective Search로 뽑은 7x5짜리 region proposal(아래)

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)

2x2로 만들기위해 Stride(7/2=**3**, 5/2=**2**)로 Pooling 구역을 정하고 Max Pooling → output : 2x2 (논문에서는 7x7 사용)

![https://deepsense.ai/wp-content/uploads/2017/02/roi_pooling-1.gif.pagespeed.ce.5V5mycIRNu.gif](https://deepsense.ai/wp-content/uploads/2017/02/roi_pooling-1.gif.pagespeed.ce.5V5mycIRNu.gif)

[https://deepsense.ai/region-of-interest-pooling-explained/](https://deepsense.ai/region-of-interest-pooling-explained/)

- RoI에 다 적용 ⇒ 모두 다 같은 크기의 벡터 → FC layer

## Classification & Bounding box Regression

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)

```markdown
**(FC layer)**
💥input : 고정된 크기의 feature vector
💥output1 : softmax value(classification)
💥output2 : bounding-box regression offsets

=========
bbox regression => K + 1개 클래스에 대해서 각각 x, y, w, h 값을 **조정**하는 t_u를 리턴
==> 즉, 이 RoI가 사람일 경우 박스를 이렇게 조절해라, 고양이일 경우 이렇게 조절해라 라는 의미

```

### Multi-task Loss

- $*L(p,u,t^u,v) = L_{cls}(p,u) + \lambda[u\ge1]L_{loc}(t^u,v)*$
    - classification loss + bbox regression loss ⇒ multi-task loss
    $p=(p_0,...,p_K)$ : softmax를 통해 얻어낸 `K+1(K-classes + None)` 개의 확률값
        
        $u$ : 해당 RoI의 G.T 클래스 값
        
        > [u≥1]은 G.T label이 Background (u=0)이면 Lloc term 날림.
        🤷🏻? ⇒ Bounding Box를 치고자 하는 대상은 Background가 아닌 Object이기 때문이다.
        > 
        
        $t^u=(t_x^u,t_y^u, t_w^u,t_h^u)$ : bbox regression offset (예측한 bbox 좌표를 조정하는 값)
        
        $v$ : ground truth bbox 조절 값
        
    - $L_{cls}(p,u) = -\log p_u$
    - $L_{loc}(t^u,v) = \sum_{i\in{x,y,w,h}} smooth_{L_1}(t_i^u-v_i)$
        
        ![/images/Fast-RCNN/스크린샷_2021-07-27_오후_4.45.17.png](/images/Fast-RCNN/스크린샷_2021-07-27_오후_4.45.17.png)
        

# Experiments

1. State-of-the-art mAP on VOC07, 2010, and 2012
2. Fast training and testing compared to R-CNN, SPPnet
3. Fine-tuning conv layers in VGG16 improves mAP

![/images/Fast-RCNN/스크린샷_2021-07-27_오후_5.13.14.png](/images/Fast-RCNN/스크린샷_2021-07-27_오후_5.13.14.png)

**왜 성능이 더 좋아졌는가? RoI 풀링의 효과인가 정말?**

# Conclusion

1. 뛰어난 성능
2. End-to-End Single Stage Training
3. No disk storage is needed
    - R-CNN → CNN에서 나온 피쳐맵을 디스크에 저장해두고 classification, bbox regression할 때 불러옴

---

| **Faster R-CNN (NIPS 2015)**

# Main Idea

- Fast R-CNN drawbacks
    - region proposal이 차지하는 시간이 매우 길다 (test time:2.3sec, proposal time:2 sec)
    - real-time 으로는 부적합
- Region Proposal Network(RPN)
    - Selective Search를 사용하지 않고 RPN을 통해서 RoI를 계산함
    - GPU를 통한 RoI 연산이 가능해짐
    - 이 외에는 Fast R-CNN과 동일한 구조 (⇒ RoI pooling → classification & bbox regression)

# Faster R-CNN architecture

![/images/Fast-RCNN/스크린샷_2021-07-27_오후_5.34.38.png](/images/Fast-RCNN/스크린샷_2021-07-27_오후_5.34.38.png)

## Region Proposal Network(RPN) module

RPN의 목적

- 생성한 Region proposal이 Object 여부와 bbox의 좌표를 예측하면서 적절한 bbox를 생성하도록 학습하는 것

```markdown
💥 input : feature map
💥 output : object proposals
```

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)

[https://oi.readthedocs.io/en/latest/computer_vision/object_detection/faster_r-cnn.html](https://oi.readthedocs.io/en/latest/computer_vision/object_detection/faster_r-cnn.html)

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)

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
        
        ![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)
        
    - k=9 ⇒ 3개의 scale(8,16,32) X 3개의 ratio(0.5,1,2)
- `1x1 conv` 를 `(2(obj/No obj.) x 9(Anchor 개수 = k)) 채널 수` 만큼 수행하며, 그 결과로 H*W*18(2x9) 크기의 feature map을 얻음.
- 18개의 채널 → 앵커박스 각각에 대한 obj, no.obj에 예측값
    - G.T Label → 만들어진 모든 Anchor들과 G.T Box의 IoU를 계산하여 결정
        - IoU > 0.7 : 1(Positive), G.T Box마다 IoU가 가장 높은 Anchor : 1(Positive)
        IoU <0.3 : 0(Negative)

*⇒ 한번의 1x1 컨볼루션으로 Anchor들에 대한 예측을 모두 수행할 수 있다.* 

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

### Loss for RPN

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)

- 전체 Anchor 개수 중 i번째 anchor에 대한 Loss
- $L_{cls}$ : Log loss
    - G.T label = 0(==no obj)라면 뒤의 regression term은 날아감.
- $L_{reg}$ : Smooth L1 Loss
    - R-CNN과 동일

# Experiments

## Training

- 4-step alternating training strategy

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)

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

![/images/Fast-RCNN/스크린샷_2021-08-02_오전_4.04.11.png](/images/Fast-RCNN/스크린샷_2021-08-02_오전_4.04.11.png)

[https://www.youtube.com/watch?v=kcPAGIgBGRs&ab_channel=JinWonLee](https://www.youtube.com/watch?v=kcPAGIgBGRs&ab_channel=JinWonLee)



## Dataset

- PASCAL VOC 2007, 2012
    
    ![/images/Fast-RCNN/스크린샷_2021-07-28_오후_9.46.28.png](/images/Fast-RCNN/스크린샷_2021-07-28_오후_9.46.28.png)
    
    ![/images/Fast-RCNN/스크린샷_2021-07-28_오후_9.46.45.png](/images/Fast-RCNN/스크린샷_2021-07-28_오후_9.46.45.png)
    
    ![/images/Fast-RCNN/스크린샷_2021-08-01_오후_11.11.02.png](/images/Fast-RCNN/스크린샷_2021-08-01_오후_11.11.02.png)
    
- MS COCO
    
    ![/images/Fast-RCNN/스크린샷_2021-07-28_오후_9.47.26.png](/images/Fast-RCNN/스크린샷_2021-07-28_오후_9.47.26.png)
    

# Conclusion

- R-CNN, Fast R-CNN, Faster R-CNN 비교 사진

![/images/Fast-RCNN/Untitled.png](/images/Fast-RCNN/Untitled.png)

[https://brunch.co.kr/@kakao-it/66](https://brunch.co.kr/@kakao-it/66) (Kakao AI Report)

- Region Proposal이 Selective Search ⇒  RPN
- 이전보다 빠르고, 성능 Good
- 복잡한 트레이닝 과정
- 그래도 아직 real-time까지는...