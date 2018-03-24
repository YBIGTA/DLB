### Coursera - Deep Learning Specializaion 
#### [Coures 2] Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization
<br/>

# (Week 1) Practical aspects of Deep Learning
----
두둥!!!Course 2가 시작되었습니다!!     
- Course 1에서는 기본적인 Deep Neural Network를 만들고 Gradient Descent로 학습시켜서 고양이인지 맞추는 Binary Classification을 배웠습니다.        
 - Course 2에서는 Deep Neural Network의 성능을 향상 시키는 방법들을 배울 것입니다. 
 - Week 1에서 다루게 될 내용들은 다음과 같습니다.
 
	 1. train/dev/test set  
	 2. bias and variance  
		 (1)Regularization  
		 (2)Drop Out  
		 (3)Other method
	  3. Optimization
		 (1)Normalizing Inputs  
		 (2)Weight Initialization
		 (3)Gradient Checking
----
<br/>  

## 0. Hyper Parameters  and Orthogonalization
<br/>
  
- Deep Neural Network의 성능을 올리려면 Hyper Parameter들을 잘 조정해주어야 한다. 
- Hyper Parameter에는 # of Layers, # of hidden units, learning rates, activation function, Regularization parameter 등이 있다.
- 딥 러닝의 학습은 크게 두가지으로 나눠 진다.
	1. Optimization - SGD, ADAM 등
	2. Prevent Overfitting - Regularization 등  
이 두가지는 독립적인 도구들로 행해져야는데(?), 이러한 특징을 Orthogonalization이라 한다. 자세히는 아직 모릅니다....
<br/>
----
## 1. Train/ Dev/ Test sets  

- Dev sets은 Developement sets를 뜻하는데, Validation sets와 같은 의미이다.  
- Train sets로 모델을 학습시키고, Dev를 통해 모델의 성능을 파악한다(Cross Validation, Hyper Parameter 조정 등) 그리고 Test sets로 test한다.
- 기존의 머신 러닝은 데이터의 비율을 Train/ Dev/ Test sets 순서대로 60/20/20%로 나누는 것을 많이 사용했다고 한다.   
- 딥러닝에서는 데이터수가 매우 많기 때문에 98/1/1과 같이 Train sets의 비율을 매우 높게 한다.  
- Train/ Dev/ Test sets은 같은 분포에서 나온 데이터일수모델 학습에 좋다.  
- bias가 상관 없으면 Test sets을 사용 안할 수 있다.
----
<br/>

## 2. Bias and Variance - Over Fitting  
<br/>  

![Imgur](https://i.imgur.com/Z18xijC.png)
- bias : 모델이 얼마나 예측 못하는지라 생각하면 된다. train set error가 클수록 높다.  Logistic Regression과 같이 선형적인 모델들은 high bias를 가져 robust하다. high bias를 갖는 것은 under fitting을 뜻한다.  
- variance : 모델이 데이터에 얼마나 민감한 정도 또는 복잡도이다. variance가 크다는 것은 over fitting을 뜻하며, 다른 데이터에는 성능이 안좋다.  
- bias variance trade-off : bias와 variance는 반비례 관계라 trade-off가 필요하다.  즉!!! __bias를 낮춰 성능을 올리고!!! variance를 낮춰 over fitting을 막자!!!__
----
<br/>

![Imgur](https://i.imgur.com/VM8p5Ho.png)
- 정확도의 목표는 사람의 정확도이다.   
- Train set error와 Dev set error차이가 많이 나면 over fitting(high variance, low bias)된 것이고,  Train set error가목표 정확도보다 많이 낮고  Dev set error 차이가 작으면 uner fitting이다.
- 목표는 low bias, low variance!! 최악은 high bias, high variance...
----
<br/>

![Imgur](https://i.imgur.com/dZFr8HG.png)
- 그럼 딥러닝에서 bias와 variance는 어떻게 조정할까?
- __bias__는 network를 크게 만들수록 줄어든다. 일반적으로 신경망은 크기가 클수록 성능이 좋아진다. 또한 Train 횟수(iteration)을 늘리거나 다른 모델을 찾아보면 된다.  
- __varaince__는 더욱 많은 데이터를 학습시키거나 Regularization(정규화), 다른 모델을 사용함으로써 줄일 수 있다.  
![Imgur](https://i.imgur.com/46uKQbz.png)
데이터를 늘리는 방법에는 Augmetation이 있다. 기존의 데이터를 변형하여 Train sets에 추가하는 것이다.<br/>

- 딥러닝에서는 bias와 variance는 trade-off관계성이 작다.데이터의 수가 매우 많고 network가 크기 때문이다.
----
### bias는 모델의 크기를 크게하면(layer, unit수 조절)하면 되는 것이므로 variance를 줄여 over fitting을 막는 방법들을 살펴보겠다. <br/>

## (1) Regularization <br/>

![Imgur](https://i.imgur.com/jMIxnwu.png)
- weight에 제약을 줘서 weight의 값들이 작아지도록 하는 것이다. L2와 L1 regularization이 있는데 주로 L1을 사용한다. L2는 Norm(2)를 이용하여서 L2라고 한다. 위의 사진은 Logistic regression에서 Cost function에 weight 패널티를 추가한 것을 보여준다.<br/>
----

![Imgur](https://i.imgur.com/RtD5h9J.png)
- Cost function에 weight 패널티를 추가하면 Gradient Descent로 weight를 업데이트 할 때 식이 추가된다. 결과적으로 weight가 더욱 줄어들게 된다. <br/>  
----

![Imgur](https://i.imgur.com/DeFlks2.png)
- 그렇다면 weight의 norm이 줄어들면 왜 overfitting을 방지할 수 있을까?
- activate function이 tanh라 하면 tanh는 입력변수가 0근처일 때 rough한 선형이다. 따라서 weight가 작아지면 그에 따라 z도 작아지게 되어 activate function이 선형에 가까워 진다. 그러면 전체적으로 신경망이 단순해 지는 효과가 있다. 모델이 단순해진다는 것은 variance가 낮아진다는 것이다. <br/>
----
![Imgur](https://i.imgur.com/21GlbVo.png)
- 극단적으로 생각해보면 정규화 파라미터가 매우 커지면 weight들은 대부분 0으로 가게 된다. 그러면 위의 그림에서 볼 수 있듯이 모델이 매우 단순한 형태가 되어 high bias를 갖는다. 파라미터가 커질수록 bias가 커진다고 생각하면 된다. 하지만 variance가 더 크게 줄어들기 때문에 적절한 parameter를 찾으면 된다. 찾는 방법은 cross validation!!<br/>
----
## (2) Dropout <br/>


![Imgur](https://i.imgur.com/FS7zCYW.png)
- Dropout은 일부 unit을 keep prob(살리는 비율)하에서 Drop(죽이는...?)하는 것을 뜻한다. 죽이는 것은 layer마다, iteration마다 랜덤이고, 비율을 달리 할 수 있다.  
- keep prop은 Hyper parameter이고 cross validation으로 정한다. <br/>
----
![Imgur](https://i.imgur.com/Z3XCQLm.png)
- weight가 치중되어 있으면 치중되어 있는 unit이 죽었을 때 모델은 큰 영향을 받는다. 따라서 영향을 줄이기 위해 weight를 분산시키게 되고 그에 따라 weight의 norm 줄어들어든다. <br/>
----
![Imgur](https://i.imgur.com/6Ys1x8g.png)
- 가장 많이 쓰는 Dropout이 "Inverted dropout"이다. 응짱도 다른 Dropout은 잘 모른다고 했다. 
- unti이 죽으면 a값이 줄어들게 되는데, a를 keep prob으로 나눠 이를 방지한다.
- Dropout은 Cost 계산을 어렵게 하기 때문에 GD를 쓸 때는 사용하지 않는다.	
## (3) Early stopping <br/>

![Imgur](https://i.imgur.com/TaDsvfz.png)
- iteration 횟수를 조절하는 방법이다. 즉 dev set error가 최소가 되도록 일찍 멈추는 것이다.  
- 하지만 이방법은 Orthogonalization을 어기기 때문에 잘 쓰지 않는다.
- 장점은 iteration 횟수가 작기 때문에 컴퓨팅이 빠르다는 것....! <br/>

# 3. Optimization <br/>
## (1) Normalizing Inputs <br/>
![Imgur](https://i.imgur.com/YWOYMMK.png)
- 입력 데이터들의 scale을 맞춰주는 것을 Normalizing이라 한다. 통계에서 하는 표준화 그거임 ㅇㅇ. test도 train의 평균과 분산으로 Normalizing한다.
- 학습속도를 빨르게 하기 때문에 무조건 하는게 좋다. 밑져야 본전! (아래 사진 참고.)
![Imgur](https://i.imgur.com/vUqPLc6.png)
Normalize안해주면 코스트 함수가 기울어 지게 나와서 같은 Learning rate일 때 더 꼬불꼬불 오래 걸린다. <br/>
----
## (2)Weight Initialization<br/>
![Imgur](https://i.imgur.com/HcUsjeR.png)
- activation function에 맞게 weight 초기화를 해준다. 표준정규분포에서 뽑은 숫자에 특정 수를 곱해주어 분산을 조정한다.
- 초기화를 잘해주면 Gradients Vanishing/exploding에 도움이 된다.
- 하지만 다른 parameter 조정에 비해 덜 중요하다.
----
## *Vanishing/explding gradients<br/>

![Imgur](https://i.imgur.com/tilbOjM.png)
- layer를 지날 때마다 점점 수가 증폭되거나 줄어드는 것을 뜻한다.
- 위의 사진은 활성 함수가 선형일 경우 증폭되는 비율이 매우 큰것을 보여준다. 
- z의 절대값이 커지면 $g'(z)$가 0에 가까워 지므로(activation function에서 확인 가능) Gradient가 작아져서 학습이 느리게 된다. <br/>
----
## (4)Gradient checking<br/>
- Gradient checking을 통해 모델이 잘 구성되어 있는지 확인하는 것은 매우 중요하다. 시간을 엄청나게 절약해준다. 
- Gradient checking을 할 때는 Dropout을 하면 안된다. Cost 계산이 어려워 지기 때문이다. 따라서 모델이 잘 구성되어 있는지 확인하고 Dropout을 하면 된다.<br>
----
![Imgur](https://i.imgur.com/O94wCGs.png)
- W, b를 벡터로 reshape해서 concat하여 j번째 iteration마다 Θj라고 한다. 
- 마찬가지로 W,b의 gradient를 모아서 dΘj라고 한다. <br/>
---
![Imgur](https://i.imgur.com/hWevogI.png)
- Gradient 근사값을 구하는 과정을 간단히 하면 위와 같다. Θ로 부터 매우 작은 값만큼 크고 작은 값의 평균 기울기를 구한다. 간격이 좁을수록 더욱 근사한다.<br/>
----

![Imgur](https://i.imgur.com/QujpdYF.png)
- Check하는 부분이 입실론 만큼작으면 모델이 잘 구성되어 있는 것이고, 그것 보다 크면 잘못된 부분이 있는 것이다.
- Regularization이 있으면 gradient 업데이트가 다르므로 유의해야 한다.
- Dropout하고 같이 쓰면 안된다. 코스트 함수 계산이 어려워지기 때문에!!
![Imgur](https://i.imgur.com/1j5YsNV.png)


# 끝!









	 
 

	 
