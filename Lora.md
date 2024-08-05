# LoRA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS
released by `Microsoft` in `2021`
paper

### Abstract

자연어 처리에서 중요한 패러다임은 일반 도메인 데이터의 초거대 규모 사전학습과 특정 작업이나 도메인에 대한 적용이다. 사전학습한 모델이 커질수록 해당 모델에 대한 풀-파인튜닝(full fine-tuning)이 어려워진다(모델의 파라미터가 너무 많다보니 학습 비용이 크게 증가하기 때문이다).

LoRA(Low-Rank Adaptation)는 사전학습된 모델의 가중치를 freeze하고 학습가능한 rank decomposition matrices를 트랜스포머 구조의 각 레이어에 주입하여 실제 수행할 작업에 대한 학습가능한 파라미터 수를 크게 줄인다.

`Adam`과 `LoRA`로 각각 파인튜닝한 GPT-3 175B 모델을 비교하면, 학습가능한 파라미터의 수는 10,000배, GPU 메모리 필요량도 3배 가까이 줄어들었다. LoRA는 학습가능한 파라미터의 수가 더 적고, 학습 처리량이 더 높으며, 어댑터와 달리 'additional inference latency'가 없음에도 불구하고 RoBERTa, DeBERTa, GPT-2, GPT-3을 파인튜닝한 것보다 성능이 좋거나 동등한 수준을 보여주었다. 

본 논문은 언어 모델 적응(adaptation)에서의 rank-deficiency에 대한 실험을 통해 LoRA의 효과를 입증한다.

github : https://github.com/microsoft/LoRA


### Introduction

Many application in NLP
  one large-scale pre-trainined model -> multiple downstream applications

이러한 adpatation은 주로 사전학습된 모델의 모든 파라미터를 업데이트하는 fine-tuning을 통해 이루어짐. fine-tuning의 주요 단점은 새로운 모델도 원 모델만큼의 파라미터를 가진다는 것이다. GPT-2나 RoBERTa large에서는 단지 "inconvenience"에서 175B 파라미터를 지닌 GPT-3에서는 치명적이다.

![LoRA_reparametrization](https://github.com/user-attachments/assets/bdc54e93-7a8d-4e3a-ba9e-809258a3c7db)

새로운 작업에 대해 외부 모듈을 학습한다던가 몇몇의 파라미터만 채택(적용)하는 등의 방식으로 해당 문제를 완화시키려 했었다. 이 방식은 작업에 특화된 파라미터만 저장하고 로드하면 되기에, deploy할 때 엄청난 계산적 효율성을 보였다. 그러나,모델의 깊이를 확장시키거나 모델의 사용가능한 시퀀스 길이를 줄이는 등의 inference latency가 존재한다. 더 중요한 점은, 이 방법들이 종종 파인튜닝 baseline에 미치지 못해, 효율성과 모델 품질 사이의 트레이드오프를 야기한다는 것이다.

Li et al. (2018a)(학습된 over-parametrized 모델들이 사실 낮은 본질적 차원에 위치한다는 것을 보여줌)에서 영감을 얻어, 모델 adaptation하는 동안의 가중치 변화 또한 low intrinsic dimension을 가진다는 가설을 세웠고, Low-Rank Adaptation (LoRA) 방법으로 발전시켰다.

LoRA는 사전 훈련된 가중치를 동결한 상태에서, 적응 과정에서 발생하는 dense layer의 변화를 rank decomposition matrices로 최적화함으로써 신경망의 일부 dense layer를 간접적으로 학습시킨다 (그림1 참고). GPT-3 175B를 예로 들면, 전체 랭크(d)가 12,288로 매우 높은 경우에도 매우 낮은 랭크(즉, 그림 1의 r이 1 또는 2)로도 충분하다는 것을 보여주기 때문에 LoRA는 저장 공간과 계산 측면에서 모두 효율적이다.

LoRA의 주요 장점
- 사전학습된 모델은 다양한 작업의 많은 소규모 LoRA 모듈을 빌드하고 공유하는데 사용된다. 저자는 그림1에 보이듯이 행렬 $A$와 $B$를 바꾸는 방식으로 작업들을 효과적으로 변경하며 공유된 모델을 동결할 수 있다. (이는 저장 공간 요구사항과 작업 전환 오버헤드를 상당히 줄인다.)
- 대부분의 파라미터에 대한 optimizer state를 유지하거나 gradients 계산 필요성이 없기 때문에 adaptive optimizers를 사용했을 때보다 최대 3배까지 더 효율적으로 학습하기 때문에 하드웨어 진입 장벽을 낮춘다. 대신, 주입된 훨씬 더 작은 저차원 행렬만을 최적화한다.
- 단순한 선형 설계는 배포 시 학습 가능한 행렬을 고정된 가중치와 병합할 수 있게 해주며, 이는 구조적으로 완전히 미세 조정된 모델과 비교하여 추론 지연 시간이 없다. latency (지연 시간)
- 이전의 많은 방법들에 orthogonal(직교)하므로, 그 방법들(예시, prefix-tuning)과 결합할 수 있다. [부록E]

Terminologies and Conventions
트랜스포머 구조에 대한 빈번한 레퍼런스를 만들고, 해당 차원에 대한 통상적인 용어를 사용한다.
입력과 출력 차원 크니는 트랜스포머 레이어 $d_{model}$.
$W_q$ q,k,v,o는 각각 셀프 어텐션 모듈 내의 query/key/value/output projection matrices를 의미한다.
$W$나 $W_0$는 사전학습된 weight matrix
$W$ 축적된 gradient update during adaptation
$r$ LoRA 모듈의 랭크
모델 optimization을 위해 Adam을 사용하고, 트랜스포머 MLP feedforward 차원 $d_{ffn}=4\times d_{model}$


### 문제 정의


[언어 모델링 문제에 대한 간단한 설명]

GPT와 같은 사전학습된 auto-regressive 언어 모델 $P_{\Phi}(y|x)$가 있다고 가정하자.
이 모델을 (text generation: summarixation, machine reading comprehension(MRC), natural language to SQL(NL2SQL))과 같은 downstream task로 adpating한다고 하면,
각각의 downstream task는 context-target쌍으로 구성된 학습 데이터에 의해 표현된다: $\mathcal{Z}={(x_i,y_i)}_{i=1,...,N}$ ($x_i$와 $y_i$는 모두 토큰들의 시퀀스이다.).
예를 들어, NL2SQL는 $x_i$가 자연어 쿼리이고 $y_i$가 그에 해당되는 SQL명령어라 볼 수 있다.

full fine-tuning하는 동안, 모델은 사전학습된 weights $\Phi_0$으로 초기화되고 아래의 조건부 언어 모델링 목적식을 최대화하도록 하는 gradient에 따라 반복 수행하여 $\Phi_0+\Delta \Phi$로 업데이트된다.

식 (1)
$$
\textnormal{max}_\Phi \sum _{(x,y)\in\mathcal{Z}} \sum _{t=1} ^{|y|} \textnormal{log} (p _{\Phi_0} + \Delta \Phi(\Theta) (y_t | x,y))
$$

main drawbacks for full fine-tuning : 각각의 downstream 작업마다 다른 사전학습 파라미터의 차원과 같은 크기의 파라미터 값을 학습해야한다. 따라서, 사전학습된 모델이 큰 경우(예: GPT-3의 |Φ0| ≈ 1,750억), 미세 조정된 모델의 많은 독립적인 인스턴스를 저장하고 배포하는 것은 실현 가능하다 하더라도 어려울 수 있다.
본 논문은 파라미터-효율적인 방법을 채택했다. 작업별 파라미터 증분 ∆Φ = ∆Φ(Θ)는 |Θ| ≪ |Φ0|인 훨씬 더 작은 크기의 파라미터 집합 Θ로 인코딩된다. 따라서 ∆Φ를 찾는 작업은 Θ에 대한 최적화를 의미한다.

식 (2)

계산과 메모리 측면에서 모두 효율적인 ∆Φ를 인코딩하기 위해 저차원 표현을 사용하는데, 이는 사전 훈련된 모델이 GPT-3 1,750억일 때, 훈련 가능한 매개변수의 수 |Θ|를 |Φ0|의 0.01% 만큼 작을 수 있다.

### 현존하는 방법이 왜 충분하지 않았는지?

위의 문제는 새로 정의된 것이 아니다. transfer learning(전이 학습)의 inception 때문에 이미 여러 연구들의 모델 adaptation을 더 parameter- 그리고 compute-efficient한 방향으로 추구해왔고, 관련 연구는 섹션6에 소개된다. 대표적으로 효과적인 adaptations측면에서 adpater layer를 더하는 것과 input layer activations의 형태롤 최적화하는 전략이 있어왔다. 그러나, 두 전략 모두 거대한 규모의 그리고 지연 시간에 민감한 작업에서 특히 한계를 보였다.

- Adapter Layers Introduce Inference Latency

  original design (Houlsby et al., 2019) : two adapter layers per Transformer block
  
  recent one (Lin et al., 2020) : only one per block but with an additional LayerNorm

  layer를 가지치기한다던가 multi-task setting을 exploit하는 방식으로 전체 지연시간을 줄였지만, adapter layer내에서의 추가적인 연산을 우회하는 직접적인 방법을 없었다.

  adapter layer는 몇몇의 파라미터만 가지도록 설계되어 추가할 수 있는 FLOPs를 제한하므로 문제가 되지 않을 것처럼 보이겠지만, 대규모 신경망은 하드웨어 parallelism에 의존하며, adapter layer는 순차적으로 처리되어야 하기 때문에 단일 GPU로 추론을 하게 되면 adapter layer를 적용했을 때 지연시간이 크게 늘어남을 확인할 수 있다. (표1)

  이 문제는 모델을 샤딩(sharding, 전체 네트워크를 분할한 뒤 트랜잭션을 영역별로 저장하고 이를 병렬적으로 처리하여 블록체인에 확장성을 부여하는 온체인 솔루션으로 데이터를 샤드라는 단위로 나눠서 저장 및 처리)해야할 때 더욱 악화됐다. adapter 파라미터를 중복적으로 여러 번 저장하지 않는 한, 추가된 깊이로 인해 AllReduce와 Broadcast 같은 더 많은 동기적 GPU 연산이 필요하기 때문이다.

  
  
- Directly Optimizing the Prompt is Hard

  prefix tuning는 최적화하기 어렵다. 학습가능한 파라미터 내에서 일정한 규칙성없이 성능이 변하기 때문이다. adapatation을 위한 시퀀스 길이의 일부를 차지하기 때문에, 실제 수행하려는 작업을 처리하기 위해 사용가능한 시퀀스 길이가 줄어들게 된다. 그래서 저자는 오히려 다른 방법과 비교하여 프롬프트 튜닝이 덜 동작할 것이라고 의심했다. (섹션 5)
  

### 제안 방법


### Empirical Experiments

### 관련 연구

### Low-Rank Update에 대한 이해

### 결론과 논의






