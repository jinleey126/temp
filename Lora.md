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
  

### 제안 방법: LoRA

4.1 Low-Rank-Parametrized Update Matrices

신경망은 행렬곱을 수행하는 많은 dense layer들을 포함한다. 이 layer들 안에 있는 가중치 행렬은 전형적으로 full-rank이다. 특정한 작업에 adpat할 때, Aghajanyan et al., 2020는 사전학습된 모델이 적은 "instrisic dimension"을 가지며, 작은 subspace에서의 랜덤한 투사임에도 불구하고 여전히 효과적으로 학습할 수 있음을 보여주었다. 이 주장에 영향을 받아, 저자는 adaptation하는 동안 가중치 업데이트 또한 작은 "intrinsic rank"를 가질것이라 가정했다. 사전학습된 모델의 가중치 행렬을 $W_0$, 저자는 이 행렬의 업데이트를 latter with a low-rank decomposition $W_0 \Delta W = W_0 + BA$로 표현했다. 


**A Generalization of Full Fine-tuning** (풀 파인튜닝의 일반화)
파인 튜닝의 더 일반화된 형태는 사전학습된 파라미터의 부분집합을 학습할 수 있게 해준다. LoRA는 adptation하는 동안 full-rank를 가진 weight 행렬에 대한 accumulated gradient update가 필요하지 않기 때문에 더 큰 step을 가진다(업데이트 시 이동하는 거리가 더 멀다는 뜻인듯). 이는 모든 weight 행렬에 LoRA를 적용하고 모든 biases를 학습했을 때, 사전학습된 weight 행렬의 랭크를 LoRA 랭크 r로 세팅함으로써 풀 파인튜닝의 표현을 recover한다고 볼 수 있다. 다시 말해, 학습가능한 파라미터의 수가 증가할수록, LoRA 학습은 대충 원래 모델의 학습에 수렴한다고 본다. 반면, adapter 기반 방법은 MLP로 수렴하고, prefix 기반 방법은 long input sequence를 처리할 수 없다.

**No Additional Inference Latency** (추가적인 추론 지연시간 없음)

생성할 때 저자는 explicity $W$를 계산하고 저장할 수 있고 보통 추론을 수행한다. $W_0$와 $BA$ 모두 $\mathcal{R}^{d \times k$에 존재한다. 저자는 다른 수행가능한 과제로 전환해야할 때, $W_0$를 $BA$를 빼고 다른 $B`A`$를 더하는 것으로 커버할 수 있는데, 아주 적은 메모리 오버헤드를 가진 빠른 operation이다. 이는 파인튜닝 모델과 비교했을 때 추론 과정에서 어떠한 추가적인 지연시간도 없을을 설명하는데 충분하다.

4.2 Applying LoRA to Transformer

원칙적으로 저자는 학습가능한 파라미터의 수를 줄이기 위해 신경망 내의 어떤 가중치 행렬에도 LoRA를 적용할 수 있다. 트랜스포머 구조에서, 셀프 어텐션 모듈 내에 4개의 가중치 행렬($W_q, W_k, W_v, W_o$)이 있고, MLP 모듈에는 2개가 존재한다. 저자는 차원 $d_\textnormal{model} \tiems d_\textnormal{model}$인 단일 행렬 $W_q$ (또는 $W_k, W_v$)를 다룬다. 비록 어텐션 헤드 내 주로 슬라이스되는 것은 출력 차원일지라도 말이다. 저자는 본 연구를 실제 수행할 작업에 대한 어텐션 가중치에만 adapting하는 것으로 제한하고 MLP 모듈을 동결(해당 부분은 직접 수행할 작업에 대해 학습되지 않는다)한다.(simplicity와 parameter-efficiency를 위해) 저자는 섹션 7.1에서 트랜스포머 내 어텐션 가중치 행렬의 다른 유형을 adapting하는 효과에 대한 연구도 수행하였다. 저자는 MLP layer, LayerNorm Layer, biases에 대한 adapting은 추후 연구로 남긴다.
  

**Practical Benefits and Limitations** : 가장 중요한 이점은 reduction in memory and storage usage. Adam으로 학습된 큰 트랜스포머에서, VRAM 사용량은 2/3까지 줄였다.(r이 모델 차원 d_model 보다 작았을 경우 동결된 파라미터에 대한 optimizer 상태를 저장할 필요가 없기 때문에) GPT-3 175B에서, 학습하는 동안의 VRAM consumption을 1.2TB->350GM까지 줄였음. $r=4$이고 query와 value projection matrices만 adaptation에서 사용될 경우 체크초인트 크기는 대충 $10000 \times$ (from 350GB to 35MB)까지 줄어든다. 이는 GPU를 덜 사용하고 I/O 병목을 피하면서 학습할 수 있게끔 해준다.
다른 장점은 작업들 간 전환을 LoRA 가중치만 바꿔 훨씬 적은 비용으로 전개할 수 있다는 것이다.
이를 통해 사전 훈련된 가중치를 VRAM에 저장하는 기기에서 즉시 교체할 수 있는 많은 맞춤형 모델을 생성할 수 있다.
또한 풀 파인튜닝과 비교했을 때 GPT-3 175B의 학습 속도가 25% 가까이 증가했음을 관찰하였다. 

LoRA에도 한계가 있다. 예를 들어, 추가적인 추론 지연을 제거하기 위해 A와 B를 W에 흡수하기로 선택한 경우, 서로 다른 A와 B를 가진 다양한 작업에 대한 입력을 단일 순전파에서 일괄 처리하는 것이 간단하지 않다. 비록 지연 시간이 중요하지 않을 경우 가중치를 병합하지 않고 배치 내 샘플에 사용할 LoRA 을 동적으로 선택하는 것이 가능하긴 하지만 말이다.

### Empirical Experiments

GPT-3 175B로 크기를 키우기 전에, GPT-2, DeBERTa, RoBERTa를 LoRA로각 실제 수행 작업에 학습하여 성능을 평가하였다.
실험은 from natural language understanding(NLU) to generation(NLG)
구체적으로, 우리는 직접적인 비교를 위해 GPT-2에 적용한 Li&Liang(2021)의 setup을 따랐고, WikiSQL(NL to SQL queries), SAMSum(대화 요약)을 추가했다. 사용한 데이터셋에 대한 상세한 내용은 부록 C에 정리되어있으며, 저자는 모든 실험을 위해 NVIDIA Tesla V100을 사용했다.

5.1 BASELINES

다른 baseline들과 많이 비교하기 위해, 이전 연구의 셋업을 그대로 사용했는데, 그래서 때에 따라 몇몇의 작업에선 특정 baseline들의 결과만 존재한다.

**Fine-Tuning(FT)**는 adapation을 위한 일반적인 접근 방법이다. 파인튜닝하는 동안, 모델은 사전학습된 weight와 bias로 초기화되고, 모든 모델 파라미터는 gradient 업데이트에 맡겨진다. 간단한 variant는 다른 layer들을 동결하고 몇몇의 layer만 업데이트하는 것이다. 저자는 GPT-2를 사용한 이전 연구(Li&Liang,2021: 마지막 두 개의 layer에만 adapts하였음, FT^(Top2)) baseline을 포함하여 기록하고 있다.

**Bias-only or BitFit** : 다른 모든 것들은 동결하고 bias 벡터들만 학습한 baseline, 이전 연구 BitFit(Zaken et al., 2021)

**Prefix-embedding tuning(PreEmbed)** : 입력 토큰 사이에 special 토큰을 넣는 방법. 이 special token들은 학습가능한 워드 임베딩을 가지며, 일반적으로 모델의 단어 내에는 없음. 그러한 토큰들이 있을 경우, 모델 성능에 영향을 미칠 수 있기 때문. 저자는 "prefixing"(프롬프트에서 그러한 토큰을 앞에 추가)과 "infixing"(프롬프트 뒤에 붙임)에 초점을 맞춘다. 저자는 prefix 토큰의 수를 $l_p$로, infix 토큰의 수를 $l_i$로 정의하였다. 학습가능한 파라미터의 수는 $|\Theta|=d_\textnormal{model} \times (l_p+l_i)$이다.

**Prefix-layer tuning (PreLayer)** : prefix-embedding tunint의 연장선. 단순히 몇몇 special 토큰들에 대해 워드 임베딩을 학습하는 것 대신에 (또는 임베딩 레이어 다음 activations과 동등함), 저자는 모든 트랜스포머 레이어 뒤의 activations을 학습한다. 이 activations은 단순히 학습가능한 것으로 교체된 이전 레이어로 부터 계산된다. 학습가능한 파라미터의 결과값은 $|\Theta|= L \times d_{model} \times (l_p+l_i)$이다. $L$은 트랜스포머 레이어의 수. 

**Adapter tuning** : (Houlsby et al., 2019)에서 제안. 셀프 어텐션 모듈(그리고 MLP 모듈)과 하위 residual connection 사이에 adapter layer를 추가한다. adapter layer내 두 개의 fully connected layers가 있다(서로 비선형인). 우리는 이 original design을 $Adapter^H$라 한다.

최근, Lin et al(2020)는 MLP 모듈 다음과 LayerNorm 다음에 adapter layer를 적용하는 더 효과적인 desing을 제안했다. 우리는 이를 $Adapter^L$이라 한다. 이와 굉장히 비슷한 또다른 design( Pfeiffer et al.(2021)이 제안)을 $Adapter^P$라 한다. 또한, 저자는 다른 baseline AdapterDrop(더 효율성을 추구하기 위해 몇몇의 adapter layer를 버리는(drop)) $Adapter^D$도 포함한다. 

저자는 저자가 비교하려고 하는 baseline의 수를 최대로 늘리기 위해서 이전 연구에서 사용된 숫자들을 가져가서 사용했다. 첫번째 컬럼 내의 asterik(*)가 달린 값들이 그러하다.
모든 경우에서 저자는 $|\Theta|=$이며, 는 adapter 레이어의 수이고, 는 학습가능한 LayerNorm의 수를 의미한다.

**LoRA** : 존재하는 weight matrices에 병렬적으로 학습가능한 rank decomposition matrices 쌍을 더해준다. 섹션 4.2에 언급했듯이 간소화하기 위해 대부분의 실험에서 $W_q$와 $W_v$에만 LoRA를 적용하였다. 학습가능한 파라미터의 수는 랭크 $r$과 원래 weight의 shape에 따라 결정된다: $|\Theta|=2 \times \hat{L}_{LoRA} \times d_{model} \times r$. $\hat{L}_{LoRA}$는 LoRA를 적용할 weight matrices의 수이다.

5.2 RoBERTa base/large

RoBERTa (Liu et al., 2019)는 BERT (Devlin et al., 2019a)에서 원래 제안된 사전 학습 방식을 최적화하여 학습 가능한 매개변수를 추가하지 않고도 후자 작업 성능을 향상시켰다. 비록  GLUE 벤치마크(Wang et al., 2019)와 같은 NLP 리더보드에서 큰 모델들에 의해 추월되었지만, 실무자 입장에선 크기 대비 경쟁력있기에 여전히 인기있는 사전학습 모델이다. 저자는 HuggingFace Transformers(Wolf et al., 2020)에서 사전 학습된 RoBERTa base (125M)와 RoBERTa large (355M)를 가져와 GLUE 벤치마크의 작업에 대해 다양한 효율적인 adapatation 방법들의 성능을 평가했다.  공정한 비교를 위해, LoRA는 모든 작업에 대해 동일한 배치 크기를 사용하고 어댑터 기준선과 일치하도록 128의 시퀀스 길이를 사용하고, MRPC, RTE, STS-B에 대해 모델을 미세 조정 기준선처럼 이미 MNLI에 적응된 모델이 아닌 사전 학습된 모델로 초기화했다. Houlsby et al. (2019)의 이 더 제한된 설정을 따르는 실행은 †로 표시했다. 결과는 (표 2).  하이퍼파라미터에 대한 자세한 내용은 (섹션 D.1).

5.3 DEBERTA XXL
DeBERTa (He et al., 2021)는 BERT의 더 최근 변형으로, 훨씬 더 큰 규모로 훈련되었으며 GLUE (Wang et al., 2019)와 SuperGLUE (Wang et al., 2020)와 같은 벤치마크에서 매우 경쟁력 있는 성능을 보입니다. 우리는 LoRA가 GLUE에서 완전히 미세 조정된 DeBERTa XXL (1.5B)의 성능과 여전히 일치할 수 있는지 평가합니다. 결과는 표 2(하단 섹션)에 제시되어 있습니다. 사용된 하이퍼파라미터에 대한 자세한 내용은 섹션 D.2를 참조하십시오.

5.4 GPT-2 MEDIUM/LARGE
LoRA가 NLU에서 완전한 미세 조정의 경쟁력 있는 대안이 될 수 있음을 보여준 후, 우리는 LoRA가 GPT-2 medium과 large (Radford et al., b)와 같은 NLG 모델에서도 여전히 우세한지 알아보고자 합니다. 우리는 직접적인 비교를 위해 Li & Liang (2021)의 설정과 최대한 가깝게 유지합니다. 공간 제약으로 인해 이 섹션에서는 E2E NLG Challenge에 대한 결과(표 3)만 제시합니다. WebNLG (Gardent et al., 2017)와 DART (Nan et al., 2020)에 대한 결과는 섹션 F.1을 참조하십시오. 섹션 D.3에 사용된 하이퍼파라미터 목록을 포함했습니다.

5.5 GPT-3 175B로의 확장

LoRA에 대한 최종 스트레스 테스트로, 우리는 1750억 개의 매개변수를 가진 GPT-3까지 확장합니다. 높은 훈련 비용으로 인해, 우리는 모든 항목에 대해 하나씩 제공하는 대신 무작위 시드에 대해 주어진 작업의 일반적인 표준 편차만 보고합니다. 사용된 하이퍼파라미터에 대한 자세한 내용은 섹션 D.4를 참조하십시오.

표 4에서 보여지듯이, LoRA는 세 가지 데이터셋 모두에서 미세 조정 기준선과 일치하거나 초과합니다. 그림 2에서 보여지듯이 모든 방법이 더 많은 훈련 가능한 매개변수를 가짐으로써 단조롭게 이익을 얻는 것은 아님을 주목하십시오. 우리는 접두사-임베딩 튜닝에 256개 이상의 특수 토큰을 사용하거나 접두사-레이어 튜닝에 32개 이상의 특수 토큰을 사용할 때 상당한 성능 하락을 관찰합니다. 이는 Li & Liang (2021)의 유사한 관찰을 뒷받침합니다. 이 현상에 대한 철저한 조사는 이 연구의 범위를 벗어나지만, 우리는 더 많은 특수 토큰을 사용하면 입력 분포가 사전 훈련 데이터 분포에서 더 멀어지는 원인이 될 수 있다고 의심합니다. 별도로, 우리는 섹션 F.3에서 적은 데이터 체제에서 다양한 적응 접근법의 성능을 조사합니다.


### 관련 연구

**Transformer Language Models** : 일반 도메인 데이터로 먼저 모델을 사전학습시킨 다음, 작업에 특화된 데이터로 파인 튜닝하는 것이 작업에 특화된 데이털 바로 학습하는 것보다 훨씬 더 성능이 좋다.

**Prompt Engineering and Fine-Tuning** : GPT-3 175B 모델이 몇개의 추가적인 학습 예제만으로도 특정 행동을 adapt할 수 있긴 하지만, 그 결과는 입력 프롬프트에 의해 크게 좌우된다. 그래서, 프롬프트를 구성하고 포맷팅하는 것은 중요하다. 파인튜닝은 특정 작업을 더 잘 수행하도록 재학습하는 것인데, GPT-3 175B와 같이 모델의 크기 어마어마할 경우 학습 자체의 장벽이 높아진다.

**Parameter-Efficient Adaptation** :많은 연구자들이 신경망의 기존 계층 사이에 *어댑터* 계층을 삽입하는 방법을 제안했습니다(Houlsby et al., 2019; Rebuffi et al., 2017; Lin et al., 2020). 우리의 방법은 가중치 업데이트에 저순위 제약을 가하기 위해 유사한 병목 구조를 사용합니다. 주요 기능적 차이점은 우리의 학습된 가중치가 추론 중에 주 가중치와 병합될 수 있어 지연 시간을 도입하지 않는다는 것입니다. 이는 어댑터 계층의 경우와는 다릅니다(섹션 3).

어댑터의 현대적 확장인 COMPACTER (Mahabadi et al., 2021)는 기본적으로 미리 정해진 가중치 공유 방식으로 크로네커 곱을 사용하여 어댑터 계층을 매개변수화합니다. 마찬가지로, LoRA를 다른 텐서 곱 기반 방법과 결합하면 매개변수 효율성을 잠재적으로 개선할 수 있습니다. 이는 향후 연구 과제로 남겨둡니다.

최근에는 미세 조정 대신 입력 단어 임베딩을 최적화하는 방법이 많이 제안되었습니다. 이는 프롬프트 엔지니어링의 연속적이고 미분 가능한 일반화와 유사합니다(Li & Liang, 2021; Lester et al., 2021; Hambardzumyan et al., 2020; Liu et al., 2021). 우리는 실험 섹션에서 Li & Liang (2021)과의 비교를 포함했습니다. 

그러나 이러한 연구 방향은 프롬프트에서 더 많은 특수 토큰을 사용함으로써만 확장할 수 있습니다. 이는 위치 임베딩이 학습될 때 작업 토큰에 사용 가능한 시퀀스 길이를 차지하게 됩니다.

**Low-Rank Structure in Deep LEearning** : 네, 이해했습니다. 원문을 한국어로 번역해 드리겠습니다:

저순위 구조는 기계 학습에서 매우 흔합니다. 많은 기계 학습 문제들이 특정한 본질적 저순위 구조를 가지고 있습니다(Li et al., 2016; Cai et al., 2010; Li et al., 2018b; Grasedyck et al., 2013). 더욱이, 많은 딥러닝 작업, 특히 과도하게 매개변수화된 신경망을 가진 작업의 경우, 학습된 신경망이 훈련 후 저순위 특성을 갖게 될 것으로 알려져 있습니다(Oymak et al., 2019). 일부 선행 연구에서는 원래 신경망을 훈련할 때 명시적으로 저순위 제약을 부과하기도 했습니다(Sainath et al., 2013; Povey et al., 2018; Zhang et al., 2014; Jaderberg et al., 2014; Zhao et al., 2016; Khodak et al., 2021; Denil et al., 2014). 그러나 우리가 아는 한, 이러한 연구 중 어느 것도 *다운스트림 작업에 대한 적응*을 위해 동결된 모델에 대한 저순위 업데이트를 고려하지 않았습니다. 

이론 문헌에서는 기본 개념 클래스가 특정 저순위 구조를 가질 때 신경망이 다른 고전적 학습 방법들, 그리고 해당하는 (유한 너비) 신경 접선 커널들보다 우수한 성능을 보인다고 알려져 있습니다(Allen-Zhu et al., 2019; Li & Liang, 2018; Ghorbani et al., 2020; Allen-Zhu & Li, 2019; Allen-Zhu & Li, 2020a). Allen-Zhu & Li (2020b)의 또 다른 이론적 결과는 저순위 적응이 적대적 훈련에 유용할 수 있음을 시사합니다. 요약하면, 우리는 우리가 제안한 저순위 적응 업데이트가 문헌에 의해 잘 뒷받침된다고 믿습니다.

### Low-Rank Update에 대한 이해

LoRA에 대한 경험적인 장점들이 주어졌을 때, 저자는 더 나아가 실제 수행할 작업들로부터 학습된 low-rank adaptation의 특성들을
설명하길 원했다. low-rank 구조는 병렬적으로 multiple experiments를 돌릴 수 있도록 하여 하드웨어 진입장벽을 낮출뿐만 아니라, 업데이트 가중치가 사전학습된 가중치와 어떻게 연관되어있는지에 대한 더 나은 설명가능성을 제공해준다. 저자는 GPT-3 175B 모델을 기준으로 작업 성능에 부정적인 영향을 미치지 않으면서 훈련 가능한 매개변수의 가장 큰 감소(최대 10,000배)를 달성했다.

저자는 다음의 질문에 대한 답하기 위해 실증적 연구를 수행했다:
1) 주어진 파라미터 예산에 대한 제약이 있을 때(parameter budget constraint), 실제 수행할 작업에서의 성능을 최대화하기 위해 사전학습된 transformer의 "어떤 가중치 행렬 부분집합"에 적용시켜야 하는지? 
2) "optimal" adaptation 행렬 ∆W가 실제로 rank-deficient한지? 만약 그렇다면 실제로 사용하기 좋은 rank는 무엇인지?
3) ∆W와 W 사이의 connection은 무엇? ∆W가 W와 높은 상관관계를 가지는지? W와 비교했을 때 ∆W의 크기는 어느 정도인지?
저자는 질문(2),(3)에 대한 적절한 답이 실제 수행할 작업을 더 잘 수행할 모델 학습에 사전학습된 언어 모델을 사용하는 근본적인 원리를 찾아줄 것이라 믿었음.

7.1 Which weight matrices in transformer should we apply LoRA to?



7.2 
### 결론






