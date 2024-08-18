# Implementation of "[MyMary](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-12)"
개인 프로젝트의 일환으로 위 링크의 챗봇을 구현해 보았습니다.

본 Repo의 모든 학습, 추론 과정은 Google Colab T4 GPU(16GB Memory)를 이용하였습니다.

## Finetuning

### **Backbone Network:**
1. [EEVE-Korean-2.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-2.8B-v1.0)
2. [kullm-polyglot-5.8b-v2](https://huggingface.co/nlpai-lab/kullm-polyglot-5.8b-v2)

_GPU 메모리에 여유가 있으시다면 10B 이상의 모델을 사용하시는 것을 권장합니다._

### **Dataset:**

[empathetic_dialogues_multi_turn_ko](ohilikeit/empathetic_dialogues_mutli_turn_ko)

### **How to train**
1. finetune 디렉토리로 이동하여, 학습에 이용할 LLM을 명시한 후 다음과 같이 실행합니다.
```bash
cd finetune
python train.py --base_model eeve or kullm --add_eos_token
```
2. 학습이 완료되면 finetune/adapter_model 디렉토리에 학습된 LoRA adapter weight가 저장됩니다.
3. 다음 실행을 통해 LLM과 adapter 모델을 결합하면 finetune/merged_model 디렉토리에 모델 weight 파일들이 저장됩니다.
```bash
python merge_lora.py --base_model eeve or kullm
```
4. 다음 실행을 통해 학습된 모델을 간단하게 테스트해 보실 수 있습니다.
```bash
python conversation.py
```


## Demo: Streamlit

모델의 모의 service를 위해 Streamlit을 활용합니다.

실제 service를 위해서는 사용자별 session을 구분해 주는 로직이 필수적이지만,

Streamlit은 자동으로 session을 구분해 주기 때문에 시연용으로 사용하기 간편합니다.

Demo app에는 채팅 및 메시지 관리 기능에 더해 다음 두 가지의 기능이 추가되었습니다.

1. 추론 hyper-parameter 조정 기능

temperature, top_p 등 추론에 민감한 영향을 미치는 parameter들을 실시간으로 조절할 수 있습니다.


```bash
streamlit run app.py
```
---



