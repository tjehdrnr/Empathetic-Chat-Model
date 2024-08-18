# Implementation of "[MyMary](https://github.com/boostcampaitech5/level3_nlp_finalproject-nlp-12)"
개인 프로젝트의 일환으로 위 링크의 챗봇을 구현해보았습니다.
본 레포지토리의 모든 학습, 추론 과정은 Google Colab T4 GPU(16GB Memory)를 이용하였습니다.

## Finetuning
Backbone Network:
1. [EEVE-Korean-2.8B-v1.0](https://huggingface.co/yanolja/EEVE-Korean-2.8B-v1.0)
2. [kullm-polyglot-5.8b-v2](https://huggingface.co/nlpai-lab/kullm-polyglot-5.8b-v2)
_GPU 메모리에 여유가 있으시다면 10B 이상의 모델을 사용하시는 것을 권장합니다._

Dataset:

[empathetic_dialogues_multi_turn_ko](ohilikeit/empathetic_dialogues_mutli_turn_ko)

