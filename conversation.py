import os
from typing import Union, List
import torch
from argparse import ArgumentParser
from utils.arguments import Arguments
from utils.prompter import Prompter
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


class EmpatheticChatbot:
    def __init__(
            self,
            config: ArgumentParser,
    ):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.config.saved_model_dir,
                device_map={"":0},
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.saved_model_dir,
                add_special_tokens=False,
            )
        except FileNotFoundError:
            pass
        
        prompter = Prompter(template_name="multi")
        model.eval()

        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.prompter = prompter


    def get_reponse(self, instruction: Union[str]) -> Union[str]:
        prompt = self.prompter.generate_prompt(
            instruction=instruction,
            label=None,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            return_token_type_ids=False,
        )

        result = self.model.generate(
            **inputs,
            streamer=TextStreamer(self.tokenizer),
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_new_tokens,
        )

        response = self.tokenizer.batch_decode(
            result,
            skip_special_tokens=True,
            # clean_up_tokenization_spaces=True,
        )

        return response


    def conversation(self):
        history = []
        instruction, response = "", ""

        print("당신의 고민에 공감해주는 챗봇입니다. 닉네임을 정해주세요.")
        print("대화를 종료하고 싶으시면 'exit', 새로운 대화를 원하시면 'flush'를 입력하세요.")

        nickname = input("당신의 닉네임: ")
        while True:
            instruction = input(f"{nickname}: ")
            if instruction == "exit":
                break
            if instruction == "flush":
                history = []

            history.append("질문: " + instruction)
            response = self.get_reponse(" ".join(history))
            history.append("답변: " + response)
        
        


if __name__ == "__main__":
    chatbot = EmpatheticChatbot(
        Arguments.define_inference_args()
    )

    chatbot.conversation()

