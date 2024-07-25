import os
import sys
import torch
from argparse import ArgumentParser

from utils.arguments import Arguments
from utils.prompter import Prompter
from utils.streamer import CustomStreamer
from transformers import AutoTokenizer, AutoModelForCausalLM

import streamlit as st
from streamlit_chat import message
from configparser import ConfigParser



class EmpatheticChatbot:

    def __init__(
            self,
            config: ArgumentParser,
    ):
        self.MESSAGE_PREFIX = "질문: "
        self.RESPONSE_PREFIX = "대답: "

        self.config = config
        self.model = None
        self.tokenizer = None
        self.prompter = None
        self.streamer = None
        self.device = None
        
        if os.path.exists(config.merge_dir):
            self.model = AutoModelForCausalLM.from_pretrained(
                config.merge_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
            ).to("cuda", non_blocking=True)
            self.tokenizer = AutoTokenizer.from_pretrained(
                config.merge_dir,
                add_special_tokens=True,
            )
            self.prompter = Prompter(template_name="multi")
            self.streamer = CustomStreamer(self.tokenizer, skip_prompt=True)
        else:
            raise FileExistsError
        
        self.device = self.model.device

        self.model.eval()


    def get_response(self, message: str) -> str:
        prompt = self.prompter.generate_prompt(message)

        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            return_token_type_ids=False,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            streamer=self.streamer,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature,
            top_k=self.config.top_k if self.config.do_sample else None,
            top_p=self.config.top_p if self.config.do_sample else None,
            repetition_penalty=self.config.repetition_penalty,
            max_new_tokens=self.config.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        generated_prompt = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        response = self.prompter.get_response(generated_prompt[0])

        return response


    def chat_validation(self):
        history = []

        print("Enter 'exit' to end conversation or 'clear' to flush conversation.")
        print("Please enter a nickname to use in conversation.")
        nickname = input("Nickname: ")
        while True:
            message = input(f"{nickname}: ").strip()
            if message == "exit":
                yorn = input("Are you really want to end the conversation? (Y/N)")
                if yorn.upper() == 'Y':
                    break
                elif yorn.upper() == 'N':
                    continue

            if message == "clear":
                if sys.platform == "linux":
                    os.system("clear")
                else:
                    os.system("cls")
                self.chat_validation()
            
            history.append(self.MESSAGE_PREFIX + message)
            response = self.get_response('\n'.join(history))
            history.append(self.RESPONSE_PREFIX + response) 



def chat_with_streamlit(chatbot):
    st.title("Empathetic-Chatbot Demo")
    
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    for content in st.session_state.history:
        print(content)
        with st.chat_message(content['role']):
            st.markdown(content['message'])
    
    message = st.chat_input("메시지를 입력하세요.")
    # if message:
    #     with st.chat_message('user', avatar='😔'):
    #         st.markdown(message)
    #         st.session_state.history.append({'role': 'user', 'message': message})
    #     with st.chat_message('bot', avatar='🤖'):
    #         response = f"{message}"
    #         st.markdown(response)
    #         st.session_state.history.append({'role': 'bot', 'message': response})





if __name__ == "__main__":
    # chatbot = EmpatheticChatbot(
    #     Arguments.define_args()
    # )

    # chatbot.chat_validation()

    chat_with_streamlit()

