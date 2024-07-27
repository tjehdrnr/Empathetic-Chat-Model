import json, os
import torch
import warnings

from argparse import ArgumentParser
from arguments import Arguments

from langchain.prompts import PromptTemplate
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM



class CustomParser(StrOutputParser):
    def __init__(self):
        super().__init__()
    
    def parse(self, text: str) -> str:
        # Returns the last response only.
        text = text.rsplit("### 응답:", 1)[-1].strip()
        return text


class EmphatheticChatbot:

    def __init__(self, config: ArgumentParser, template=str):
        torch.cuda.empty_cache()

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).to(f"cuda:{config.gpu_id}", non_blocking=True)

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=config.gpu_id,
            min_new_tokens=config.min_new_tokens,
            max_new_tokens=config.max_new_tokens,
            do_sample=config.do_sample,
            early_stopping=config.early_stopping,
            temperature=config.temperature if config.do_sample else None,
            top_k=config.tok_k if config.do_sample else None,
            top_p=config.tok_p if config.do_sample else None,
            repetition_penalty=config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        self.llm = HuggingFacePipeline(pipeline=pipe)

        self.prompt = PromptTemplate.from_template(template)

        self.model.use_cache = True
        self.model.eval()
    

    def get_response(self, message: str):
        # input_dict = {"instruction": message}

        conversation = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=ConversationBufferMemory(
                memory_key="history",
                human_prefix="### 명령어",
                ai_prefix="### 응답",
            ),
            input_key="instruction",
            output_parser=CustomParser(),
            verbose=False,
        )

        response = conversation.predict(
            instruction="어제는 수학 공부를 했어요. 수학책 129페이지까지 했어요."
        )
        print(response)


        # print(conversation.memory.load_memory_variables({})['history'])

        response = conversation.predict(
            instruction="요즘 공부가 너무 재미있어요. 오늘은 과학 공부를 하려고요."
        )
        print(response)

        response = conversation.predict(
            instruction="과학 공부하기 전에 어제 했던 공부 복습 좀 해야겠어요. 제가 어제 수학책을 어디까지 했었죠?"
        )
        print(response)




if __name__ == "__main__":
    config = Arguments.demo_args()
    if config.gpu_id < 0:
        warnings.warn("Start inference with CPU")

    template_path = os.path.join(os.getcwd(), config.template_path)

    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f).get('prompt')
        
    chatbot = EmphatheticChatbot(config, template)

    message = "저 요즘 고민이 너무 많아요.."
    chatbot.get_response(message)



