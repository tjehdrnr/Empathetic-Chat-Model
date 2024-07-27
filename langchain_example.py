import json, os
import torch
import warnings
from argparse import ArgumentParser

from langchain.prompts import PromptTemplate
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from arguments import Arguments



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

        self.prompt = PromptTemplate(
            input_variables=["history", "instruction"], template=template
        )

        self.model.use_cache = True
        self.model.eval()
    

    def get_response(self, message: str):

        conversation = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=ConversationBufferMemory(
                memory_key="history",
                human_prefix="### 명령어",
                ai_prefix="### 응답",
            ),
            input_key="instruction",
            output_key="### 응답",
            verbose=False,
        )

        response = conversation.predict(
            instruction=message
        )
        print(response)
        print(conversation.memory.load_memory_variables({}))





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



