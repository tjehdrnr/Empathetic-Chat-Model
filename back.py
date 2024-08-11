import json, os
import torch
import warnings

from argparse import ArgumentParser
from arguments import Arguments
import dotenv

import backend.docstore as docstore
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.memory import VectorStoreRetrieverMemory
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

from langchain_openai import OpenAIEmbeddings



class CustomParser(StrOutputParser):
    def __init__(self):
        super().__init__()
    
    def parse(self, text: str) -> str:
        # Returns the last response only.
        text = text.rsplit("### 응답:", 1)[-1].strip()
        return text


class EmphatheticChatbot:

    def __init__(self, config: ArgumentParser, template: str):
        dotenv.load_dotenv()
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

        self.vectorstore = FAISS(
            embedding_function=OpenAIEmbeddings(),
            index=docstore.IndexFlatL2(1536),
            docstore=InMemoryDocstore({}),
            normalize_L2=False,
        )


        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={'k': 2},
        )

        self.memory = VectorStoreRetrieverMemory(
            retriever=self.retriever,
            memory_key="history",
            input_key="instruction",
            return_docs=False,
        )

        self.prompt = PromptTemplate(
            input_variables=['history', 'instruction'], template=template
        )

        self.conversation_chain = ConversationChain(
            llm=self.llm,
            prompt=self.prompt,
            memory=self.memory,
            # input_key="instruction",
            output_parser=CustomParser(),
            verbose=True,
        )
        
        self.model.use_cache = True
        self.model.eval()
    

    def get_response(self, message: str):
        # input_dict = {"instruction": message}

        response = self.conversation_chain.predict(
            instruction="어제는 수학 공부를 했어요. 수학책 129페이지까지 했어요."
        )
        print(self.conversation_chain.memory.load_memory_variables)




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



