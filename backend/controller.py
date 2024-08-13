import os, json
from arguments import Arguments

import torch
import faiss
import logging
import pandas as pd
from backend.retriever import FaissRetriever
from backend.docstore import DocumentStore
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate

# from langchain_openai import ChatOpenAI
import dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppController:

    def __init__(self, user_id: str):
        dotenv.load_dotenv()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.user_id = user_id
        self.config = Arguments.app_args()
        self.dpo_data = pd.DataFrame(columns=["context", "chosen", "rejected"])

        if os.path.exists(self.config.model_path):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to("cuda:0", non_blocking=True)

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        else:
            raise FileExistsError(f"Can not find a model that you specified: {self.config.model_path}")
        
        # self.model = ChatOpenAI()
        
        self.docstore = DocumentStore(user_id)

        index = faiss.IndexFlatL2(1024)
        index = faiss.IndexIDMap(index)
        self.retriever = FaissRetriever(
            embedding_model=BGEM3FlagModel('BAAI/bge-m3', use_fp16=True),
            index=index, # Default: 1024 dim
            max_length=512,
            normalize_L2=True,
        )

        template_fn = os.path.join(os.getcwd(), self.config.template_path)
        try:
            with open(template_fn, 'r', encoding='utf-8') as fp:
                template = json.load(fp).get('prompt')
        except FileExistsError as e:
            print(f"File not exist: {str(e)}")

        self.prompt_template = PromptTemplate(
            input_variables=['history', 'instruction'], 
            template=template,
            template_format='f-string',
        )
    

    def get_response(self, user_input: str, **kwargs) -> str:
        dpo_mode = kwargs.get('dpo_mode', False)

        if dpo_mode:
            _, indices = self.retriever.search_similar_without_time(
                user_input, **kwargs
            )
        else:
            _, indices = self.retriever.search_similar(
                user_input, **kwargs
            )

        history = ""
        if self.docstore.history:
            indices = indices[indices != -1]
            top_k_history = [self.docstore.history[i]['text'] for i in indices]
            history = '\n'.join(top_k_history)
        
        prompt = self.prompt_template.format(
            history=history,
            instruction=user_input,
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            return_token_type_ids=False,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            do_sample=kwargs['do_sample'],
            temperature=kwargs.get('temperature', None),
            top_k=kwargs.get('top_k', None),
            top_p=kwargs.get('top_p', None),
            repetition_penalty=kwargs['repetition_penalty'],
            min_new_tokens=kwargs['min_new_tokens'],
            max_new_tokens=kwargs['max_new_tokens'],
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=2 if dpo_mode else 1,
        )

        generated_prompt = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        print(generated_prompt)
        # response = generated_prompt[0].rsplit('### 응답:', 1)[-1].strip()

        # # response = self.model.invoke(prompt).content

        # return response, history
    
    def delete_chat(self, _id: str):
        target_history_index = self.docstore.delete(_id)
        self.retriever.remove_id(target_history_index)

    def clear_all(self):
        self.docstore = DocumentStore(self.user_id)

        index = faiss.IndexFlatL2(1024)
        index = faiss.IndexIDMap(index)
        self.retriever = FaissRetriever(
            embedding_model=BGEM3FlagModel('BAAI/bge-m3', use_fp16=True),
            index=index, # Default: 1024 dim
            max_length=512,
            normalize_L2=True,
        )

        logger.info("Cleared all messages and history")
    

    def write_dpo_data(self, **kwargs):
        new_row = {
            "context": kwargs["context"],
            "chosen": kwargs["chosen"],
            "rejected": kwargs["rejected"],
        }

        self.dpo_data.loc[len(self.dpo_data)] = new_row
    

    def save_dpo_data(self):
        save_dir = self.config.save_dir
        save_fn = os.path.join(save_dir, "dpo_data.tsv")

        if os.path.exists(save_fn):
            original_data = pd.read_csv(save_fn, sep='\t', encoding='utf-8')
            pd.concat([original_data, self.dpo_data]).to_csv(save_fn, encoding='utf-8')
        else:
            self.dpo_data.to_csv(save_fn, sep='\t', encoding='utf-8')
    