import os, json
import logging

import torch
import faiss

from arguments import Arguments
from backend.retriever import FaissRetriever
from backend.docstore import DocumentStore
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.prompts import PromptTemplate


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AppController:

    def __init__(self, user_id: str):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.user_id = user_id
        self.config = Arguments.app_args()

        # Load finetuned model and tokenizer
        if os.path.exists(self.config.model_path):
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            ).to("cuda:0", non_blocking=True)

            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path)
        else:
            raise FileExistsError(f"Can not find a model that you specified: {self.config.model_path}")
        
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
            logger.error(f"File not exist: {str(e)}")

        self.prompt_template = PromptTemplate(
            input_variables=['history', 'instruction'], 
            template=template,
            template_format='f-string',
        )
    

    def get_response(self, user_input: str, **kwargs) -> str:
        """
        Generate assistant response(s) about user input.
        By default, the assistant's response is a single string. 
        However, when DPO mode is activated, the response is a tuple consisting of two responses.
        """
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

        if len(generated_prompt) > 1:
            response = (
                generated_prompt[0].rsplit('### 응답:', 1)[-1].strip(),
                generated_prompt[1].rsplit('### 응답:', 1)[-1].strip(),
            )
        else:
            response = generated_prompt[0].rsplit('### 응답:', 1)[-1].strip()

        return response, history
    

    def delete_chat(self, _id: str) -> None:
        """
        Delete pair of user input and assistant response.
        If user clicked delete button, this method receives message's id(metadatas: '_id').
        """
        target_history_index = self.docstore.delete(_id)
        self.retriever.remove_id(target_history_index)


    def clear_all(self) -> None:
        """
        Clear all stored messages and histories.
        """
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
    
    
    def get_context(self, current_history: str, current_user_input: str) -> str:
        """
        Format the context for saving the DPO dataset.
        """
        replaced = current_history.replace("### 명령어", "질문")
        replaced = replaced.replace("### 응답", "대답")
        replaced = replaced.replace('\n', ' ')
        new_format = replaced + ' 질문: ' + current_user_input

        return new_format.strip()


    def write_dpo_data(self, **kwargs) -> None:
        """
        When DPO mode is activated, 
        write the context and responses to the Pandas dataframe.
        The responses(chosen, rejected) are reflecting user's preference.
        """
        try:
            new_row = {
                "context": kwargs["context"],
                "chosen": kwargs["chosen"],
                "rejected": kwargs["rejected"],
            }
            self.docstore.dpo_data.loc[len(self.docstore.dpo_data)] = new_row
            logger.info(f"Added data: {new_row}")
        except Exception as e:
            logger.error(f"Failed to add data: {e}")
