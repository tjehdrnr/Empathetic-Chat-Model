import json, os
import torch

from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    cur_dir = os.getcwd()
    template_path = os.path.join(cur_dir, "finetune\\templates\\multi.json")
    model_dir = os.path.join(cur_dir, "finetune\\merged_model")

    with open(template_path, 'r', encoding='utf-8') as f:
        template = json.load(f).get('prompt')

    prompt_template = PromptTemplate(
        input_variables=["history", "instruction"], template=template
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    ).to("cuda", non_blocking=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    pipe = pipeline(
        "text-generation",
        model,
        tokenizer=tokenizer,
        device=0,
        min_new_tokens=10,
        max_new_tokens=128,
        early_stopping=True,
        do_sample=True,
        temperature=0.3,
        top_k=30,
        top_p=0.90,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.pad_token_id,
    )

    chat_pipeline = HuggingFacePipeline(pipeline=pipe)

    message = "요즘 너무 힘든 일이 있어요..ㅜㅜ"
    message = prompt_template.format(
        history=None,
        instruction=message
    )

    response = chat_pipeline.invoke(message)
    print(response)



# ConversationChain()
# prompt = prompt_template.format(country="대한민국")

# print(prompt)