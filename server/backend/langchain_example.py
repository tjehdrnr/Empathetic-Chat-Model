import json, os
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline

cur_dir = os.getcwd()
template_path = os.path.join(cur_dir, "finetune\\templates\\multi.json")

with open(template_path, 'r', encoding='utf-8') as f:
    template = json.load(f).get('prompt')

prompt_template = PromptTemplate(
    input_variables=["history", "instruction"], template=template
)



ConversationChain()
# prompt = prompt_template.format(country="대한민국")

# print(prompt)