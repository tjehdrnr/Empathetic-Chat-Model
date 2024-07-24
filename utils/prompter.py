import os, re
import json
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            template_name = "multi"
        file_name = os.path.join("templates", f"{template_name}.json")
        if not os.path.exists(file_name):
            raise ValueError(f"Can't find {file_name}")
        try:
            with open(file_name) as fp:
                self.template = json.load(fp)
        except UnicodeDecodeError:
            with open(file_name, encoding='utf-8') as fp:
                self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
    
    def generate_prompt(
        self,
        instruction: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from optional input
        # if a label (=response, =output) is provided, it's also appended.

        # process for multi-turn conversation.
        def converter(sentence):
            result = re.sub(r"질문\s*", "### 명령어", sentence)
            result = re.sub(r"답변\s*", "### 응답", result)
            
            return result

        instruction = converter(instruction)
        new_instruction = instruction.split('\n')[-1]
        history = instruction[:-len(new_instruction)]
        
        try:
            new_instruction = new_instruction.split("### 명령어: ")[1]
        except:
            new_instruction = new_instruction.split("### 명령어: ")[0]

        res = self.template["prompt"].format(history=history, instruction=new_instruction)

        if label is not None:
            res = f"{res}{label}"
        if self._verbose:
            print('*' * 100)
            print(res)
            print('*' * 100)

        return res


    def get_response(self, output: str) -> str:
        return output.rsplit(self.template["response_split"], 1)[-1].strip()

        