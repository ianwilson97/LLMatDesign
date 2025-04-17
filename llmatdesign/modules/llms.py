import os
import re

class AskLLM:
    def __init__(
        self,
        tokenizer,
        model, 
        api_key=None,
        openai_organization=None, 
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model

    def ask(self, prompt):
        # print(prompt)
        print("ASKING DEEPSEEK FOR VALUES")
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_length=8192, pad_token_id=self.tokenizer.eos_token_id)

        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        after_think = response.split("</think>", 1)[-1]
        print(after_think)
        return after_think
