import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

DEEPSEEK_VERSION = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_VERSION)
model = AutoModelForCausalLM.from_pretrained(
    DEEPSEEK_VERSION,
    torch_dtype=torch.float16,
    device_map="auto"
)

input_text = """
I have a material and its band gap value. A band gap is the distance \
between the valence band of electrons and the conduction band, \
representing the minimum energy that is required to excite an electron to the conduction band.

(Sr2Ti2O6, 1.85)

Please propose a modification to the material that results in a band gap of 1.4 eV. \
You can choose one of the four following modifications:
1. exchange: exchange two elements in the material
2. substitute: substitute one element in the material with another
3. remove: remove an element from the material
4. add: add an element to the material

IT IS CRUCIAL THAT YOU ADHERE TO THIS OUTPUT: {Hypothesis: $HYPOTHESIS, Modification: [$TYPE, $ELEMENT_1, $ELEMENT_2]}. \
Here are the requirements:
1. $HYPOTHESIS should be your analysis and reason for choosing a modification
2. $TYPE should be the modification type; one of "exchange", "substitute", "remove", "add"
3. $ELEMENT should be the selected element type to be modified. For "exchange" and "substitute", \
    two $ELEMENT placeholders are needed. For "remove" and "add", one $ELEMENT placeholder is needed.\n

For example: 

{
  'Hypothesis': 'Substituting Ti with Ru in Sr₂Ti₂O₆ introduces a different electronic structure, altering the band gap to 1.4 eV.',
  'Modification': ['substitute', 'Ti', 'Ru']
}
"""

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_length=8192, pad_token_id=tokenizer.eos_token_id)

response = tokenizer.decode(output[0], skip_special_tokens=True)

after_think = response.split("</think>", 1)[-1]

# Pattern to match "Key: value" where value may span multiple lines
# pattern = r"(?P<key>Hypothesis|Modification):\s*(?P<value>.*?)(?=(\n[A-Z][a-z]+:|$))"

# matches = re.finditer(pattern, after_think, re.DOTALL)

# result = {m.group("key"): m.group("value").strip() for m in matches}

print(after_think)
