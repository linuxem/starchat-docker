import torch
##from transformers import pipeline

##pipe = pipeline("text-generation", model="/home/eli/WorkSpace/starchat-alpha/models", torch_dtype=torch.bfloat16, device_map="auto")
#pipe = pipeline("text-generation", model="HuggingFaceH4/starchat-alpha", torch_dtype=torch.bfloat16, device_map="auto")

##prompt_template = "<|system|>\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
##prompt = prompt_template.format(query="How do I sort a list in Python?")
# We use a special <|end|> token with ID 49155 to denote ends of a turn
##outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95, eos_token_id=49155)
# You can sort a list in Python by using the sort() method. Here's an example:\n\n```\nnumbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\nnumbers.sort()\nprint(numbers)\n```\n\nThis will sort the list in place and print the sorted list.

from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "/home/eli/WorkSpace/starchat-alpha/models"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# to save memory consider using fp16 or bf16 by specifying torch_dtype=torch.float16 for example
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
# clean_up_tokenization_spaces=False prevents a tokenizer edge case which can result in spaces being removed around punctuation
print(tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False))
