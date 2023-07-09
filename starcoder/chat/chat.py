# coding=utf-8
# Copyright 2023 The BigCode and HuggingFace teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import torch
from flask import Flask, render_template, request
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    set_seed,
)
from dialogues import DialogueTemplate, get_dialogue_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        model_id = request.form['model_id']
        revision = request.form['revision']
        system_prompt = request.form['system_prompt']
        outputs = generate_samples(model_id, revision, system_prompt)
        return render_template('index.html', outputs=outputs)
    return render_template('index.html', outputs='')

def generate_samples(model_id, revision, system_prompt):
    # Set seed for reproducibility
    set_seed(42)

    prompts = [
        [
            {
                "role": "user",
                "content": "Develop a C++ program that reads a text file line by line and counts the number of occurrences of a specific word in the file.",
            }
        ],
        # ... rest of the prompts ...
    ]

    try:
        dialogue_template = DialogueTemplate.from_pretrained(model_id, revision=revision)
    except Exception:
        print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
        dialogue_template = get_dialogue_template("no_system")

    if system_prompt is not None:
        dialogue_template.system = system_prompt

    formatted_prompts = []
    for prompt in prompts:
        dialogue_template.messages = [prompt] if isinstance(prompt, dict) else prompt
        formatted_prompts.append(dialogue_template.get_inference_prompt())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    generation_config = GenerationConfig(
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.convert_tokens_to_ids(dialogue_template.end_token),
        min_new_tokens=32,
        max_new_tokens=256,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
    )

    outputs = ""
    for idx, prompt in enumerate(formatted_prompts):
        batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
        generated_ids = model.generate(**batch, generation_config=generation_config)
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False).lstrip()
        outputs += generated_text + "\n\n"

    raw_model_name = model_id.split("/")[-1]
    model_name = f"{raw_model_name}"
    if revision is not None:
        model_name += f"-{revision}"

    with open(f"data/samples-{model_name}.txt", "w", encoding="utf-8") as f:
        f.write(outputs)

    return outputs

if __name__ == "__main__":
    app.run(debug=True)

