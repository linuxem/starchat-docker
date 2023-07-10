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
from dialogues import DialogueTemplate, get_dialogue_template
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          GenerationConfig, set_seed)

from flask import Flask, render_template, request
port = 3303

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_input = request.form['prompt']
        generated_text = generate_text(user_input)
        return render_template('index.html', prompt=user_input, output=generated_text)
    return render_template('index.html')

def generate_text(prompt):
    set_seed(42)

    try:
        dialogue_template = DialogueTemplate.from_pretrained(args.model_id, revision=args.revision)
    except Exception:
        print("No dialogue template found in model repo. Defaulting to the `no_system` template.")
        dialogue_template = get_dialogue_template("no_system")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
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
        args.model_id, revision=args.revision, load_in_8bit=True, device_map="auto", torch_dtype=torch.float16
    )

    batch = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    generated_ids = model.generate(**batch, generation_config=generation_config)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False).lstrip()
    
    return generated_text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        help="Name of model to generate samples with",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="The model repo's revision to use",
    )
    args = parser.parse_args()

    app.run(debug=True, host='0.0.0.0', port=port)
