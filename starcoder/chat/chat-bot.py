import flask
from flask import Flask
import requests
from flask import render_template
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import request

webPage = 'chat.html'
history_file = "/home/eli/WorkSpace/web-appchat_history.txt/chat_history.txt"
port = 3301

app = flask.Flask(__name__)
app.config["DEBUG"] = True #False #True

@app.route('/', methods=['GET','POST'])
def home():
    # checkpoint = "/home/guy/Downloads/code/starchat-docker/data"
    # device = "cpu" # for GPU usage or "cpu" for CPU usage

    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # # to save memory consider using fp16 or bf16 by specifying torch_dtype=torch.float16 for example
    # model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    # inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
    # outputs = model.generate(inputs)
    # # clean_up_tokenization_spaces=False prevents a tokenizer edge case which can result in spaces being removed around punctuation
    # # result = (tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False))

    if 'prompt' in request.args:
        with open(history_file, "a") as hf:
            chat_history = str(hf.write(str(request.args['prompt'])))
            chat_history = hf.write(str("\n"))

    with open(history_file, "r") as hf:
        chat_history = hf.read()
    print(chat_history)
    
    return render_template(webPage, history = str(chat_history))

app.run(host='0.0.0.0', port=port)
