from transformers import AutoModel, AutoTokenizer

# Do this on a machine with internet access
model = AutoModel.from_pretrained("HuggingFaceH4/starchat-alpha")
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/starchat-alpha")

_ = model.save_pretrained("/home/eli/WorkSpace/HuggingFaceH4/starchat-alpha/models")
_ = tokenizer.save_pretrained("/home/eli/WorkSpace/HuggingFaceH4/starchat-alpha/models")


#model = AutoModel.from_pretrained("/home/eli/WorkSpace/starcoder/starcoder-gpteacher-code-instruct/starcoder/models")
#tokenizer = AutoTokenizer.from_pretrained("/home/eli/WorkSpace/starcoder/starcoder-gpteacher-code-instruct/models")
