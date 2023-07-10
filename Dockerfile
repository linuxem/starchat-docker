FROM linuxem/starchat:1.8.6
#FROM nvcr.io/nvidia/pytorch:22.04-py3 

# home directory 
#RUN mkdir /home/eli/

# upgrade pip and install pip packages
#RUN pip install --no-cache-dir --upgrade pip && \
#    pip install --no-cache-dir -r requirement.txt 
    # Note: we had to merge the two "pip install" package lists here, otherwise
    # the last "pip install" command in the OP may break dependency resolution...
WORKDIR "/home/eli/startchat/"

COPY start-chat.py .

COPY starcoder /home/eli/startchat

# run python program
CMD ["python3", "./starcoder/chat/generate.py", "--model_id", "./models"]
