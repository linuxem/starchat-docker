FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
#FROM nvcr.io/nvidia/pytorch:22.04-py3 


# home directory 
RUN mkdir /home/eli
COPY starcoder /home/eli/

# Copy files to home

COPY *.* /home/eli

# Add requierment.txt file
ADD requirement.txt .

# Add python file and directory
ADD main.py .

# upgrade pip and install pip packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirement.txt \
    pip install --upgrade transformers
    # Note: we had to merge the two "pip install" package lists here, otherwise
    # the last "pip install" command in the OP may break dependency resolution...

# run python program
CMD ["python3", "/home/eli/startchat/starcoder/chat/chat.py"]
