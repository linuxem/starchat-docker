# Run the model 

# first run the docker 
docker run --gpus all -it --rm -v$PWD:/home/eli/startchat starchat:1.8.6 bash

cd /home/eli/chat 

# Generate samples

To generate a few coding examples from your model, run:

```shell
python generate.py --model_id path/to/your/model
```
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm -p5000:5000 -v$PWD:/home/eli/startchat starchat:1.9.2.6 bash
