# Run the model 

# first run the docker 
docker run --gpus all -it --rm -v$PWD:/home/eli/startchat starchat:1.8.6 bash

cd /home/eli/chat 

# Generate samples

To generate a few coding examples from your model, run:

```shell
python generate.py --model_id path/to/your/model
```
