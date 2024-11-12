# Build Instructions

1. docker build -f docker/Dockerfile.pennylane -t pytorch-pennylane
2. . ./pm_martin.source


# Visulize the data

## Moon dataset 

view it interactively 
1. `./pl_sum.py -p a -Y`

## Spiral dataset
1. `./pl_sum.py -p b -Y`


### GPU requriements NVIDIA A100 80GBs

```
git clone https://github.com/PennyLaneAI/pennylane-lightning.git
cd pennylane-lightning
docker build -f docker/Dockerfile --target ${TARGET} .
```


the other real world datasets stored in data directory


