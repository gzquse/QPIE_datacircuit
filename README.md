# Build Instructions

1. docker build -f docker/Dockerfile.pennylane -t pytorch-pennylane
2. . ./pm_martin.source

## CUDA-Q version
https://nvidia.github.io/cuda-quantum/latest/using/install/install.html

# Visulize the data

## Moon dataset 

view it interactively 
1. `./pl_sum.py -p a -Y`

## Spiral dataset
1. `./pl_sum.py -p b -Y`

# run with shifter
shifter --image=nersc/pytorch:24.06.01 --module gpu,nccl-plugin --env PYTHONUSERBASE=$SCRATCH/cudaq

# dry run qpie
./qpie_model_inference.py

### GPU requriements NVIDIA A100 80GBs

```
git clone https://github.com/PennyLaneAI/pennylane-lightning.git
cd pennylane-lightning
docker build -f docker/Dockerfile --target ${TARGET} .
```

the other real world datasets stored in data directory


