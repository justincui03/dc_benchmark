# SOTA
Here we list all the commands to reproduce the SOTA results for Trajection Matching.

**Note** All the sota parameters are providded by the author.

For instructions on how to setup the code, see [instructions](instructions.md)

# Generate training trajectories
TM requires us to pre-generate training trajectories to use during synthetic image generation phase. Therefore we have to run the following command before starting to generate synthetic images. 

## CIFAR-10
With ZCA.  
```
python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100 --zca
```
Without ZCA
```
python buffer.py --dataset=CIFAR10 --model=ConvNet --train_epochs=50 --num_experts=100
```
## CIFAR-100
With ZCA.  
```
python buffer.py --dataset=CIFAR100 --model=ConvNet --train_epochs=50 --num_experts=100 --zca
```
Without ZCA
```
python buffer.py --dataset=CIFAR100 --model=ConvNet --train_epochs=50 --num_experts=100
```

### TinyImagenet
As TinyImagenet is not included in PyTorch, make sure you have set up the dataset before running the following command
With ZCA.  
```
python buffer.py --dataset=Tiny --model=ConvNet --train_epochs=50 --num_experts=100 --zca
```
Without ZCA
```
python buffer.py --dataset=Tiny --model=ConvNet --train_epochs=50 --num_experts=100
```

# Train synthetic datasets
There a few things to note before running TM:  
- Some TM SOTA results are achieved with ZCA, some are not
- TM reports the maximum accuracy across all epochs, the result from the final epoch could be lower than the paper reported result.
## CIFAR-10
### IPC 1
```
python distill.py --dataset=CIFAR10 --ipc=1 --syn_steps=50 --expert_epochs=2 --max_start_epoch=2 --lr_img=100 --lr_lr=1e-07 --lr_teacher=0.01 --zca
```
### IPC 10
```
python distill.py --dataset=CIFAR10 --ipc=10 --syn_steps=30 --expert_epochs=2 --max_start_epoch=20 --lr_img=100 --lr_lr=1e-05 --lr_teacher=0.001 --zca
```
### IPC 50
```
python distill.py --dataset=CIFAR10 --ipc=50 --syn_steps=30 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.001
```
## CIFAR-100
### IPC 1
```
python distill.py --dataset=CIFAR100 --ipc=1 --syn_steps=20 --expert_epochs=3 --max_start_epoch=20 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --zca
```
### IPC 10
```
python distill.py --dataset=CIFAR100 --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=20 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01
```
### IPC 50
```
python distill.py --dataset=CIFAR100 --ipc=50 --syn_steps=80 --expert_epochs=2 --max_start_epoch=40 --lr_img=1000 --lr_lr=1e-05 --lr_teacher=0.01 --batch_syn=125 --zca
```
### TinyImagenet
### IPC 1
```
python distill.py --dataset=Tiny --ipc=1 --syn_steps=10 --expert_epochs=2 --max_start_epoch=10 --lr_img=10000 --lr_lr=1e-04 --lr_teacher=0.01
```
### IPC 10
```
python distill.py --dataset=Tiny --ipc=10 --syn_steps=20 --expert_epochs=2 --max_start_epoch=40 --lr_img=10000 --lr_lr=1e-04 --lr_teacher=0.01 --batch_syn 200
```
### IPC 50
```
python distill.py --dataset=Tiny --ipc=50 --syn_steps=20 --expert_epochs=2 --max_start_epoch=40 --lr_img=10000 --lr_lr=1e-04 --lr_teacher=0.01 --batch_syn=300 --model=ConvNetD4
```
