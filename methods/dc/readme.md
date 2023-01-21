# SOTA

We provide all the commands to reproduce the SOTA results. For instructions on how to setup the code, see [instructions](instructions.md)

# DC
```
python main.py  --dataset CIFAR10  --model ConvNet  --ipc 10
# --dataset: MNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100
# --ipc (images/class): 1, 10, 20, 30, 40, 50
```
# DSA
```
python main.py  --dataset CIFAR10  --model ConvNet  --ipc 10  --init real  --method DSA  --dsa_strategy color_crop_cutout_flip_scale_rotate
# --dataset: MNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100
# --ipc (images/class): 1, 10, 20, 30, 40, 50
```
# DM
```
python main_DM.py  --dataset CIFAR10  --model ConvNet  --ipc 10  --dsa_strategy color_crop_cutout_flip_scale_rotate  --init real  --lr_img 1  --num_exp 5  --num_eval 5 
# Empirically, for CIFAR10 dataset we set --lr_img 1 for --ipc = 1/10/50, --lr_img 10 for --ipc = 100/200/500/1000/1250. For CIFAR100 dataset, we set --lr_img 1 for --ipc = 1/10/50/100/125.
```

# OOM Problem
It's common for DC/DSA/DM to run into OOM problems with large datasets or large IPCs, here is the trick to mitigate the problem.  
Find the following code in <em>main.py</em>
```
optimizer_img.zero_grad()
loss.backward()
optimizer_img.step()
```
And move these code into the for loop before it, after modification, it will look something like this 
```
for c in range(num_classes):
   loss = ...
   optimizer_img.zero_grad()
   loss.backward()
   optimizer_img.step()
```
