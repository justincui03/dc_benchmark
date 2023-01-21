# New augmentation
In DC-Bench, the following commonly used augmentations are provided
- Autoaug
- Randaug
- Imagenet_aug
- DSA

All augmentations are applied on the fly, in order to add a new augmentation method
## Step 1
In evaluator_utils.py inside evaluator folder, find method: <em>custom_aug</em>

## Step 2
Add new data augmentation methods in the form of Pytorch [Transform](https://pytorch.org/vision/0.9/transforms.html)

One example is
```
data_transforms = transforms.Compose([transforms.RandAugment(num_ops=1)])
```

## Step 3
Run the eval commands with new aug
```
python evaluator/evaluator.py --method DC  --dataset tinyimagenet --aug NEW_AUG
```
