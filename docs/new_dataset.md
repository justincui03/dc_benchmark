# Add a new dataset

Adding a new dataset is easy with DC-Bench.  
- For dataset already included in PyTorch, they can directly be used **out of the box**. 
- For external datasets, they can be integrated with **a few lines** of code .

We will use TinyImagenet dataset used in our benchmark to demonstrate how to integrate external datasets.

## Step 1: Dataset Preparation
We use the standard [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) interface provided in PyTorch to hold all datasets. 

For example our TinyImagenet uses [ImageFolder](https://pytorch.org/vision/stable/datasets.html#base-classes-for-custom-datasets) to convert into a PyTorch Dataset.

## Step 2: Placement of new Dataset
After the new dataset is processed into the right format that can be load into a PyTorch Dataset, just put it anywhere that can be accessed.

## Step 3: Load it in DC-Bench
Inside evaluator_utils.py under evaluator folder, add the following lines to load the data
```
dst_train = datasets.ImageFolder(os.path.join('/nfs/data/justincui/data/tiny-imagenet-200', "train"), transform=transform)
dst_test = datasets.ImageFolder(os.path.join('/nfs/data/justincui/data/tiny-imagenet-200', "val", "images"), transform=transform)
```

## Step 4: Ready to go
Run the eval commands with your new dataset
```
python evaluator/evaluator.py --method DC  --dataset tinyimagenet
```
