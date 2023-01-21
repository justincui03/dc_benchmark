# Add new model
DC-Bench includes most of the commonly used models to evaluate the performance of synthetic datasets.

But it's also easy to extend to add new models.

# Step 1: Create the new model
You can create the new model in whatever way you like as long as it's a subclass of the PyTorch [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html)

# Step 2: Integrate
After you are done creating the model, put it under the networks folder. 
Then open network_utils.py in the networks folder and add your model similar to the following code
```
elif model_name == 'resnet18':
  return ResNet18(channel=channel, num_classes=num_classes)
```

# Step 3: Run the eval
Now you have integrated your new model, you can starting using the new model as
```
python evaluator/evaluator.py --method DC  --model NEW_MODEL
```
