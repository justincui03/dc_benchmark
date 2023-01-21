# Nueral Architecture Search

## Commands to start a NAS evaluation
```
cd darts-pt
bash darts-201.sh --dc_method tm
```

## How to perform NAS using a new condensed dataset
Testing a new condensed dataset can be done in the following steps:

- Locate **nasbench201** directory under darts-pt
- Open file **train_search.py** python file
- Add your data loading logic starting at line: **230**
- **run** the script to start using the new condensed dataset

## Note
By the default, we use CIFAR-10 to test the ranking correlation. We will include the support for more datasets(CIFAR-100 and TinyImageNet) in future releases. 
