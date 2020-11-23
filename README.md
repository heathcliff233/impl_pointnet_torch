# Implementation of PointNet in PyTorch
## Project structure
```text
impl_pointnet/
├── README.md
├── data
├── model
│   ├── __init__.py
│   ├── pointnet.py
│   ├── pointnet_cls.py
│   ├── pointnet_sem.py
│   └── summary.py
├── train_cls.py
├── train_sem.py
└── utils
    ├── __init__.py
    └── loader.py
```

## Train 
### Classification
#### Dataset preparation
options
* set `--download` to `True`
* download and unzip and specify in `--dataset_path`

#### Run
```python
python3 train_cls.py [--dataset_path] [--download] [--batch_size] [--trained_model] [--epoch] [--learning_rate] [--decay_rate] [--regularize]
```

### Part Segmentation
#### Run
```python
python3 train_part.py [--dataset_path] [--download] [--batch_size] [--trained_model] [--epoch] [--learning_rate] [--decay_rate] [--regularize]
```

## TODO
* add support for semantic segmentation
* add test script
* add visualizer  