procodex
==============================

Onboarding Python, ML and OOP Refresher



<p align="center">
<img src="https://images.squarespace-cdn.com/content/v1/5b360fd2b27e3991639fd27a/1566182675079-K746TPTB53AQFLE4SKZC/Word+Cloud+banner.jpg"  />
</p>


Project Organization
------------
```
├── api
│   ├── app.py
│   ├── config
│   │   └── config.py
│   ├── resources
│   ├── static
│   └── templates
│  
├── checkpoints
│  
├── data
│   ├── processed
│   └── raw
│  
├── docs
│   ├── Analysis.md
│   └── Requirements.md
│  
├── notebooks
│ 
├── procodex
│   ├── config
│   │   └── config.py
│   ├── data
│   │   └── make_dataset.py
│   ├── dispatcher
│   ├── features
│   │   ├── build_features.py
│   ├── models
│   │   ├── test_model.py
│   │   └── train_model.py
│   ├── utils
│   ├── visualisation
│   |   └── visualisation.py
│   └── main.py
│ 
├── Dockerfile
│ 
├── run.sh
├── logs
├── references
├── requirements.txt
├── README.md
├── LICENSE
└── tests
    └── test_environment.py
```
--------


## Getting Started

### Requirements

```
pip install -r requirements.txt
```

### Download the dataset

The following command will download the dataset from the URL given in `src/config/config.py` file .

```
python -m procodex.data.make_dataset
```

### Run

```
python -m procodex.main
```
OR

```
./run.sh
```

### Test

```
python -m tests.test_environment
```


### To-do List

- [ ] Download dataset
- [ ] Pre-process data
- [ ] Train model
- [ ] Test model
- [ ] Main Pipeline

-------------------------------
