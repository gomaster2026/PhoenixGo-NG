# Distributed Training Framework

PhoenixGo-NG supports distributed training through community computing power contributions.

## Architecture

```
Contributor Client (node_client.py)
    | Generates self-play data
    v
Training Server (kaggle_train.py)
    | Aggregates chunks, runs RL
    v
Weight Publication (weight_release.py)
    | Publishes next-gen weights
    v
Contributors (download new weights)
```

## Quick Start

### As a Contributor

```bash
pip install requests
python distributed/node_client.py
```

### As a Trainer

```bash
python distributed/kaggle_train.py --data train_data.gz --epochs 10
```

### Generate Initial Weights

```bash
python distributed/make_dummy_weights.py
```