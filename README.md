# OASIS Implementation (NeurIPS 2024 submission)

### Installation
This code is tested on an Ubuntu 18.04 system.
To install the packages, please first create a python environment with python==3.8, then run:

```
cd OSRL
pip install -e .
cd ../DSRL
pip install -e .
cd ..
pip install -r requirements.txt
```

### OASIS Training
To train an OASIS agent, run:
```
cd OSRL/examples/train
python train_oasis.py
```
It will train an OASIS model for the Ball-Circle task using tempting dataset.

### Dataset Generation
To generate a dataset based on an OASIS model, run:
```
cd Generation
python dataset_generation.py
```
It will use the pre-trained OASIS model "BallCircle.pt" in the "models" folder, and use pre-trained cost/reward models "BC_cost.pt" and "BC_reward.pt" to label the dataset. The generated dataset is saved to the "dataset" folder. The target cost limit is 20.
 
### BCQ-Lag Training
To Train an offline safe RL agent based on the OASIS generated dataset, run:
```
cd OSRL/examples/train
python train_bcql.py
```
It will use the dataset saved in the "dataset" folder to train an BCQ-Lag agent. The cost limit is 20.

### Github Reference
- Tianshou: https://github.com/thu-ml/tianshou
- Decision Diffuser: https://github.com/anuragajay/decision-diffuser
- AdaptDiffuser: https://github.com/Liang-ZX/AdaptDiffuser
- OSRL: https://github.com/liuzuxin/osrl
- DSRL: https://github.com/liuzuxin/dsrl