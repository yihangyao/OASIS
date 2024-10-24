# <img src="figure/oasis.png" width="40"> OASIS: Conditional Distribution Shaping for Offline Safe Reinforcement Learning 
Conference on Neural Information Processing Systems (NeurIPS), 2024

[**[Project Page]**](https://zhengyinan-air.github.io/FISOR/) [**[Arxiv]**](https://arxiv.org/pdf/2401.10700.pdf) [**[Openreview]**](https://openreview.net/forum?id=j5JvZCaDM0)

[Yihang Yao*]( ), [Zhepeng Cen*]( ), [Wenhao Ding]( ), [Haohong Lin]( ), [Shiqi Liu]( ), [Tingnan Zhang]( ), [Wenhao Yu]( ), [Ding Zhao]( )

The official implementation of OASIS, a **Data-centric** approach for offline safe RL.

# Methods

<p float="left">
<img src="figure/oasis-overview.png" width="800">
</p>


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

## Bibtex

If you find our code and paper can help, please cite our paper as:
```
@article{
    yao2024oasis,
    title={OASIS: Conditional Distribution Shaping for Offline Safe Reinforcement Learning},
    author={Yao, Yihang and Cen, Zhepeng and Ding, Wenhao and Lin, Haohong and Liu, Shiqi and Zhang, Tingnan and Yu, Wenhao and Zhao, Ding},
    journal={arXiv preprint arXiv:2407.14653},
    year={2024}
}
```