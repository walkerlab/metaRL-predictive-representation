# metaRL-predictive-representation
Code for the paper: "Predictive Coding Enhances Meta-RL To Achieve Interpretable Bayes-Optimal Belief Representation Under Partial Observability" Po-Chen Kuo, Han Hou, Will Dabney, Edgar Y. Walker. Published in [NeurIPS 2025](https://openreview.net/forum?id=ykDUVoelgj). 
(arXiv: [https://arxiv.org/abs/2510.22039](https://arxiv.org/abs/2510.22039))

```
@inproceedings{
kuo2025predictive,
title={Predictive Coding Enhances Meta-{RL} To Achieve Interpretable Bayes-Optimal Belief Representation Under Partial Observability},
author={Po-Chen Kuo and Han Hou and Will Dabney and Edgar Y. Walker},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=ykDUVoelgj}
}
```

### Overview
The entry point for model training is in `train.py`, where args and configs are parsed to initialize and train the models.
The meta-RL models, including both black-box meta-RL and meta-RL with predictive modeuls, are defined in `metalearner.py`.
State machine simulation analysis, including comments on how to use the script, is demonstrated in `sms.py`.
The main requirements are listed in `requirements.txt`.

### Model training

To train meta-RL with predictive modules on a task (e.g. OracleBandit), use:
```
python train.py --exp_label=mpc --env_name=OracleBanditDeterministic-v0
```

To train black-box meta-RL (e.g. RL<sup>2</sup>) on a task (e.g. OracleBandit), use:
```
python train.py --exp_label=rl2 --env_name=OracleBanditDeterministic-v0
```

You can find other tasks defined in the `environments/` folder.
The trained models and checkpoints will be saved at `./logs` by default.
The default configs are in the `config/` folder. 
Hyperparameters can be overwritten command line arguments.

### State machine simulation
To evalute representational equivalence on trained models, modify the `sms.py` script to add the paths to the trained models and adjust parameters to the tasks you're evaluating on accordingly.
