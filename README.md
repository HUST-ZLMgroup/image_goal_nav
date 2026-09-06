
Imitation inspired deep reinforcement learning for monocular RGB embodied indoor navigation

 Yi Yan1, Yiheng Su1, Jiaqi Wang1, Limao Zhang1*, Lijun Zhu2, Changyong Liu3, Jing Liu1, Zhuang Xia1, Mirosław J. Skibniewski4, Lieyun Ding1
1. National Center of Technology Innovation for Digital Construction, Huazhong University of Science and Technology; Wuhan, China.
2. School of Artificial Intelligence and Automation, Huazhong University of Science and Technology; Wuhan, China.
3. School of Civil Engineering, Harbin Institute of Technology; Harbin, China.
4. Department of Civil & Environment Engineering, University of Maryland; College Park, USA.




## 1. Install 
### 1.1 Install habitat-lab 
```bash
# clone our repo
git clone https://github.com/HUST-ZLMgroup/image_goal_nav
cd image_goal_nav

# clone habitat-lab code
git submodule init
git submodule update

# create conda env
conda create -n image_goal_nav 

# install habitat-sim
conda install habitat-sim=0.2.2 withbullet headless -c conda-forge -c aihabitat

# install pytorch (>=1.10)
pip install torch

# install habitat-lab and habitat-baselines
cd habitat-lab
git checkout 1f7cfbdd3debc825f1f2fd4b9e1a8d6d4bc9bfc7
pip install -e habitat-lab 
pip install -e habitat-baselines
```
### 1.2 Install other requirements 
```bash
cd ..
pip install -r requirements.txt
```

## 2. Prepare dataset 
<!-- 
| ObjectNav   |   Gibson     | train    |  [objectnav_gibson_train](https://utexas.box.com/s/7qtqqkxa37l969qrkwdn0lkwitmyropp)    | `./data/datasets/zer/objectnav/gibson/v1/` |
| ObjectNav   |   Gibson     | val    |  [objectnav_gibson_val](https://utexas.box.com/s/wu28ms025o83ii4mwfljot1soj5dc7qo)    | `./data/datasets/zer/objectnav/gibson/v1/` | -->

### 2.1 Download Datasets 

For gibson dataset, we borrow the episodes generated from [`ZER`](https://github.com/ziadalh/zero_experience_required). We then follow the original [imagenav paper](https://arxiv.org/abs/2101.05181) to test . 

### 2.2 Download Scene Datasets 
Please read the [official guidance](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#gibson-and-3dscenegraph-datasets) to download `Gibson`, `HM3D`, and `MP3D` scene datasets, and put them in the `data/scene_datasets` directory using lower-case naming. 

## 3.  Training 

### 3.1 Train the Image-Goal Navigation Agent
```bash
python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=15344 --nnodes=1 \
--node_rank=0 --master_addr=127.0.0.1 \
main.py \
--exp-config exp_config/gibson_experiment.yaml,agent,task_objective,gibson_source,task_observations,context_prior \
--run-type train --model-dir results/train
```

## 4. Run Evaluation! 
### 4.1 Download the Trained Model to Reproduce the Results 

Download the trained checkpoint [model]( https://pan.baidu.com/s/1Hriw1U9iVpznKztb-qTnzg?pwd=1022 pwd: 1022), and move to $checkpoint_path.

Eval the model 

```bash
python -m torch.distributed.launch \
--nproc_per_node=1 --master_port=15244 --nnodes=1 \
--node_rank=0 --master_addr=127.0.0.1 \
main.py \
--exp-config exp_config/gibson_experiment.yaml,agent,task_objective,gibson_source,task_observations,context_prior,evaluation \
--run-type eval --model-dir results/eval_gibson \
habitat_baselines.eval_ckpt_path_dir $checkpoint_path
```



## Cite This Paper! 
