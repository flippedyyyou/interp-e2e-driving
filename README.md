# Interpretable End-to-end Autonomous Driving
[[Project webpage]](https://sites.google.com/berkeley.edu/interp-e2e/) [[Paper]](https://arxiv.org/abs/2001.08726)

This repo contains code for [Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning](https://arxiv.org/abs/2001.08726). This work introduces an end-to-end autonomous driving approach which is able to handle complex urban scenarios, and at the same time generates a semantic birdeye mask interpreting how the learned agents reasons about the environment. This repo also provides implementation of popular model-free reinforcement learning algorithms (DQN, DDPG, TD3, SAC) on the urban autonomous driving problem in CARLA simulator. All of the algorithms take raw camera and lidar sensor inputs.

## System Requirements
- Ubuntu 16.04
- NVIDIA GPU with CUDA 10. See [GPU guide](https://www.tensorflow.org/install/gpu) for TensorFlow.

## Installation
1. Setup conda environment
```
$ conda create -n env_name python=3.6
$ conda activate env_name
```

2. Install the gym-carla wrapper following the installation steps 2-4 in [https://github.com/cjy1992/gym-carla](https://github.com/cjy1992/gym-carla).

3. Clone this git repo to an appropriate folder
```
$ git clone https://github.com/cjy1992/interp-e2e-driving.git
```

4. Enter the root folder of this repo and install the packages:
```
$ pip install -r requirements.txt
$ pip install -e .
```

## Usage
1. Enter the CARLA simulator folder and launch the CARLA server by:
```
$ ./CarlaUE4.sh -windowed -carla-port=2000
```
You can use ```Alt+F1``` to get back your mouse control.
Or you can run in non-display mode by:
```
$ DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000
```
It might take several seconds to finish launching the simulator.

2. Enter the root folder of this repo and run:
```
$ ./run_train_eval.sh
```
It will then connect to the CARLA simulator, collect exploration data, train and evaluate the agent. Parameters are stored in ```params.gin```. Set train_eval.agent_name from ['latent_sac', 'dqn', 'ddpg', 'td3', 'sac'] to choose the reinforcement learning algorithm.

3. Run `tensorboard --logdir logs` and open http://localhost:6006 to view training and evaluation information.

## Trouble Shootings
1. If out of system memory, change the parameter ```replay_buffer_capacity``` and ```initial_collect_steps``` the function ```tran_eval``` smaller.

2. If out of CUDA memory, set parameter ```model_batch_size``` or ```sequence_length``` of the function ```tran_eval``` smaller.

## Citation
If you find this useful for your research, please use the following.

```
@article{chen2020interpretable,
  title={Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning},
  author={Chen, Jianyu and Li, Shengbo Eben and Tomizuka, Masayoshi},
  journal={arXiv preprint arXiv:2001.08726},
  year={2020}
}
```

# Carla0.9.11

**问题1:** [trafic_manger runtime_error](https://github.com/carla-simulator/carla/issues/3543)
       
**解决方法:** 将tm_port设置为6000，0.9.11存在bug，使用默认的8000无法开启traffic_manger<br><br>
**问题2:** [lidar图像与效果图相差甚远](https://github.com/cjy1992/gym-carla/issues/31)

**解决方法:** 0.9.11雷达的成员发生了改变，查阅Carla官方文档，对原代码雷达部分进行修改，且需要将雷达旋转90°<br><br>
**问题3:** 除自车外其他车辆大部分静止

**解决方法:** 0.9.11多车辆设置为autopilot模式必须开启traffic_manger,并且开启同步模式，否则大部分车辆会静止<br><br>
**问题4:** lidar图像中的红线

**尚未解决**

# 数据可视化分析方法
使用tensorboard进行数据可视化分析，切换至相应虚拟环境，开启终端输入如下指令

```
tensorboard --logdir=/home/[user_name]/interp-e2e-driving/logs/carla-v0/[experiment_name]
```

## 实验记录

|experiment_num|experiment_name|obs_size|final_global_step|analysis|
|---|---|---|---|---|
|1|latent_sac|64|189000|因输入尺寸过小过于模糊，训练效果不佳|
|2|latent_sac2|128|3000|显存不够|

# 目前已知重要参数

## train_eval.py/gym_carla
obs_range:观察范围<br>
laser_bin:obs_range/laser_bin为render显示的图像分辨率，源代码默认为64<br>
input_names:输入通道名<br>
mask_names:mask通道名<br>

## sequential_latent_network.py
obs_size:输入图像的清晰度
