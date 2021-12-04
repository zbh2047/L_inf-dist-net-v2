# L_infinity-distance Net v2

## Introduction

L_infinity-distance Net is a theoretically principled neural network that is inherently robust to L_infinity-norm perturbations. It was first proposed in [ICML 2021](https://arxiv.org/abs/2102.05363) and implemented [here](https://github.com/zbh2047/L_inf-dist-net). L_infinity-distance Net consistently achieves state-of-the-art certified L_infinity robustness on multiple datasets. 

This github repository gives an upgrade of L_infinity-distance Net. The main features are described as follows.

### A faster CUDA implementation

We significantly speed up the calculation of L_p-distance operation with a better CUDA implementation, both for forward propagation and backward propagation. It therefore makes the training process considerably faster.

Our implementation adopts different parallel algorithms for the three cases: when p is a small integer (e.g. 8), when p is infinity, and when p is a general floating-point number. The extent of acceleration in each case is shown in the table below (tested on a NVIDIA RTX 3090 GPU).

Table: Comparison of the wall-clock time (seconds) of a training iteration (forward+backward propagation).

| p             | Previous implementation | Our implementation | Speed up |
| ------------- | ----------------------- | ------------------ | -------- |
| general       | 0.2927                  | 0.1866             | 56%      |
| small integer | 0.2927                  | 0.1133             | 160%     |
| infinity      | 0.09433                 | 0.07961            | 19%      |

With the accelerated CUDA implementation, training an L_infinity-distance net (800 epochs) only takes about 3.7 hours.

### An improved training strategy

We design an improved training strategy that substantially improves the certified robust accuracy on multiple datasets. Results are listed in the table below. See our latest paper (titled "[Boosting the Certified Robustness of L-infinity Distance Nets](https://arxiv.org/abs/2110.06850)") for details of the training strategy.

| Dataset  | eps    | Clean Acc | PGD-100 Acc | Certified Acc |
| -------- | ------ | --------- | ----------- | ------------- |
| MNIST    | 0.1    | 98.93     | 98.03       | 97.95         |
| MNIST    | 0.3    | 98.56     | 94.73       | 93.20         |
| CIFAR-10 | 2/255  | 60.61     | 54.28       | 54.12         |
| CIFAR-10 | 8/255  | 54.30     | 41.84       | **40.06**     |
| CIFAR-10 | 16/255 | 48.50     | 32.73       | 29.04         |

### A portable "core library"

To facilitate future research, we provide an interface containing the major components of this repository in the `core` folder. This library is portable and you can simply copy this folder to your work space and use it outside the box. We list the functionalities of the core library:

- A function `norm_dist(x, w, p)` that calculates the L_p distance between the input tensor  `x` and the parameter tensor `w` with CUDA acceleration (in [core/norm_dist.py](https://github.com/zbh2047/L_inf-dist-net-v2/tree/main/core/norm_dist.py)).
- A base class ` NormDistBase` that wraps the above function into a model layer suitable for pytorch `torch.nn.Module`. Then subclasses ` NormDist` and `NormDistConv` specialize the fully-connected layer and convolution layer  (in [core/modules/norm_dist.py](https://github.com/zbh2047/L_inf-dist-net-v2/tree/main/core/modules/norm_dist.py)).
- The library also supports Interval Bound Propagation (IBP). We provide IBP layer for the linear transformation, convolution, and commonly used activation functions  (in [core/modules/basic_modules.py](https://github.com/zbh2047/L_inf-dist-net-v2/tree/main/core/modules/basic_modules.py)) as well as the L-infinity distance function. They can be used to build the top MLP to form a composite architecture (L_infinity-distance net + MLP).



## Dependencies

- Pytorch 1.8.0 (or a later version)
- Tensorboard (optional)



## Getting Started with the Code

### Installation

After cloning this repo into your computer, first run the following command to install the CUDA extension.

```
python setup.py install --user
```

### Reproducing SOTA results

We provide complete training scripts to reproduce the results in our latest paper. These scripts are in the `command` folder. 

For example, to reproduce the result of CIFAR-10 with perturbation eps=8/255, simply run

```
bash command/cifar_0.03137.sh
```

### Reproducing baseline results in [ICML 2021](https://arxiv.org/abs/2102.05363)

For example, to reproduce the baseline results of L_infinity-distance net on CIFAR-10 dataset with perturbation eps=8/255, run the following command

```
python main.py --dataset CIFAR10 --model 'MLPModel(depth=6,width=5120,identity_val=10.0,scalar=False)' --loss 'hinge' --p-start 8 --p-end 1000 --epochs 0,0,100,750,800 --eps-test 0.03137 --eps-train 0.1569 -b 512 --lr 0.02 --gpu 0 -p 200 --seed 2021 --visualize
```

To reproduce the baseline results of L_infinity-distance net + MLP on CIFAR-10 dataset with perturbation eps=8/255, run the following command

```
python main.py --dataset CIFAR10 --model 'HybridModel(depth=7,width=5120,identity_val=10.0,hidden=512)' --loss 'crossentropy' --p-start 8 --p-end 1000 --epochs 0,100,100,750,800 --eps-test 0.03137 --eps-train 0.03451 -b 512 --lr 0.02 --gpu 0 -p 200 --seed 2021 --visualize
```



## Advanced Training Options

### Multi-GPU Training

We also support multi-GPU training using distributed data parallel. By default the code will use all available GPUs for training. To use a single GPU, add the following parameter `--gpu GPU_ID` where `GPU_ID` is the GPU ID. You can also specify `--world-size`, `--rank` and `--dist-url` for advanced multi-GPU training.

### Saving and Loading

The model is automatically saved when the training procedure finishes. Use `--checkpoint model_file_name.pth` to load a specified model before training. You can use `--start-epoch NUM_EPOCHS` to skip training and only test the model's performance for a pretrained model, where `NUM_EPOCHS` is the number of epochs in total.

### Displaying training curves

By default the code will generate five files named `train.log`, `test.log`,  `train_inf.log`, `test_inf.log`and `log.txt` which contain all training logs. If you want to further display training curves, you can add the parameter `--visualize` to show these curves using Tensorboard. 



## Pretrained Models

We also provide pretrained models with SOTA certified robust accuracy. These models can be downloaded [here](https://drive.google.com/drive/folders/1ybyWxotjjaxIIiHbV9Tcy965badTpRJx?usp=sharing). The models are compressed into 7z file format. To use these models, follow the **Saving and Loading** instruction above.



## Contact

Please contact [zhangbohang@pku.edu.cn](zhangbohang@pku.edu.cn)  if you have any question on our paper or the codes. Enjoy! 



## Citation

```
@article{zhang2021boosting,
      title={Boosting the Certified Robustness of L-infinity Distance Nets}, 
      author={Bohang Zhang and Du Jiang and Di He and Liwei Wang},
      year={2021},
      eprint={2110.06850},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

