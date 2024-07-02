# dive-into-model-knowledge-distillation
This is an implementation of model distillation, which includes basic concepts and algorithm 
implementation.

**Can help you quickly get started with model knowledge distillation**

However, it is worth noting that all code does not run in a command-line manner. You can refer to the 
readme in each directory to learn and execute the code.

***
## Prerequisites
- python >= 3.5
- torch==2.3.1
- torchvision==0.16.0

You can install required packages by:

```bash
pip3 install -r requirements.txt
```
***

Here are some test results：

1. SampleNet

|            模型设置            |  教师模型 : 蒸馏模型大小  |  教师模型 : 蒸馏模型精度  |   教师模型 : 蒸馏模型训练时间   |
|:--------------------------:|:---------------:|:---------------:|:-------------------:|
| alpha=0.3, T=7, mode='cse' | 9.583M : 0.068M | 98.030 : 91.750 | 148.745s : 164.497s |
| alpha=0.5, T=7, mode='cse' | 9.583M : 0.068M | 97.810 : 92.280 | 157.503s : 169.399s |
| alpha=0.5, T=7, mode='kl'  | 9.583M : 0.068M | 97.920 : 92.040 | 148.770s : 165.305s |

2. Resnet

|           模型设置            |   教师模型 : 蒸馏模型大小   |  教师模型 : 蒸馏模型精度  |    教师模型 : 蒸馏模型训练时间    |
|:-------------------------:|:-----------------:|:---------------:|:---------------------:|
| alpha=0.3, T=7, mode='kl' | 87.315M : 46.828M | 73.660 : 72.780 | 1678.092s : 3669.204s |


#### 　　<center>Finally, welcome to download and provide feedback.</center>