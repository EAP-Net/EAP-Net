# EAP-Net
Enhanced attention with pose assistance for Person-Reid

## Require
- [pytorch 1.0.0+](https://pytorch.org/)
- torchvision
- [yacs](https://github.com/rbgirshick/yacs)
- [ignite](https://pypi.org/project/pytorch-ignite/)

## DataSet
- Market-1501: [[google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op)
- DukeMTMC-reid
- [CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)
- MSMT17

## Test
1. Prepare pretrained model.
  In this project, We only provide the checkpoints and test code for our model.The code for model training will be updated later.If you want to test our model, you need to download the model's checkpoints in advance.
2. Change the config file.
  Our model test parameters is saved at  `./configs/inference.yml `. You need to update your own test parameters to this file, for example`path to the checkpoint`,`dataset name`.
