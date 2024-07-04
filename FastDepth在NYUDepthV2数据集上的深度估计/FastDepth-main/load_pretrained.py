import os
import os.path
import time
from collections import OrderedDict
import torch
from models import load_network_structure, weights_init
pool_layer_list = ['enc_layer4.conv2.2.weight', 'enc_layer2.conv2.2.weight', 'enc_layer7.conv2.2.weight', 'enc_layer14.conv2.2.weight']
def load_pretrained_encoder(encoder,weights_path,backbone):
    choice=load_network_structure()
    if backbone == 'mobilenetv2':
        state_dict = torch.load(f'{weights_path}/Mobilenetv2_pretrained.pth')
    else:
        checkpoint = torch.load(f'{weights_path}/model_best.pth.tar')
        state_dict = checkpoint['state_dict']
    target_state_dict = OrderedDict()
    for k, v in state_dict.items():
        target_state_dict[k] = v

    new_dict = encoder.state_dict()

    if 'pool' in choice: # 如果池化-反池化，由于池化层没参数，把池化替换下来的卷积核参数删除
        for f in new_dict.keys():
            if f in pool_layer_list:# 如果属于池化层
                print('pool layer popped')
            elif f in target_state_dict:
                new_dict[f] = target_state_dict[f] # 不变

    elif 'carafe' in choice:
        for f in new_dict.keys():
            if 'enc_layer0.0.weight' in f or 'comp' in f or 'content' in f:  # 如果属于carafe层
                print('carafe layer: random weights ')
            elif f in target_state_dict:
                new_dict[f] = target_state_dict[f]  # 其余不变

    else:#对于卷积下采样
        for f, b  in zip(new_dict,target_state_dict):
            new_dict[f] = target_state_dict[b]

    encoder.load_state_dict(new_dict,strict=False)


    return encoder

def load_pretrained_fastdepth(model,weights_path):
        assert os.path.isfile(weights_path), "No pretrained model found. abort.."
        print('Model found, loading...')
        checkpoint = torch.load(weights_path)
        model_state_dict = checkpoint['model_state_dict']
        args = checkpoint['args']
        criterion = args.criterion
        model.load_state_dict(model_state_dict)
        print('Finished loading')
        return model,criterion

