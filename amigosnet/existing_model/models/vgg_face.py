# -*- coding: utf-8 -*-

import cv2, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchfile
import sys



class VGGnet(nn.Module):
    def __init__(self):
        super().__init__()

        self.basenet = nn.ModuleDict(
            {
                'conv_1_1-conv': nn.Conv2d(3, 64, 3, stride=1, padding=1),
                'conv_1_1-relu': nn.ReLU(inplace=True),
                'conv_1_2-conv': nn.Conv2d(64, 64, 3, stride=1, padding=1),
                'conv_1_2-relu': nn.ReLU(inplace=True),
                'conv_1_2-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                'conv_2_1-conv': nn.Conv2d(64, 128, 3, stride=1, padding=1),
                'conv_2_1-relu': nn.ReLU(inplace=True),
                'conv_2_2-conv': nn.Conv2d(128, 128, 3, stride=1, padding=1),
                'conv_2_2-relu': nn.ReLU(inplace=True),
                'conv_2_2-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                'conv_3_1-conv': nn.Conv2d(128, 256, 3, stride=1, padding=1),
                'conv_3_1-relu': nn.ReLU(inplace=True),
                'conv_3_2-conv': nn.Conv2d(256, 256, 3, stride=1, padding=1),
                'conv_3_2-relu': nn.ReLU(inplace=True),
                'conv_3_3-conv': nn.Conv2d(256, 256, 3, stride=1, padding=1),
                'conv_3_3-relu': nn.ReLU(inplace=True),
                'conv_3_3-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                'conv_4_1-conv': nn.Conv2d(256, 512, 3, stride=1, padding=1),
                'conv_4_1-relu': nn.ReLU(inplace=True),
                'conv_4_2-conv': nn.Conv2d(512, 512, 3, stride=1, padding=1),
                'conv_4_2-relu': nn.ReLU(inplace=True),
                'conv_4_3-conv': nn.Conv2d(512, 512, 3, stride=1, padding=1),
                'conv_4_3-relu': nn.ReLU(inplace=True),
                'conv_4_3-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                'conv_5_1-conv': nn.Conv2d(512, 512, 3, stride=1, padding=1),
                'conv_5_1-relu': nn.ReLU(inplace=True),
                'conv_5_2-conv': nn.Conv2d(512, 512, 3, stride=1, padding=1),
                'conv_5_2-relu': nn.ReLU(inplace=True),
                'conv_5_3-conv': nn.Conv2d(512, 512, 3, stride=1, padding=1),
                'conv_5_3-relu': nn.ReLU(inplace=True),
                'conv_5_3-maxp': nn.MaxPool2d(kernel_size=2, stride=2)
            }
        )

        self.fc_1 = nn.ModuleDict(
            {
                'fc_1': nn.Linear(512 * 3 * 3, 4096),
                'fc_1-act': nn.ReLU(inplace=True),
                'fc_1-dropout': nn.Dropout(0.5)
            }
        )

        self.fc_2 = nn.ModuleDict(
            {
                'fc_2': nn.Linear(4096, 4096),
                'fc_2-act': nn.ReLU(inplace=True),
                'fc_2-dropout': nn.Dropout(0.5),
            }
        )

        self.num_heads = 2
        self.heads = nn.ModuleDict(
            {
                'fc_3': nn.Linear(4096, self.num_heads)#,
            }
        )



    def forward(self, x):

        for k, v in self.basenet.items():
            x = v(x)
        x = x.view(x.size(0), -1)

        # TODO: add comment
        for k, v in self.fc_1.items():
            x = v(x)
        for k, v in self.fc_2.items():
            x = v(x)
        for k, v in self.heads.items():
            x = v(x)

        return x


class VGGnetAdapted(nn.Module):
    def __init__(self, extra_feature_shape=None):
        super().__init__()

        self.basenet = nn.ModuleDict(
            {
                'conv_1_1-conv': nn.Conv2d(3, 64, 3, stride=1, padding=1),
                'conv_1_1-relu': nn.ReLU(inplace=True),
                'conv_1_2-conv': nn.Conv2d(64, 64, 3, stride=1, padding=1),
                'conv_1_2-relu': nn.ReLU(inplace=True),
                'conv_1_2-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                'conv_2_1-conv': nn.Conv2d(64, 128, 3, stride=1, padding=1),
                'conv_2_1-relu': nn.ReLU(inplace=True),
                'conv_2_2-conv': nn.Conv2d(128, 128, 3, stride=1, padding=1),
                'conv_2_2-relu': nn.ReLU(inplace=True),
                'conv_2_2-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                'conv_3_1-conv': nn.Conv2d(128, 256, 3, stride=1, padding=1),
                'conv_3_1-relu': nn.ReLU(inplace=True),
                'conv_3_2-conv': nn.Conv2d(256, 256, 3, stride=1, padding=1),
                'conv_3_2-relu': nn.ReLU(inplace=True),
                'conv_3_3-conv': nn.Conv2d(256, 256, 3, stride=1, padding=1),
                'conv_3_3-relu': nn.ReLU(inplace=True),
                'conv_3_3-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                'conv_4_1-conv': nn.Conv2d(256, 512, 3, stride=1, padding=1),
                'conv_4_1-relu': nn.ReLU(inplace=True),
                'conv_4_2-conv': nn.Conv2d(512, 512, 3, stride=1, padding=1),
                'conv_4_2-relu': nn.ReLU(inplace=True),
                'conv_4_3-conv': nn.Conv2d(512, 512, 3, stride=1, padding=1),
                'conv_4_3-relu': nn.ReLU(inplace=True),
                'conv_4_3-maxp': nn.MaxPool2d(kernel_size=2, stride=2),
                'conv_5_1-conv': nn.Conv2d(512, 512, 3, stride=1, padding=1),
                'conv_5_1-relu': nn.ReLU(inplace=True),
                'conv_5_2-conv': nn.Conv2d(512, 512, 3, stride=1, padding=1),
                'conv_5_2-relu': nn.ReLU(inplace=True),
                'conv_5_3-conv': nn.Conv2d(512, 512, 3, stride=1, padding=1),
                'conv_5_3-relu': nn.ReLU(inplace=True),
                'conv_5_3-maxp': nn.MaxPool2d(kernel_size=2, stride=2)
            }
        )

        self.fc_1 = nn.ModuleDict(
            {
                'fc_1': nn.Linear(512 * 3 * 3, 4096),
                'fc_1-act': nn.ReLU(inplace=True),
                'fc_1-dropout': nn.Dropout(0.5)
            }
        )
        self.fc_1_additional = nn.ModuleDict(
            {
                'fc_1_additional': nn.Linear(extra_feature_shape, 4096),
                'fc_1_additional-act': nn.ReLU(inplace=True),
                'fc_1_additional-dropout': nn.Dropout(0.5)
            }
        )

        self.fc_2 = nn.ModuleDict(
            {
                'fc_2': nn.Linear(4096 + extra_feature_shape, 4096),
                'fc_2-act': nn.ReLU(inplace=True),
                'fc_2-dropout': nn.Dropout(0.5),
            }
        )

        self.num_heads = 2
        self.heads = nn.ModuleDict(
            {
                'fc_3': nn.Linear(4096, self.num_heads)#,
            }
        )


    def forward(self, x, additional_feature):

        for k, v in self.basenet.items():
            x = v(x)

        x = x.view(x.size(0), -1)

        # TODO: add comment
        for k, v in self.fc_1.items():
            x = v(x)

        for k, v in self.fc_1_additional.items():
            y = v(additional_feature)
        x = torch.cat((x, y), axis=1)

        for k, v in self.fc_2.items():
            x = v(x)
        for k, v in self.heads.items():
            x = v(x)

        return x


def initialise_model_affect_adapted(file_path, feature_size, affectnet):
    device = torch.device("cuda:1")
    model = VGGnetAdapted(feature_size)
    # model = torch.nn.DataParallel(model, device_ids=[1])
    # net = model.cuda()
    model.to(device)

    if affectnet == True:
        state = torch.load(file_path,  map_location=lambda storage, loc: storage)
        del state['fc_2.fc_2.weight']
        del state['fc_2.fc_2.bias']
        model.load_state_dict(state, strict=False)
    model.train()
    return model


def initialise_model_affect(file_path, affectnet):
    device = torch.device("cuda:1")
    model = VGGnet()
    # model = torch.nn.DataParallel(model, device_ids=[1])
    # net = model.cuda()
    model.to(device)
    if affectnet == True:
        state = torch.load(file_path,  map_location=lambda storage, loc: storage)
        model.load_state_dict(state)
    model.train()
    return model

