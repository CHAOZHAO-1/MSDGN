import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import torch
import itertools
import mmd
import torch.nn as nn
from utils import *
from torch.autograd import Variable
import  numpy as np
import torch.nn.functional as F


def kl_categorical(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                  - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(_kl)



class CNN_1D(nn.Module):

    def __init__(self, num_classes=31):
        super(CNN_1D, self).__init__()
        # self.sharedNet = resnet18(False)
        # self.cls_fc = nn.Linear(512, num_classes)

        self.sharedNet = CNN()
        self.cls_fc = nn.Linear(256, num_classes)


    def forward(self, source):

        # source= source.unsqueeze(1)

        feature = self.sharedNet(source)
        source=self.cls_fc(feature)

        return source



class CNN(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, num_classes=10):
        super(CNN, self).__init__()


        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            )  # 32, 12,12     (24-2) /2 +1


        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16,stride=1),  # 128,8,8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))# 128, 4,4

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5,stride=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5,stride=1),  # 128,8,8
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4)
        )





        # self.fc = nn.Linear(256, num_classes)

    def forward(self, x):

        x = x.unsqueeze(1)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)

        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # print(x.shape)


        x = x.view(x.size(0), -1)

        # x = self.layer5(x)

        # x = self.fc(x)

        return x


class CNN_CSD(nn.Module):
    def __init__(self, pretrained=False, in_channel=1, num_classes=10):
        super(CNN_CSD, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=1),  # 32, 24, 24
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )  # 32, 12,12     (24-2) /2 +1

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16, stride=1),  # 128,8,8
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))  # 128, 4,4

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1),  # 32, 24, 24
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=1),  # 128,8,8
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(16)
        )


        K = 2
        self.sms = torch.nn.Parameter(torch.normal(0, 1e-1, size=[K, 1024, num_classes], dtype=torch.float, device='cuda'),
                                      requires_grad=True)
        self.sm_biases = torch.nn.Parameter(torch.normal(0, 1e-1, size=[K, num_classes], dtype=torch.float, device='cuda'),
                                            requires_grad=True)

        self.embs = torch.nn.Parameter(
            torch.normal(mean=0., std=1e-4, size=[3, K - 1], dtype=torch.float, device='cuda'), requires_grad=True)
        self.cs_wt = torch.nn.Parameter(torch.normal(mean=.1, std=1e-4, size=[], dtype=torch.float, device='cuda'),
                                        requires_grad=True)

        # self.fc = nn.Linear(256, num_classes)

    def forward(self, x,uids):

        x = x.unsqueeze(1)
        # print(x.shape)

        x = self.layer1(x)
        # print(x.shape)

        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.layer4(x)
        # x = self.layer5(x)
        # print(x.shape)


        x = x.view(x.size(0), -1)

        # x = self.layer5(x)

        # x = self.fc(x)

        w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
        # 8th Layer: FC and return unscaled activations
        logits_common = torch.matmul(x, w_c) + b_c

        c_wts = torch.matmul(uids, self.embs)
        # B x K
        batch_size = uids.shape[0]
        c_wts = torch.cat((torch.ones((batch_size, 1), dtype=torch.float, device='cuda') * self.cs_wt, c_wts), 1)
        c_wts = torch.tanh(c_wts)
        w_d, b_d = torch.einsum("bk,krl->brl", c_wts, self.sms), torch.einsum("bk,kl->bl", c_wts, self.sm_biases)
        logits_specialized = torch.einsum("brl,br->bl", w_d, x) + b_d

        return logits_specialized, logits_common, x


class M2(nn.Module):

    def __init__(self, num_classes=31):
        super(M2, self).__init__()


        self.sharedNet1 = CNN()
        self.sharedNet2 = CNN()
        self.sharedNet3 = CNN_CSD()

        self.cls_fc1 = nn.Linear(256, num_classes)
        self.cls_fc2 = nn.Linear(256, num_classes)



    def forward(self, data, flag,uids=None):

        if flag==1:
            feature = self.sharedNet1(data)
            label_pre = self.cls_fc1(feature)
            return feature, label_pre
        if flag==2:
            feature = self.sharedNet2(data)
            label_pre = self.cls_fc2(feature)
            return feature, label_pre
        if flag==3:
            ogits_specialized, logits_common, x = self.sharedNet3(data,uids)

            return ogits_specialized, logits_common, x

