
#author:zhaochao time:2022/8/30

import torch
import torch.nn.functional as F

from torch.autograd import Variable
import os
import math
import data_loader_1d
import resnet18_1d as models
import torch.nn as nn

import numpy as np

from utils import *

import mmd


def one_hot(ids, depth):
    z = np.zeros([len(ids), depth])
    z[np.arange(len(ids)), ids] = 1
    return z

def EntropyLoss(input_):
    mask = input_.ge(0.000001)# 逐元素比较
    mask_out = torch.masked_select(input_, mask)# 筛选
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Training settings



momentum = 0.9
no_cuda = False
seed = 8
log_interval = 10
l2_decay = 5e-4


def selection(src_unlabeled_psudo_1,src_unlabeled_pred_1,source_unlabeled_pred_1,real_label):

    for class_num in range(9):
        m = nn.Softmax(dim=1)

        S2_C1_index = (src_unlabeled_psudo_1 == class_num)  ###选出伪标签为0的样本

        s2 = torch.sum(S2_C1_index != 0)

        S2_C1_instance_H = torch.zeros(s2)

        S2_C1 = source_unlabeled_pred_1[S2_C1_index]

        for ttt in range(s2):
            S2_C1_instance_H[ttt] = EntropyLoss(m(S2_C1[ttt].reshape(1, -1)))

        _, S2_C1_sort_index = torch.sort(S2_C1_instance_H, descending=False)


        S2_C1_selection_index = S2_C1_sort_index[:int(s2*ratio)]

        if class_num == 0:
            Final_pred = src_unlabeled_pred_1[S2_C1_index][S2_C1_selection_index]
            Final_label = src_unlabeled_psudo_1[S2_C1_index][S2_C1_selection_index]
            Final_real_label=real_label[S2_C1_index][S2_C1_selection_index]
        else:
            Final_pred = torch.cat([Final_pred, src_unlabeled_pred_1[S2_C1_index][S2_C1_selection_index]], dim=0)
            Final_label = torch.cat([Final_label, src_unlabeled_psudo_1[S2_C1_index][S2_C1_selection_index]], dim=0)
            Final_real_label =  torch.cat([Final_real_label, real_label[S2_C1_index][S2_C1_selection_index]], dim=0)


        selction_acc=torch.eq(Final_label, Final_real_label.squeeze(dim=-1)).float().mean()


    return  Final_label,Final_pred,selction_acc



def train(model):
    src_iter = iter(src_loader)

    Train_Loss_list = []
    Train_Accuracy_list = []
    Test_Loss_list = []
    Test_Accuracy_list = []

    for i in range(1, iteration + 1):
        model.train()
        LEARNING_RATE = lr / math.pow((1 + 10 * (i - 1) / (iteration)), 0.75)
        if (i - 1) % 100 == 0:
            print('learning rate{: .4f}'.format(LEARNING_RATE))

        optimizer = torch.optim.Adam([
            {'params': model.sharedNet1.parameters()},
            {'params': model.sharedNet2.parameters()},
            {'params': model.sharedNet3.parameters()},

            {'params': model.cls_fc1.parameters(), 'lr': LEARNING_RATE},
            {'params': model.cls_fc2.parameters(), 'lr': LEARNING_RATE},


        ], lr=LEARNING_RATE / 10, weight_decay=l2_decay)

        try:
            src_data, src_label = src_iter.next()
        except Exception as err:
            src_iter = iter(src_loader)
            src_data, src_label = src_iter.next()

        if cuda:
            src_data, src_label = src_data.cuda(), src_label.cuda()


        domain_label = src_label[:, 1]


        domian_index_1 = (src_label[:, 1] == 0)
        domian_index_2 = (src_label[:, 1] == 1)
        domian_index_3 = (src_label[:, 1] == 2)


        source_labeled_data=src_data[domian_index_1]

        source_unlabeled_data_1 = src_data[domian_index_2]

        source_unlabeled_data_2 = src_data[domian_index_3]



        src_label=src_label[:,0]



        src_1_label = src_label[domian_index_1]

        src_2_label = src_label[domian_index_2]
        src_3_label = src_label[domian_index_3]




        src_labeled_feature_1,src_labeled_pred_1=model(source_labeled_data,flag=1)

        source_unlabeled_feature_1, source_unlabeled_pred_1 = model(source_unlabeled_data_1, flag=1)


        src_labeled_feature_2, src_labeled_pred_2 = model(source_labeled_data, flag=2)

        source_unlabeled_feature_2, source_unlabeled_pred_2 = model(source_unlabeled_data_2, flag=2)



        s1, d = src_labeled_feature_1.shape
        s2, d = source_unlabeled_feature_1.shape

        minlen=min([s1,s2])


        feature1=src_labeled_feature_1[:minlen,:]
        feature2=source_unlabeled_feature_1[:minlen,:]


        s3, d = src_labeled_feature_2.shape
        s4, d = source_unlabeled_feature_2.shape

        minlen = min([s3, s4])

        feature3 = src_labeled_feature_2[:minlen, :]
        feature4 = source_unlabeled_feature_2[:minlen, :]

        cls_loss = F.nll_loss(F.log_softmax(src_labeled_pred_1, dim=1), src_1_label)+F.nll_loss(F.log_softmax(src_labeled_pred_2, dim=1), src_1_label)
        MMD_loss = a*mmd.mmd_rbf_noaccelerate(feature1, feature2)+b*mmd.mmd_rbf_noaccelerate(feature3, feature4)







        NUM_DOMAINS = 3
        oh_dids = torch.tensor(one_hot(domain_label.cpu(), NUM_DOMAINS), dtype=torch.float, device='cuda')


        specific_logit, class_logit,_ = model(src_data, flag=3,uids=oh_dids)


        sms = model.sharedNet3.sms
        K = 2
        diag_tensor = torch.stack([torch.eye(K) for _ in range(class_num)], dim=0).cuda()
        cps = torch.stack(
            [torch.matmul(sms[:, :, _], torch.transpose(sms[:, :, _], 0, 1)) for _ in range(class_num)], dim=0)

        orth_loss = torch.mean((cps - diag_tensor) ** 2)




        src_labeled_pred    = class_logit[domian_index_1]
        src_unlabeled_pred_1= class_logit[domian_index_2]
        src_unlabeled_pred_2= class_logit[domian_index_3]


        src_unlabeled_psudo_1=  source_unlabeled_pred_1.data.max(1)[1]
        src_unlabeled_psudo_2 = source_unlabeled_pred_2.data.max(1)[1]



        lambd = 2 / (1 + math.exp(-10 * (i) / iteration)) - 1


        m = nn.Softmax(dim=1)
        H = HC1*EntropyLoss(m(source_unlabeled_pred_1))\
            +HC2*EntropyLoss(m(source_unlabeled_pred_2))

        S2_selection_label, S2_selection_pred, s2_acc = selection(src_unlabeled_psudo_1, src_unlabeled_pred_1,
                                                                  source_unlabeled_pred_1, src_2_label)

        S3_selection_label, S3_selection_pred, s3_acc = selection(src_unlabeled_psudo_2, src_unlabeled_pred_2,
                                                                  source_unlabeled_pred_2, src_3_label)

        cls_loss_Final = F.nll_loss(F.log_softmax(src_labeled_pred, dim=1), src_1_label) + \
                         F.nll_loss(F.log_softmax(S2_selection_pred, dim=1), S2_selection_label) + \
                         F.nll_loss(F.log_softmax(S3_selection_pred, dim=1), S3_selection_label)


        loss = cls_loss + MMD_loss + cls_loss_Final*lambd + H +orth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % log_interval == 0:
            print('Train iter: {} [({:.0f}%)]\tLoss: {:.6f}\tsoft_Loss: {:.6f}\tMMD_Loss: {:.6f}'.format(
                i, 100. * i / iteration, loss.item(), cls_loss.item(),MMD_loss))

        if i % (log_interval * 10) == 0:


            train_correct, train_loss = test_source(model, src_loader)


            test_correct, test_loss = test_target(model, tgt_test_loader)


            Train_Accuracy_list.append(train_correct.cpu().numpy() / len(src_loader.dataset))
            Train_Loss_list.append(train_loss)

            Test_Accuracy_list.append(test_correct.cpu().numpy() / len(tgt_test_loader.dataset))
            Test_Loss_list.append(test_loss)



def test_source(model,test_loader):
    model.eval()
    test_loss = 0
    correct_2 = 0
    correct_3 = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)

            domian_index_2 = (tgt_test_label[:, 1] == 1)
            domian_index_3 = (tgt_test_label[:, 1] == 2)

            tgt_test_label=tgt_test_label[:,0]


            tgt_test_label_2 = tgt_test_label[domian_index_2]
            tgt_test_label_3 = tgt_test_label[domian_index_3]

            tgt_test_data_2=tgt_test_data[domian_index_2]
            tgt_test_data_3=tgt_test_data[domian_index_3]


            _,tgt_pred_2= model(tgt_test_data_2, flag=1)
            _,tgt_pred_3 =model(tgt_test_data_3, flag=2)


            pred_2 = tgt_pred_2.data.max(1)[1]  # get the index of the max log-probability
            pred_3 = tgt_pred_3.data.max(1)[1]

            correct_2 += pred_2.eq(tgt_test_label_2.data.view_as(pred_2)).cpu().sum()
            correct_3 += pred_3.eq(tgt_test_label_3.data.view_as(pred_3)).cpu().sum()


    print('\nAccuracy: {}/{} ({:.2f}%) Accuracy: {}/{} ({:.2f}%) \n'.format(correct_2,7200,10000. * correct_2 /7200, correct_3,7200,10000. * correct_3 /7200))
    return correct_2,test_loss


def test_target(model,test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for tgt_test_data, tgt_test_label in test_loader:
            if cuda:
                tgt_test_data, tgt_test_label = tgt_test_data.cuda(), tgt_test_label.cuda()
            tgt_test_data, tgt_test_label = Variable(tgt_test_data), Variable(tgt_test_label)


            dummy_ids = one_hot(np.zeros(len(tgt_test_data), dtype=np.int32), 3)

            specific_logit, class_logit, _ = model(tgt_test_data,flag=3,uids=
                                                   torch.tensor(dummy_ids, dtype=torch.float, device='cuda'))
            _, cls_pred = class_logit.max(dim=1)
            _, specific_pred = specific_logit.max(dim=1)

            correct += torch.sum(cls_pred == tgt_test_label.data)



    print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(tgt_name,test_loss, correct, len(test_loader.dataset),10000. * correct / len(test_loader.dataset)))
    return correct,test_loss





def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)

    print('Total:{} Trainable:{}'.format( total_num, trainable_num))


if __name__ == '__main__':
    # setup_seed(seed)

    iteration = 10000
    batch_size = 512
    lr = 0.001

    a=1
    b=1
    FFT = False

    dataset = 'SQbearing'
    class_num = 9
    src_tar = np.array([[0, 1, 2, 10], [1, 0, 2, 10], [2, 0, 1, 10],
                        [0, 1, 3, 10], [1, 0, 3, 10], [3, 0, 1, 10],
                        [0, 2, 3, 10], [2, 0, 3, 10], [3, 0, 2, 10],
                        [1, 2, 3, 10], [2, 1, 3, 10], [3, 1, 2, 10]
                        ])

    ratio= 0.8

    HC1 = 0.1

    HC2 =0.1

    for taskindex in range(12):

        source1 = src_tar[taskindex][0]
        source2 = src_tar[taskindex][1]
        source3 = src_tar[taskindex][2]
        target = src_tar[taskindex][3]
        src = src_tar[taskindex][:-1]

        for repeat in range(10):

            root_path = '/home/zhaochao/research/DTL/data/' + dataset + 'data' + str(class_num) + '.mat'

            src_name1 = 'load' + str(source1) + '_train'
            src_name2 = 'load' + str(source2) + '_train'
            src_name3 = 'load' + str(source3) + '_train'

            tgt_name = 'load' + str(target) + '_train'
            test_name = 'load' + str(target) + '_test'

            cuda = not no_cuda and torch.cuda.is_available()
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)

            kwargs = {'num_workers': 0, 'pin_memory': True} if cuda else {}

            src_loader = data_loader_1d.load_training(root_path, src_name1, src_name2, src_name3, src, FFT,
                                                      class_num,
                                                      batch_size, kwargs)

            tgt_test_loader = data_loader_1d.load_testing(root_path, test_name, FFT, class_num,
                                                          batch_size, kwargs)

            src_dataset_len = len(src_loader.dataset)

            src_loader_len = len(src_loader)
            model = models.M2(num_classes=class_num)
            # get_parameter_number(model) 计算模型训练参数个数
            print(model)
            if cuda:
                model.cuda()
            train(model)

































