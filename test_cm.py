# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 19:43:19 2020

@author: Peter
"""

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import *
from taskcv_loader import CVDataLoader
from basenet import *
import os
from sklearn.manifold import TSNE
from itertools import cycle

# Training settings
parser = argparse.ArgumentParser(description='Visda Classification')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=8, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--optimizer', type=str, default='momentum', metavar='OP',
                    help='the name of optimizer')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--num_k', type=int, default=4, metavar='K',
                    help='how many steps to repeat the generator update')
parser.add_argument('--num-layer', type=int, default=2, metavar='K',
                    help='how many layers for classifier')
parser.add_argument('--name', type=str, default='board', metavar='B',
                    help='board dir')
parser.add_argument('--save', type=str, default='C:/Users/Peter/Desktop/prepared_for_check/results', metavar='B',
                    help='board dir')
parser.add_argument('--train_path', type=str, default=r'D:/AerialtoSatellite/RSDataset', metavar='B',
                    help='directory of source datasets')
parser.add_argument('--val_path', type=str, default=r'D:/AerialtoSatellite/UCMerced_LandUse', metavar='B',
                    help='directory of target datasets')
parser.add_argument('--resnet', type=str, default='101', metavar='B',
                    help='which resnet 18,50,101,152,200')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
train_path = args.train_path
val_path = args.val_path
num_k = args.num_k
num_layer = args.num_layer
batch_size = args.batch_size
save_path = args.save

num_stop_train = 445
num_stop_test = 900
num_class = 9
num_dataset = 900

data_transforms = {
    train_path: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    val_path: transforms.Compose([
        transforms.Scale(256),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
dsets = {x: datasets.ImageFolder(os.path.join(x), data_transforms[x]) for x in [train_path, val_path]}
dset_sizes = {x: len(dsets[x]) for x in [train_path, val_path]}
dset_classes = dsets[train_path].classes
print('classes' + str(dset_classes))
use_gpu = torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
train_loader = CVDataLoader()
train_loader.initialize(dsets[train_path], dsets[val_path], batch_size)
dataset = train_loader.load_data()
test_loader = CVDataLoader()
opt = args
test_loader.initialize(dsets[train_path], dsets[val_path], batch_size, shuffle=True)
dataset_test = test_loader.load_data()
option = 'resnet' + args.resnet
G1 = ResBase(option)
G2 = ResBase(option)
F1 = ResClassifier(num_layer=num_layer)
F2 = ResClassifier(num_layer=num_layer)
F1.apply(weights_init)
F2.apply(weights_init)
lr = args.lr
if args.cuda:
    G1.cuda()
    G2.cuda()
    F1.cuda()
    F2.cuda()
if args.optimizer == 'momentum':
    optimizer_g = optim.SGD(list(G1.features.parameters()) + list(G2.features.parameters()), lr=args.lr,
                            weight_decay=0.0005)
    optimizer_f = optim.SGD(list(F1.parameters()) + list(F2.parameters()), momentum=0.9, lr=args.lr,
                            weight_decay=0.0005)
elif args.optimizer == 'adam':
    optimizer_g = optim.Adam(list(G1.features.parameters()) + list(G2.features.parameters()), lr=args.lr,
                             weight_decay=0.0005)
    optimizer_f = optim.Adam(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)
else:
    optimizer_g = optim.Adadelta(G.features.parameters(), lr=args.lr, weight_decay=0.0005)
    optimizer_f = optim.Adadelta(list(F1.parameters()) + list(F2.parameters()), lr=args.lr, weight_decay=0.0005)

def get_L2norm_loss_self_driven(x):
    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 0.3
    l = ((x.norm(p=2, dim=1) - radius) ** 2).mean()
    return l

def test(epoch):
    G1.load_state_dict(torch.load('C:/Users/Peter/Desktop/prepared_for_check/checkpoints/epoch_30_101_tensor(92)_G1.pth'))
    G2.load_state_dict(torch.load('C:/Users/Peter/Desktop/prepared_for_check/checkpoints/epoch_30_101_tensor(92)_G2.pth'))
    F1.load_state_dict(torch.load('C:/Users/Peter/Desktop/prepared_for_check/checkpoints/epoch_30_101_tensor(92)_F1.pth'))
    F2.load_state_dict(torch.load('C:/Users/Peter/Desktop/prepared_for_check/checkpoints/epoch_30_101_tensor(92)_F2.pth'))
    G1.eval()
    G2.eval()
    F1.eval()
    F2.eval()
    test_loss = 0
    correct = 0
    correct2 = 0
    size = 0
    val = False

    acc_class = [0 for _ in range(num_class)]
    count_class = [0 for _ in range(num_class)]

    tsne_results = np.array([])
    tsne_labels = np.array([])


    pred_all = np.array([])
    label_all = np.array([])

    with torch.no_grad():
        for batch_idx, data in enumerate(dataset_test):
            if batch_idx * batch_size >= num_stop_test:
                break
            if args.cuda:
                data2 = data['T']
                target2 = data['T_label']
                if val:
                    data2 = data['S']
                    target2 = data['S_label']
                data2, target2 = data2.cuda(), target2.cuda()
            data1, target1 = Variable(data2), Variable(target2)
            data1 = data1.cuda()
            target1 = target1.cuda()

            output_temp, temp_visual_G1 = G1(data1)
            output, temp_visual_G2 = G2(data1)
            emb1, output1 = F1(output)
            emb2, output2 = F2(output)
            test_loss += F.nll_loss(output1, target1).item()
            pred = output1.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.eq(target1.data).cpu().sum()
            pred = output2.data.max(1)[1]  # get the index of the max log-probability
            k = target1.data.size()[0]
            correct2 += pred.eq(target1.data).cpu().sum()

            size += k

            pred_all = np.concatenate((pred_all, pred.cpu().numpy()))
            label_all = np.concatenate((label_all,target1.data.cpu().numpy()))
            
            index_temp = pred.eq(target1.data)
            for acc_index in range(k):
                temp_label_index = target1.data[acc_index]
                count_class[temp_label_index] += 1
                if index_temp[acc_index]:
                    acc_class[temp_label_index] += 1
            if len(tsne_labels)==0:
                tsne_results = output1.cpu().data.numpy()
                tsne_labels = target1.cpu().numpy()
            else:
                tsne_results = np.concatenate((tsne_results, output1.cpu().data.numpy()))
                tsne_labels = np.concatenate((tsne_labels, target1.cpu().numpy()))                    
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
        label_c = ['agricultural', 'beach', 'buildings', 'forest', 'harbor', 'overpass', 'parkinglot', 'residential', 'river']
        cmatrix = confusion_matrix(label_all, pred_all)

        cmn = cmatrix.astype('float') / cmatrix.sum(axis=1)[:, np.newaxis]
        fig = plt.figure(figsize=(13, 10))
        sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=label_c, yticklabels=label_c)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        fig.savefig('C:/Users/Peter/Desktop/prepared_for_check/results/MCD_cmatrix_9c_nfs.png', dpi=fig.dpi, bbox_inches='tight')

        plot_only = 1000
        tsne_model = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        tsne_transformed = tsne_model.fit_transform(tsne_results[:plot_only, :])
        tsne_labels = tsne_labels[:plot_only]
        cycol = ['black','red','green','brown','blue','gray','darkorchid','coral','darkblue'] #,'chocolate','olive','turquoise','steelblue','indigo','lightseagreen']
        #colors = cm.rainbow(np.linspace(0, 0.5, num_class))
        fig1 = plt.figure( figsize=(11, 10))
        for l, c, co, in zip(label_c, cycol, range(num_class)):
            plt.scatter(tsne_transformed[np.where(tsne_labels == co), 0],
                        tsne_transformed[np.where(tsne_labels == co), 1],
                        marker='o',
                        color=c,
                        linewidth='1',
                        alpha=0.8,
                        label=l)
        plt.axis("off")
        plt.legend(bbox_to_anchor=(1, 1), fontsize=26, fancybox=True, shadow=True)
        #plt.show()
        path = 'C:/Users/Peter/Desktop/prepared_for_check/results/tSNE_DTR_dataset3_9classes_' + str(epoch) + '.png'
        fig1.savefig(path, dpi=fig1.dpi, bbox_inches="tight")

        for print_index in range(len(acc_class)):
            print('Class:{},number:{}, Accuracy:{:.2f}%'.format(
                print_index, count_class[print_index],
                100. * acc_class[print_index] / count_class[print_index]))
        file_save =open("../tsne/acc.txt",mode = "a")
        for print_index in range(len(acc_class)):
            file_save.write('Class:{},number:{}, Accuracy:{:.2f}%'.format(
                print_index, count_class[print_index],
                100. * acc_class[print_index] / count_class[print_index]))
        test_loss = test_loss
        test_loss /= len(test_loader)  # loss function already averages over batch size
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)'.format(
            test_loss, correct, size,
            100. * correct / size, 100. * correct2 / size))
        file_save.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) ({:.0f}%)'.format(
            test_loss, correct, size,
            100. * correct / size, 100. * correct2 / size))
        file_save.close()
        value = max(100. * correct / size, 100. * correct2 / size)


test(30)
