import utils
import backbone
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from io_utils import parse_args


# get a decoder for BSRTrain to use after BSRTrain with following parameters    
# Linear, 512×512
# Linear, 512×6272
# Reshape to 32×14×14
# 2× 2 Deconv-ReLU, 32 filters, stride 2, padding 0
# 2× 2 Deconv-ReLU, 32 filters, stride 2, padding 0
# 2× 2 Deconv-ReLU, 64 filters, stride 2, padding 0
# 2× 2 Deconv-ReLU, 64 filters, stride 2, padding 0
# 3×3 Conv-Sigmoid, 3 filters, stride 1, padding 1

class Decoder(nn.Module):
    def __init__(self,):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 6272),
            nn.ReLU(),
            nn.Unflatten(-1, (32, 14, 14)),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


class BSRTrain(nn.Module):
    def __init__(self, model_func, num_class, lamda=0.001):
        super(BSRTrain, self).__init__()
        self.feature = model_func()

        self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()
        self.lamda = lamda

    def forward(self, x):
        x = Variable(x.cuda())
        feature = self.feature(x)
        u, s, v = torch.svd(feature.t())
        BSR = torch.sum(torch.pow(s, 2))
        scores = self.classifier(feature)
        return scores, feature, BSR

    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores, BSR = self.forward(x)
        loss_c = self.loss_fn(scores, y)
        loss = loss_c + self.lamda * BSR

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item() * 100 / (y.size(0) + 0.0), y.size(0))

        return loss

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = 10
        avg_loss = 0
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = self.forward_loss(x, y)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()
            if i % print_freq == 0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f} | Top1 Val {:f} | Top1 Avg {:f}'.
                      format(epoch, i, len(train_loader), avg_loss / float(i + 1), self.top1.val, self.top1.avg))


class PBSRTrain(nn.Module):
    def __init__(self, model_func, num_class, P_matrix, lamda=0.001):
        super(PBSRTrain, self).__init__()

        self.resnet1 = nn.Sequential(*list(model_func()._modules.values())[0][0:6])
        self.resnet2 = nn.Sequential(*list(model_func()._modules.values())[0][6:7])
        self.resnet3 = nn.Sequential(*list(model_func()._modules.values())[0][7:8])
        self.layer1 = nn.Sequential(*list(model_func()._modules.values())[0][8:])

        self.classifier = nn.Linear(model_func().final_feat_dim - 1, num_class)
        self.classifier.bias.data.fill_(0)

        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()

        self.P_matrix = P_matrix
        self.lamda = lamda

    def forward(self, x):
        x = Variable(x.cuda())
        out1 = self.resnet1(x)
        out2 = self.resnet2(out1)
        out3 = self.resnet3(out2)
        fea_b = self.layer1(out3)
        fea_e = torch.mm(fea_b, self.P_matrix)
        u, s, v = torch.svd(fea_e.t())
        BSR = torch.sum(torch.pow(s, 2))
        scores = self.classifier(fea_e)
        return scores, BSR

    def forward_loss(self, x, y):
        y = Variable(y.cuda())

        scores, BSR = self.forward(x)

        loss_c = self.loss_fn(scores, y)
        loss = loss_c + self.lamda * BSR

        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item() * 100 / (y.size(0) + 0.0), y.size(0))

        return loss

if __name__ == '__main__':
    params = parse_args('train')
    model = ResNet10 = backbone.ResNet10
    model = BSRTrain(model, params.num_classes, lamda=params.lamda).to('cuda')
    decoder = Decoder().to('cuda')

    out = model(torch.randn(10,3,224,224))
    print(f"out[0]: {out[0].shape}, out[1]: {out[1].shape}")
    out2 = decoder(out[1])
    print(f"out2: {out2.shape}")
    pdb.set_trace()