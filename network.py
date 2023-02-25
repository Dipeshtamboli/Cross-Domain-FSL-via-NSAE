import utils
import backbone
import pdb
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from io_utils import parse_args


# get a decoder for NSAE_model to use after NSAE_model with following parameters    
# Linear, 512×512
# Linear, 512×6272
# Reshape to 32×14×14
# 2× 2 Deconv-ReLU, 32 filters, stride 2, padding 0
# 2× 2 Deconv-ReLU, 32 filters, stride 2, padding 0
# 2× 2 Deconv-ReLU, 64 filters, stride 2, padding 0
# 2× 2 Deconv-ReLU, 64 filters, stride 2, padding 0
# 3×3 Conv-Sigmoid, 3 filters, stride 1, padding 1

class NSAE_model(nn.Module):
    def __init__(self, model_func, num_class, lamda1=1, lamda2=1):
        super(NSAE_model, self).__init__()
        self.feature = model_func()

        self.classifier = nn.Linear(self.feature.final_feat_dim, num_class)
        self.classifier.weight.data.normal_(0, 0.01)
        self.classifier.bias.data.fill_(0.0)

        self.num_class = num_class
        self.loss_fn = nn.CrossEntropyLoss()
        self.top1 = utils.AverageMeter()
        self.lamda1 = lamda1
        self.lamda2 = lamda2

        self.decoder = nn.Sequential(
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 6272), nn.ReLU(),
            nn.Unflatten(-1, (32, 14, 14)),
            nn.ConvTranspose2d(32, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(32, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 64, 2, stride=2), nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = Variable(x.cuda())

        feature = self.feature(x)
        x_hat = self.decoder(feature)
        x_hat_feats = self.feature(x_hat)
        scores = self.classifier(feature)
        x_hat_scores = self.classifier(x_hat_feats)
        # u, s, v = torch.svd(feature.t())
        # BSR = torch.sum(torch.pow(s, 2))
        return scores, x_hat, x_hat_scores, feature

    def forward_loss(self, x, y):
        y = Variable(y.cuda())
        scores, x_hat, x_hat_scores, _ = self.forward(x)
        loss_c = self.loss_fn(scores, y)
        loss_c_hat = self.loss_fn(x_hat_scores, y)

        loss = loss_c + self.lamda1 * torch.mean((x.cuda() - x_hat) ** 2) + self.lamda2 * loss_c_hat
        _, predicted = torch.max(scores.data, 1)
        correct = predicted.eq(y.data).cpu().sum()
        self.top1.update(correct.item() * 100 / (y.size(0) + 0.0), y.size(0))
        return loss

    def train_autoencoder(self, epoch, train_loader, optimizer):
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



if __name__ == '__main__':
    params = parse_args('train')
    model = ResNet10 = backbone.ResNet10
    # model = NSAE_model(model, params.num_classes, lamda=params.lamda).to('cuda')
    model = NSAE_model(model, params.num_classes, lamda1=params.lamda1, lamda2=params.lamda2).to('cuda')
    # decoder = Decoder().to('cuda')

    out = model(torch.randn(10,3,224,224).to('cuda'))
    print(f"out[0]: {out[0].shape}, out[1]: {out[1].shape}")
    # out2 = decoder(out[1])
    # print(f"out2: {out2.shape}")
    pdb.set_trace()