from scipy.linalg import sqrtm
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch
import torch.optim
import os
import pdb
# import configs
from network import NSAE_model
import configs
from io_utils import model_dict, parse_args
from datasets import miniImageNet_few_shot
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors


def train(base_loader, model, optimization, start_epoch, stop_epoch, params):


    out_pre = '%s_%s' % (params.model, params.method)
    model.train()
    for epoch in range(start_epoch, stop_epoch):
        model.train_autoencoder(epoch, base_loader, optimizer)

        if (epoch % params.save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(params.checkpoint_dir, '%s_%d.tar' % (out_pre, epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    outfile_final = os.path.join(params.checkpoint_dir, '%s.tar' % out_pre)
    torch.save({'epoch': stop_epoch - 1, 'state': model.state_dict()}, outfile_final)
        
    return model

def finetune(model, optimizer, novel_loader, finetune_epochs_recon, finetune_epochs, n_way, n_support, n_query):
    model.train()

    acc_all_ori = []
    acc_all_lp = []

    # model.finetune(novel_loader, n_way, n_support, n_query)
    for i, (x, y) in enumerate(novel_loader):
        # n_query = x.size(1) - n_support
        x = x.cuda()
        x_var = Variable(x)
    
        batch_size = 4
        support_size = n_way * n_support

        y_a_i = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).cuda()

        x_b_i = x_var[:, n_support:, :, :, :].contiguous().view(n_way * n_query, *x.size()[2:])
        x_a_i = x_var[:, :n_support, :, :, :].contiguous().view(n_way * n_support, *x.size()[2:])
        for epoch in range(finetune_epochs_recon):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
               
                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id] 
                scores, x_hat, x_hat_scores, _ = model(z_batch)
                loss = torch.mean((z_batch.cuda() - x_hat) ** 2)
                print('finetune recon loss:', loss.item())
                loss.backward()

                optimizer.step()

        optimizer = torch.optim.SGD(list(model.parameters()), lr=0.01, momentum=0.9, weight_decay=1e-3)
        cross_ent_loss = nn.CrossEntropyLoss()
        for epoch in range(finetune_epochs):
            rand_id = np.random.permutation(support_size)

            for j in range(0, support_size, batch_size):
                selected_id = torch.from_numpy( rand_id[j: min(j+batch_size, support_size)]).cuda()
               
                z_batch = x_a_i[selected_id]
                y_batch = y_a_i[selected_id] 
                scores, x_hat, x_hat_scores, _ = model(z_batch)
                # loss = torch.mean((z_batch.cuda() - x_hat) ** 2)
                loss = cross_ent_loss(scores, y_batch)
                print('finetune loss:', loss.item())
                loss.backward()

                optimizer.step()

        model.eval()

        with torch.no_grad():
            scores, x_hat, x_hat_scores, output = model(x_b_i)
            # output = pretrained_model(x_b_i)
            # scores = classifier(output)
            x_lp = output.cpu().numpy()
            y_lp = F.softmax(scores, 1).cpu().numpy()
            topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
            topk_ind_ori = topk_labels.cpu().numpy()
       
        y_query = np.repeat(range(n_way), n_query)

        neigh = NearestNeighbors(params.k_lp)
        neigh.fit(x_lp)
        d_lp, idx_lp = neigh.kneighbors(x_lp)
        d_lp = np.power(d_lp, 2)
        sigma2_lp = np.mean(d_lp)

        n_lp = len(y_query)
        del_n = int(n_lp * (1.0 - params.delta))
        for i in range(n_way):
            yi = y_lp[:, i]
            top_del_idx = np.argsort(yi)[0:del_n]
            y_lp[top_del_idx, i] = 0

        w_lp = np.zeros((n_lp, n_lp))
        for i in range(n_lp):
            for j in range(params.k_lp):
                xj = idx_lp[i, j]
                w_lp[i, xj] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
                w_lp[xj, i] = np.exp(-d_lp[i, j] / (2 * sigma2_lp))
        q_lp = np.diag(np.sum(w_lp, axis=1))
        q2_lp = sqrtm(q_lp)
        q2_lp = np.linalg.inv(q2_lp)
        L_lp = np.matmul(np.matmul(q2_lp, w_lp), q2_lp)
        a_lp = np.eye(n_lp) - params.alpha * L_lp
        a_lp = np.linalg.inv(a_lp)
        ynew_lp = np.matmul(a_lp, y_lp)

        count_this = len(y_query)

        top1_correct_ori = np.sum(topk_ind_ori[:, 0] == y_query)
        correct_ori = float(top1_correct_ori)
        print('BSR: %f' % (correct_ori / count_this * 100))
        acc_all_ori.append((correct_ori / count_this * 100))

        topk_ind_lp = np.argmax(ynew_lp, 1)
        top1_correct_lp = np.sum(topk_ind_lp == y_query)
        correct_lp = float(top1_correct_lp)
        print('BSR+LP: %f' % (correct_lp / count_this * 100))
        acc_all_lp.append((correct_lp / count_this * 100))
        ###############################################################################################

    acc_all_ori  = np.asarray(acc_all_ori)
    acc_mean_ori = np.mean(acc_all_ori)
    acc_std_ori  = np.std(acc_all_ori)
    print('BSR: %d Test Acc = %4.2f%% +- %4.2f%%' %
          (finetune_epochs,  acc_mean_ori, 1.96 * acc_std_ori / np.sqrt(finetune_epochs)))

    acc_all_lp = np.asarray(acc_all_lp)
    acc_mean_lp = np.mean(acc_all_lp)
    acc_std_lp = np.std(acc_all_lp)
    print('BSR+LP: %d Test Acc = %4.2f%% +- %4.2f%%' %
          (finetune_epochs, acc_mean_lp, 1.96 * acc_std_lp / np.sqrt(finetune_epochs)))




if __name__=='__main__':
    np.random.seed(10)
    params = parse_args('train')
    finetune_params = parse_args('finetune')
    finetune_epochs_recon = 1 #30
    finetune_epochs = 1 #200

    model = NSAE_model(model_dict[params.model], params.num_classes, lamda1=params.lamda1, lamda2=params.lamda2)
    model = model.cuda()

    image_size = 224

    optimization = 'SGD'
    if optimization == 'Adam':
        optimizer = torch.optim.Adam(model.parameters())
    elif optimization == 'SGD':
        optimizer = torch.optim.SGD(list(model.parameters()), lr=0.001, momentum=0.9, weight_decay=0.0005)
    else:
       raise ValueError('Unknown optimization, please define by yourself')

    datamgr = miniImageNet_few_shot.SimpleDataManager(image_size, batch_size=16)
    base_loader = datamgr.get_data_loader(aug=params.train_aug)
    few_shot_params = dict(n_way=finetune_params.test_n_way, n_support=finetune_params.n_shot, n_query=15)
    datamgr = miniImageNet_few_shot.SetDataManager(image_size, n_eposide=1, **few_shot_params)
    novel_loader = datamgr.get_data_loader(aug=False)



    save_dir = configs.save_dir

    params.method = 'bsr'
    params.checkpoint_dir = '%s/checkpoints/%s_%s' % (save_dir, params.model, params.method)
    if params.train_aug:
        params.checkpoint_dir += '_aug'

    if not os.path.isdir(params.checkpoint_dir):
        os.makedirs(params.checkpoint_dir)

    start_epoch = params.start_epoch
    stop_epoch = params.stop_epoch

    # model = train(base_loader, model, optimizer, start_epoch, stop_epoch, params)

    model = finetune(model, optimizer, novel_loader, finetune_epochs_recon, finetune_epochs, **few_shot_params)