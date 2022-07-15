import torch
from torch.nn import functional as F
from new_networks.conformer import build_model
import numpy as np
import os
import cv2
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
writer = SummaryWriter('log/run' + time.strftime("%d-%m"))
import torch.nn as nn
import argparse
import os.path as osp
import os
size_coarse = (20, 20)


class Solver(object):
    def __init__(self, train_loader, test_loader,val_loader config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader=val_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        #self.build_model()
        self.net = build_model(self.config.network, self.config.arch)
        #self.net.eval()
        if config.mode == 'test':
            print('Loading pre-trained model for testing from %s...' % self.config.model)
            self.net.load_state_dict(torch.load(self.config.model, map_location=torch.device('cpu')))
        if config.mode == 'train':
            if self.config.load == '':
                print("Loading pre-trained imagenet weights for fine tuning")
                self.net.JLModule.load_pretrained_model(self.config.pretrained_model
                                                        if isinstance(self.config.pretrained_model, str)
                                                        else self.config.pretrained_model[self.config.network])
                # load pretrained backbone
            else:
                print('Loading pretrained model to resume training')
                self.net.load_state_dict(torch.load(self.config.load))  # load pretrained model
        
        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        #self.print_network(self.net, 'Conformer based SOD Structure')

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params_t = 0
        num_params=0
        for p in model.parameters():
            if p.requires_grad:
                num_params_t += p.numel()
            else:
                num_params += p.numel()
        print(name)
        print(model)
        print("The number of trainable parameters: {}".format(num_params_t))
        print("The number of parameters: {}".format(num_params))

    # build the network
    '''def build_model(self):
        self.net = build_model(self.config.network, self.config.arch)

        if self.config.cuda:
            self.net = self.net.cuda()

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.wd)

        self.print_network(self.net, 'JL-DCF Structure')'''

    def test(self):
        print('Testing...')
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size, depth = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size']), \
                                           data_batch['depth']
            with torch.no_grad():
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    images = images.to(device)
                    depth = depth.to(device)

                input = torch.cat((images, depth), dim=0)
                preds, pred_coarse = self.net(input)
                #print(preds.shape)
                preds = F.interpolate(preds, tuple(im_size), mode='bilinear', align_corners=True)
                pred = np.squeeze(torch.sigmoid(preds)).cpu().data.numpy()

                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
                multi_fuse = 255 * pred
                filename = os.path.join(self.config.test_folder, name[:-4] + '_convtran.png')
                cv2.imwrite(filename, multi_fuse)
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))
        print('Test Done!')
    
    def val(self):
        #print('validating model..')
        
        self.net.eval()
        avg_mae = 0.0
        r_sal_loss = 0
        rv_loss_item=0
        with torch.no_grad():
            for i, data_batch in enumerate(self.val_loader):
                sal_image, sal_depth, sal_label,name,im_size= data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                    'sal_label'], data_batch['name'][0], np.asarray(data_batch['size'])
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label= sal_image.to(device), sal_depth.to(device), sal_label.to(device)
                sal_label_coarse = torch.cat((sal_label, sal_label), dim=0)
                
                sal_input = torch.cat((sal_image, sal_depth), dim=0)
                sal_final, sal_coarse = self.net(sal_input)
                
                sal_loss_coarse = F.binary_cross_entropy_with_logits(sal_coarse, sal_label_coarse, reduction='sum')
                sal_final_loss = F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                
                sal_loss_fuse = sal_final_loss+ 256* sal_loss_coarse
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                rv_loss_item+=sal_loss.item() * sal_input.size(0)
 
                #mean absolute error
                avg_mae, img_num = 0.0, 0.0
                mea = torch.abs(sal_final - sal_label).mean()
                if mea == mea: # for Nan
                    avg_mae += mea
                    img_num += 1.0
                avg_mae /= img_num
                
                # fmeasure
                beta2 = 0.3
                avg_f, img_num = 0.0, 0.0
                prec, recall = Eval_pr(sal_final, sal_label, 255)
                f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
                f_score[f_score != f_score] = 0 # for Nan
                avg_f += f_score
                img_num += 1.0
                score = avg_f / img_num
            
        

                    
        self.net.train()
        return avg_mae.item() ,score.max().item(), rv_loss_item, r_sal_loss
    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        loss_vals=  []
        self.optimizer.zero_grad()
        for epoch in range(self.config.epoch):
            r_sal_loss = 0
            r_sal_loss_item=0
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_depth, sal_label = data_batch['sal_image'], data_batch['sal_depth'], data_batch[
                    'sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                if self.config.cuda:
                    device = torch.device(self.config.device_id)
                    sal_image, sal_depth, sal_label = sal_image.to(device), sal_depth.to(device), sal_label.to(device)

               
                sal_label_coarse = torch.cat((sal_label, sal_label), dim=0)
                
                sal_input = torch.cat((sal_image, sal_depth), dim=0)
                sal_final, sal_coarse = self.net(sal_input)
                
                sal_loss_coarse = F.binary_cross_entropy_with_logits(sal_coarse, sal_label_coarse, reduction='sum')
                sal_final_loss = F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                
                sal_loss_fuse = sal_final_loss+ 256* sal_loss_coarse
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data
                r_sal_loss_item+=sal_loss.item() * sal_input.size(0)
                sal_loss.backward()

                # accumulate gradients as done in DSS
                aveGrad += 1
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if (i + 1) % (self.show_every // self.config.batch_size) == 0:
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f ' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss / (self.show_every / self.iter_size)))
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_coarse training loss', sal_loss_coarse.data,
                                      epoch * len(self.train_loader.dataset) + i)

                    r_sal_loss = 0
                    res = sal_coarse[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('sal_coarse', torch.tensor(res), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)
                    
                    fsal = sal_final[0].clone()
                    fsal = fsal.sigmoid().data.cpu().numpy().squeeze()
                    fsal = (fsal - fsal.min()) / (fsal.max() - fsal.min() + 1e-8)
                    writer.add_image('sal_final', torch.tensor(fsal), i, dataformats='HW')
                    grid_image = make_grid(sal_label[0].clone().cpu().data, 1, normalize=True)


            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            loss_vals.append(train_loss)
            mae,fMeasure,loss_val_r,loss_val= self.val()
            print('Epoch:[%2d/%2d] | Train Loss : %.3f' % (epoch, self.config.epoch,train_loss))
            print('Epoch:[%2d/%2d] | Validation Loss : %.3f | mae : %.3f|fMeasure: %.3f' % (epoch, self.config.epoch,loss_val_r/len(self.val_loader.dataset),mae,fMeasure))
            writer.add_scalar('validation loss', loss_val ,epoch * len(self.val_loader.dataset) + i)
        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
        


def Eval_pr(y_pred, y, num):
    if self.config.cuda:
        device = torch.device(self.config.device_id)
        prec, recall = torch.zeros(num).to(device), torch.zeros(num).to(device)
        thlist = torch.linspace(0, 1 - 1e-10, num).to(device)
    else:
        prec, recall = torch.zeros(num), torch.zeros(num)
        thlist = torch.linspace(0, 1 - 1e-10, num)
    for i in range(num):
        y_temp = (y_pred >= thlist[i]).float()
        tp = (y_temp * y).sum()
        prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
    return prec, recall
    
   
