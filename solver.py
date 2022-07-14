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

size_coarse = (20, 20)


class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
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
                preds,sal_lde_conv,sal_lde_tran,sal_gde_conv,sal_gde_tran, pred_coarse = self.net(input)
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

                #sal_label_coarse = F.interpolate(sal_label, size_coarse, mode='bilinear', align_corners=True)
                #sal_label_coarse = torch.cat((sal_label_coarse, sal_label_coarse), dim=0)
                sal_label_coarse = torch.cat((sal_label, sal_label), dim=0)
                
                sal_input = torch.cat((sal_image, sal_depth), dim=0)
                sal_final,sal_lde_conv,sal_lde_tran,sal_gde_conv,sal_gde_tran, sal_coarse = self.net(sal_input)
                sal_labels = torch.cat((sal_label, sal_label), dim=0)#need to be deleted
                sal_loss_coarse = F.binary_cross_entropy_with_logits(sal_coarse, sal_label_coarse, reduction='sum')
                sal_loss_lde_conv = F.binary_cross_entropy_with_logits(sal_lde_conv, sal_labels, reduction='sum')
                #print('sal_loss_lde_conv',sal_loss_lde_conv)
                sal_loss_lde_tran = F.binary_cross_entropy_with_logits(sal_lde_tran, sal_labels, reduction='sum')
                #print('sal_loss_lde_tran',sal_loss_lde_tran)
                sal_loss_gde_conv= F.binary_cross_entropy_with_logits(sal_gde_conv, sal_labels, reduction='sum')
                #print('sal_loss_gde_conv',sal_loss_gde_conv)
                sal_loss_gde_tran = F.binary_cross_entropy_with_logits(sal_gde_tran, sal_labels, reduction='sum')
                #print('sal_loss_gde_tran',sal_loss_gde_tran)
                sal_final_loss = F.binary_cross_entropy_with_logits(sal_final, sal_label, reduction='sum')
                sal_loss_fuse = sal_final_loss+sal_loss_lde_conv/4 + sal_loss_lde_tran/4 + sal_loss_gde_conv/4 + sal_loss_gde_tran/4 + sal_loss_coarse
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
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f ,lde_c : %.4f,lde_t: %.4f,gde_c: %.4f,gde_t: %.4f' % (
                        epoch, self.config.epoch, i + 1, iter_num, r_sal_loss / (self.show_every / self.iter_size),sal_loss_lde_conv.data,sal_loss_lde_tran.data,sal_loss_gde_conv.data,sal_loss_gde_tran.data))
                    # print('Learning rate: ' + str(self.lr))
                    writer.add_scalar('training loss', r_sal_loss / (self.show_every / self.iter_size),
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_coarse training loss', sal_loss_coarse,
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar(' sal_loss_lde_conv training loss', sal_loss_lde_conv ,
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_lde_tran training loss', sal_loss_lde_tran,
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar('sal_loss_gde_conv training loss', sal_loss_gde_conv,
                                      epoch * len(self.train_loader.dataset) + i)
                    writer.add_scalar(' sal_loss_gde_trantraining loss', sal_loss_gde_tran,
                                      epoch * len(self.train_loader.dataset) + i)
                    r_sal_loss = 0
                    res = sal_coarse[0].clone()
                    res = res.sigmoid().data.cpu().numpy().squeeze()
                    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                    writer.add_image('sal_coarse', torch.tensor(res), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)
                    res_sal_lde_conv = sal_lde_conv[0].clone()
                    res_sal_lde_conv = res_sal_lde_conv.sigmoid().data.cpu().numpy().squeeze()
                    res_sal_lde_conv = (res_sal_lde_conv - res_sal_lde_conv.min()) / (res_sal_lde_conv.max() - res_sal_lde_conv.min() + 1e-8)
                    writer.add_image('sal_lde_conv', torch.tensor(res_sal_lde_conv), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)
                    res_sal_lde_tran = sal_lde_tran[0].clone()
                    res_sal_lde_tran = res_sal_lde_tran.sigmoid().data.cpu().numpy().squeeze()
                    res_sal_lde_tran = (res_sal_lde_tran - res_sal_lde_tran.min()) / (res_sal_lde_tran.max() - res_sal_lde_tran.min() + 1e-8)
                    writer.add_image('sal_lde_tran', torch.tensor(res_sal_lde_tran), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)
                    res_sal_gde_conv = sal_gde_conv[0].clone()
                    res_sal_gde_conv= res_sal_gde_conv.sigmoid().data.cpu().numpy().squeeze()
                    res_sal_gde_conv = (res_sal_gde_conv - res_sal_gde_conv.min()) / (res_sal_gde_conv.max() - res_sal_gde_conv.min() + 1e-8)
                    writer.add_image('sal_gde_conv', torch.tensor(res_sal_gde_conv), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)
                    res_sal_gde_tran = sal_gde_tran[0].clone()
                    res_sal_gde_tran= res_sal_gde_tran.sigmoid().data.cpu().numpy().squeeze()
                    res_sal_gde_tran = (res_sal_gde_tran - res_sal_gde_tran.min()) / (res_sal_gde_tran.max() - res_sal_gde_tran.min() + 1e-8)
                    writer.add_image('sal_gde_tran', torch.tensor(res_sal_gde_tran), i, dataformats='HW')
                    grid_image = make_grid(sal_label_coarse[0].clone().cpu().data, 1, normalize=True)

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/epoch_%d.pth' % (self.config.save_folder, epoch + 1))
            train_loss=r_sal_loss_item/len(self.train_loader.dataset)
            loss_vals.append(train_loss)

        # save model
        torch.save(self.net.state_dict(), '%s/final.pth' % self.config.save_folder)
