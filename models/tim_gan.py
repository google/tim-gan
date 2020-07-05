# Copyright 2020 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import torch
from third_party import networks
from models import text_models
import models.operator as op
from models.localizer import LocalizerAttn
import functools
import torchvision
from tensorboardX import SummaryWriter

def pairwise_dis(x,y):
  x = x.view(x.shape[0],-1)
  y = y.view(y.shape[0],-1)
  return torch.abs(x-y).sum(dim=1)

class LocalGAN(torch.nn.Module):
  def __init__(self, opts):
    super(LocalGAN, self).__init__()
    self.isTrain = (opts.phase == 'train')
    self.gpu_ids = opts.gpu_ids
    self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
    img_dim = opts.input_dim
    n_downsampling = opts.n_downsampling
    if self.isTrain:
      self.w_gate = opts.w_gate

    # image encoder
    self.img_E = networks.ContentEncoder(n_downsample=n_downsampling,
                                         n_res=0,
                                         input_dim=img_dim,
                                         dim=64,
                                         norm='in', activ='relu', pad_type='reflect')

    # text encoder
    self.text_E = text_models.BertTextEncoder(pretrained=True,img_dim=self.img_E.output_dim)

    # localizer (predict attention mask on image)
    self.localizer = LocalizerAttn(img_dim=self.img_E.output_dim, text_dim=512)

    # operator
    if opts.operator == 'adaroute':
      self.operator = op.Adaptive_Routing(n_res=opts.num_adablock, dim=self.img_E.output_dim, text_dim=512, temperature=opts.temperature)
    else:
      raise Exception('no such operator %s' % (opts.operator))    

    # image decoder
    self.G = networks.Decoder(n_upsample=n_downsampling,
                              n_res=2,
                              dim=self.img_E.output_dim,
                              output_dim=img_dim, activ='relu', pad_type='reflect')

    # discriminator
    if self.isTrain:
      self.D = networks.D_NLayers(img_dim, ndf=64, n_layers=3, 
               norm_layer=functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False)) 

    # optimizer
    if self.isTrain:
      self.criterionGAN = networks.GANLoss(gan_mode=opts.gan_mode)
      self.criterionL1 = torch.nn.L1Loss()
      params = list(self.localizer.parameters()) + list(self.operator.parameters()) + list(self.G.parameters())
      if opts.pretrain == '':
        params += list(self.img_E.parameters())
      else:
        self.load_pretrain(os.path.join(opts.output_dir, 'model', opts.pretrain, '30.pth'))
        self.set_requires_grad(self.img_E, False)
        self.img_E.eval()
      self.opt_G = torch.optim.Adam([{'params': self.text_E.parameters(), 'lr': opts.lr/10.},
                                     {'params': params}], lr=opts.lr, betas=(0.5, 0.999))
      self.opt_D = torch.optim.Adam(self.D.parameters(), lr=opts.lr, betas=(0.5, 0.999))

    # tf board
    if self.isTrain and not opts.no_tensorboard:
      self.tf_board = SummaryWriter(logdir=os.path.join(opts.output_dir, 'tfboard', opts.name))

  def set_input(self, src_img, text, tgt_img):
    self.real_A = src_img.cuda()
    self.text = text
    self.real_B = tgt_img.cuda()
   
  def forward_recon(self, img):
    self.real_B = img
    self.fake_B = self.G(self.img_E(img))
    self.img_dis_recon = [self.real_B[:1].detach().cpu(), self.fake_B[:1].detach().cpu()]
    return

  def forward(self, use_gt_attn_rate=0., temperature_rate=0.):
    batch_size = self.real_A.size(0)    

    # image feature
    feat = self.img_E(torch.cat([self.real_A, self.real_B], dim=0))
    real_A_feat, real_B_feat = torch.split(feat, batch_size, dim=0)
    self.real_B_feat = real_B_feat.detach()

    # text feature
    img1d = torch.mean(real_A_feat, dim=(2,3))
    self.text1, self.text2, self.text_tokens, rawtext = self.text_E.extract_text_feature(self.text, img1d)

    # ground-truth attention mask
    with torch.no_grad():
      diff = torch.mean(torch.abs(real_A_feat - real_B_feat), dim=1).view(batch_size, -1)
      diff = diff - torch.min(diff, dim=1, keepdim=True)[0].expand_as(diff)
      self.attn_gt = (diff/(torch.max(diff, dim=1, keepdim=True)[0] + 1e-5)).view(batch_size, 1, real_A_feat.size(2), real_A_feat.size(3))

    # attention mask
    #self.attn = self.attn_gt # for sanity check
    self.attn = self.localizer(real_A_feat, self.text1[0])

    # schedule samping
    if self.isTrain and use_gt_attn_rate > 0:
      use_gt_attn = torch.rand(batch_size, 1, 1, 1, device=self.attn_gt.device)
      use_gt_attn = torch.lt(use_gt_attn, use_gt_attn_rate).float().expand_as(self.attn_gt)
      attn = use_gt_attn*self.attn_gt + (1 - use_gt_attn)*self.attn
    else:
      attn = self.attn

    # edit feature
    edit_feat, self.gates = self.operator(real_A_feat, self.text2[0], temperature_rate)

    # fuse original and edited feature
    self.fake_B_feat = torch.mul(edit_feat, attn) + torch.mul(real_A_feat, (1 - attn))

    # decode
    self.fake_B = self.G(self.fake_B_feat)

    # display
    if self.isTrain:
      self.img_dis = [self.real_A[:1].detach().cpu(), self.real_B[:1].detach().cpu(), self.fake_B[:1].detach().cpu()]
      self.attn_dis = [self.attn_gt[:1].detach().cpu(), self.attn[:1].detach().cpu()]
    else:
      self.attn_dis = torch.nn.functional.interpolate(self.attn, size=(self.real_A.size(2),self.real_A.size(3)), mode="bilinear")
      self.gt_attn_dis = torch.nn.functional.interpolate(self.attn_gt, size=(self.real_A.size(2),self.real_A.size(3)), mode="bilinear")
    return

  def backward_D(self, netD, real, fake):
    # real
    pred_real = netD(real)
    loss_D_real, _ = self.criterionGAN(pred_real, True)

    # fake
    pred_fake = netD(fake.detach())
    loss_D_fake, _ = self.criterionGAN(pred_fake, False)

    # loss
    loss_D = loss_D_fake + loss_D_real
    loss_D.backward()
    return loss_D, [loss_D_fake, loss_D_real]

  def backward_G_GAN(self, fake, netD=None, ll=0.0):
    if ll > 0.0:
      pred_fake = netD(fake)
      loss_G_GAN, _ = self.criterionGAN(pred_fake, True, loss_on_D=False)
    else:
      loss_G_GAN = 0
    return loss_G_GAN * ll

  def backward_G_recon(self):
    self.loss_G_GAN = self.backward_G_GAN(self.fake_B, self.D, 1.)
    self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
    self.loss_G = self.loss_G_GAN + self.loss_G_L1*10
    self.loss_G.backward(retain_graph=True)

  def backward_G(self):
    self.loss_G_GAN = self.backward_G_GAN(self.fake_B, self.D, 1.)
    self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)

    # attention loss
    self.loss_G_attn = self.criterionL1(self.attn, self.attn_gt)

    # gate divergence loss for visualization purpose, set W=0 in the bash script if you don't need it
    t2 = self.text2[0].detach()
    self.loss_disgate = -((pairwise_dis(self.gates,torch.cat((self.gates[1:],self.gates[:1]),dim=0)))/(pairwise_dis(t2,torch.cat((t2[1:],t2[:1]),dim=0))+1e-5)/3. +\
                        (pairwise_dis(self.gates,torch.cat((self.gates[2:],self.gates[:2]),dim=0)))/(pairwise_dis(t2,torch.cat((t2[2:],t2[:2]),dim=0))+1e-5)/3.+\
                        (pairwise_dis(self.gates,torch.cat((self.gates[3:],self.gates[:3]),dim=0)))/(pairwise_dis(t2,torch.cat((t2[3:],t2[:3]),dim=0))+1e-5)/3.)
    self.loss_disgate = self.loss_disgate.mean()

    # feat loss
    feat_dis = torch.mean(torch.abs(self.fake_B_feat - self.real_B_feat), dim=1, keepdim=True)
    weighted_feat_dis = torch.sum(feat_dis * self.attn_gt, dim=(1,2,3)) / (torch.sum(self.attn_gt, dim=(1,2,3)) + 1e-5)
    self.loss_G_feat = torch.mean(weighted_feat_dis)
    
    # summation
    self.loss_G = self.loss_G_GAN + self.loss_G_L1*10 + self.loss_G_feat*10 + self.loss_G_attn*100 + self.loss_disgate*self.w_gate
    self.loss_G.backward(retain_graph=True)

  def update_D(self):
    self.set_requires_grad(self.D, True)
    self.opt_D.zero_grad()
    self.loss_D, _ = self.backward_D(self.D, self.real_B, self.fake_B)
    self.opt_D.step()
  
  def update_G_recon(self):
    self.set_requires_grad(self.D, False)
    self.opt_G.zero_grad()
    self.backward_G_recon()
    self.opt_G.step()
    return
  
  def update_G(self):
    self.set_requires_grad(self.D, False)
    self.opt_G.zero_grad()
    self.backward_G()
    self.opt_G.step()
    return

  def set_requires_grad(self, nets, requires_grad=False):
    if not isinstance(nets, list):
      nets = [nets]
    for net in nets:
      if net is not None:
        for param in net.parameters():
          param.requires_grad = requires_grad

  def write_display(self, total_it):
    
    # write losses
    if (total_it + 1) % 10 == 0:
      members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and 'loss' in attr]
      for m in members:
        self.tf_board.add_scalar(m, getattr(self, m).item(), total_it)

    # write images & attention
    if (total_it + 1) % 100 == 0:
      img_dis = torch.cat([img for img in self.img_dis], dim=0)
      img_dis = torchvision.utils.make_grid(img_dis, nrow=max(1, img_dis.size(0))) / 2 + 0.5
      self.tf_board.add_image('Image [input/output/generated]', img_dis, total_it)
      
      attn_dis = torch.cat([torch.cat([img, img, img], dim=1) for img in self.attn_dis], dim=0)
      attn_dis = torchvision.utils.make_grid(attn_dis, nrow=max(1, attn_dis.size(0)))
      self.tf_board.add_image('Attention [ground-truth/generated]', attn_dis, total_it)

    return

  def save(self, filename, ep, total_it):
    print('--- save the model @ ep {} ---'.format(ep + 1))
    state = {
        'ep': ep,
        'total_it': total_it,
        'img_E': self.img_E.state_dict(),
        'text_E': self.text_E.state_dict(),
        'localizer': self.localizer.state_dict(),
        'operator': self.operator.state_dict(),
        'G': self.G.state_dict(),
        'D': self.D.state_dict(),
        'opt_G': self.opt_G.state_dict(),
        'opt_D': self.opt_D.state_dict()}
    torch.save(state, filename)
    return

  def load(self, filename):
    print('load the model file from %s' % (filename))
    ck = torch.load(filename, map_location=self.device)
    self.img_E.load_state_dict(ck['img_E'])
    self.text_E.load_state_dict(ck['text_E'])
    self.localizer.load_state_dict(ck['localizer'])
    self.operator.load_state_dict(ck['operator'])
    self.G.load_state_dict(ck['G'])
    if self.isTrain:
      print('load training related stuffs')
      self.D.load_state_dict(ck['D'])
      self.opt_G.load_state_dict(ck['opt_G'])
      self.opt_D.load_state_dict(ck['opt_D'])
  
  def load_pretrain(self, filename):
    print('load pre-trained model file from %s' % (filename))
    ck = torch.load(filename, map_location=self.device)
    self.img_E.load_state_dict(ck['img_E'])
    self.G.load_state_dict(ck['G'])
