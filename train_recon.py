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
from options.train_options import TrainOptions
from dataset.recon_dataset import ReconDataset
from models.tim_gan import LocalGAN
from tensorboardX import SummaryWriter
import torchvision

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()
  opts.trainE = True
  opts.no_tensorboard = True
  if not os.path.exists(os.path.join(opts.output_dir, 'model', opts.name + '_recon')):
    os.makedirs(os.path.join(opts.output_dir, 'model', opts.name + '_recon'))

  # data loader
  print('\n--- config dataset ---')
  dataset = ReconDataset(opts)
  loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

  # model
  print('\n--- create model ---')
  model = LocalGAN(opts)
  model.cuda()

  # tensorboard
  tf_board = SummaryWriter(logdir=os.path.join(opts.output_dir, 'tfboard', opts.name + '_recon'))

  # training
  print('\n--- training ---')
  total_it = 0
  for ep in range(opts.n_ep):
    for it, img in enumerate(loader):
      
      # on GPU
      img = img.cuda()

      # forward
      model.forward_recon(img)
      
      # update
      model.update_D()
      model.update_G_recon()

      # display
      if (total_it + 1) % 10 == 0:
        tf_board.add_scalar('loss_G_L1', model.loss_G_L1.item(), total_it)
        tf_board.add_scalar('loss_G_GAN', model.loss_G_GAN.item(), total_it)
        tf_board.add_scalar('loss_D', model.loss_D.item(), total_it)
      if (total_it + 1) % 100 == 0:
        img_dis = torch.cat([img for img in model.img_dis_recon], dim=0)
        img_dis = torchvision.utils.make_grid(img_dis, nrow=2) / 2 + 0.5
        tf_board.add_image('Image', img_dis, total_it)
      
      # print
      if (it + 1) % (len(loader) // 5) == 0:
        print('Iteration {}, EP[{}/{}]'.format(total_it + 1, ep + 1, opts.n_ep))
      total_it += 1
      
    # write model file
    if (ep + 1) % 10 == 0:
      model.save(os.path.join(opts.output_dir, 'model', opts.name + '_recon', '{}.pth'.format(ep + 1)), ep, total_it)

  return

if __name__ == '__main__':
  main()


