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
from dataset.aligned_dataset import AlignedDataset
from models.tim_gan import LocalGAN

def main():
  # parse options
  parser = TrainOptions()
  opts = parser.parse()
  if not os.path.exists(os.path.join(opts.output_dir, 'model', opts.name)):
    os.makedirs(os.path.join(opts.output_dir, 'model', opts.name))

  # data loader
  print('\n--- config dataset ---')
  dataset = AlignedDataset(opts)
  loader = torch.utils.data.DataLoader(dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.nThreads)

  # model
  print('\n--- create model ---')
  model = LocalGAN(opts)
  model.cuda()

  # training
  print('\n--- training ---')
  total_it = 0
  for ep in range(opts.n_ep):
    for it, (src_img, text, tgt_img) in enumerate(loader):
      
      # forward
      temperature_rate = max(0, 1 - (ep + 1)/float(opts.n_ep))
      use_gt_attn_rate = max(0, 1 - ep/float(opts.n_ep))
      model.set_input(src_img, text, tgt_img)
      model.forward(use_gt_attn_rate=use_gt_attn_rate,temperature_rate=temperature_rate)
      
      # update
      model.update_D()
      model.update_G()

      # display
      model.write_display(total_it)
      
      # print
      if (it + 1) % (len(loader) // 5) == 0:
        print('Iteration {}, EP[{}/{}]'.format(total_it + 1, ep + 1, opts.n_ep))
      total_it += 1
      
    # write model file
    if (ep + 1) % 10 == 0:
      model.save(os.path.join(opts.output_dir, 'model', opts.name, '{}.pth'.format(ep + 1)), ep, total_it)

  return

if __name__ == '__main__':
  main()
