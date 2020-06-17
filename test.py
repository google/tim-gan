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
from collections import OrderedDict
import torch
from options.test_options import TestOptions
from dataset.aligned_dataset import AlignedDataset
from models.tim_gan import LocalGAN
from util import util
from util import html
from util import visualizer
import numpy as np

def main():
  # parse options
  parser = TestOptions()
  opts = parser.parse()

  # data loader
  print('\n--- config dataset ---')
  dataset = AlignedDataset(opts)
  loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

  # model
  print('\n--- create model ---')
  model = LocalGAN(opts)
  model.cuda()
  model.load(os.path.join(opts.output_dir, 'model', opts.name, '%s.pth' % (opts.which_epoch)))
  model.eval()

  # create website
  web_dir = os.path.join(opts.result_dir, '%s_%s_%s' % (opts.name, opts.phase, opts.which_epoch))
  webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opts.name, opts.phase, opts.which_epoch))

  # training
  print('\n--- testing ---')
  gate_out = []
  #tfeat = []
  alltexts = []
  texttokens = []
  textattn1 = []
  textattn2 = []
  for it, (src_img, text, tgt_img) in enumerate(loader):
   
    # forward
    with torch.no_grad():   
      model.set_input(src_img, text, tgt_img)
      model.forward()
      
    # print
    if (it + 1) % (len(loader) // 10) == 0:
      print('Iteration {}/{}'.format(it + 1, len(loader)))
    
    # save result  
    visuals = OrderedDict([('input', util.tensor2im(model.real_A[0].detach())),
                           ('output'+': '+text[0], util.tensor2im(model.fake_B[0].detach())),
                           ('real_image', util.tensor2im(model.real_B[0].detach())),
                           ('attn_mask', util.tensor2im(model.attn_dis[0].detach(), normalize=False)),
                           ('gt_attn_mask', util.tensor2im(model.gt_attn_dis[0].detach(), normalize=False))])
    img_path = [str(it)]
    gate_out.append(model.gates[0].detach().cpu().numpy().reshape(2*3))
    textattn1.append(model.text1[1][0].detach().cpu().numpy().reshape(-1))
    textattn2.append(model.text2[1][0].detach().cpu().numpy().reshape(-1))
    alltexts.append(text[0])
    texttokens.append(model.text_tokens[0])
    visualizer.save_images(webpage, visuals, img_path)

  webpage.save()
  np.savez(os.path.join(web_dir,'gate.npz'),gate=np.array(gate_out),text=np.array(alltexts),token=np.array(texttokens),attn1=np.array(textattn1),attn2=np.array(textattn2))
  return

if __name__ == '__main__':
  main()
