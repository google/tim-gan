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
from os.path import join
import numpy as np
import os
import torch.utils.data as data
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms.functional as F
from PIL import Image
import random
from dataset import util

class AlignedDataset(data.Dataset):

  def __init__(self, opts):
    self.dataroot = opts.dataroot
    self.train = (opts.phase == 'train')
    self.resize_ratio = opts.resize_ratio
    self.crop_size = opts.crop_size

    # image names
    self.images_A = util.image_file(os.listdir(join(self.dataroot, opts.phase + '_A')))
    self.images_A.sort()
    self.images_A = [join(self.dataroot, opts.phase + '_A', name) for name in self.images_A]
    self.images_B = util.image_file(os.listdir(join(self.dataroot, opts.phase + '_B')))
    self.images_B.sort()
    self.images_B = [join(self.dataroot, opts.phase + '_B', name) for name in self.images_B]

    # text
    self.texts = np.load(os.path.join(opts.dataroot,'%s_text.npy'%opts.phase))

    # image transformation
    self.input_dim = opts.input_dim
    self.flip = opts.flip
    transforms = [ToTensor()]
    transforms.append(Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    self.transforms = Compose(transforms)
    
    self.dataset_size = len(self.images_A)
    print('{} aligned dataset size {}'.format(opts.phase, self.dataset_size))
    return

  def load_image(self, filename, flip, resize_size, crop):
    img = Image.open(filename).convert('RGB')
    
    # flip
    if flip == 1:
      img = img.transpose(Image.FLIP_LEFT_RIGHT)

    # resize
    img = F.resize(img, resize_size, Image.BICUBIC)

    # crop
    if self.train:
      img = F.crop(img, crop[0], crop[1], crop[2], crop[3])
    else:
      img = F.center_crop(img, self.crop_size)

    # transform
    img = self.transforms(img)

    # dimension stuff
    if self.input_dim == 1:
      img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
      img = img.unsqueeze(0)
    
    return img

  def __getitem__(self, index):
    
    # image augmentation
    flip = random.randint(0, 1) if self.flip and self.train else 0
    resize_ratio = random.uniform(1, self.resize_ratio) if self.train else 1
    resize_size = (int(self.crop_size[0]*resize_ratio), int(self.crop_size[1]*resize_ratio))
    crop = util.get_crop_params(resize_size, self.crop_size)

    # images
    src_img = self.load_image(self.images_A[index], flip, resize_size, crop)
    tgt_img = self.load_image(self.images_B[index], flip, resize_size, crop)

    # text
    A_path = self.images_A[index]
    if 'Clevr' in self.dataroot:
      text_idx = int(A_path[-15:-9]) - 1
    else:
      text_idx = int(A_path[A_path.rfind('/')+1:A_path.rfind('.')])
    text = self.texts[text_idx]

    return src_img, text, tgt_img

  def __len__(self):
    return self.dataset_size
