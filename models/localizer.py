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

import torch
from third_party import networks

class LocalizerAttn(torch.nn.Module):
  def __init__(self, img_dim, text_dim, n_res=2):
    super(LocalizerAttn, self).__init__()

    self.text_fc = torch.nn.Sequential(
                   torch.nn.BatchNorm1d(text_dim),
                   torch.nn.Linear(text_dim, img_dim + 2),
                   torch.nn.ReLU(inplace=True),
                   torch.nn.Linear(img_dim + 2, img_dim + 2))
    self.img_conv = torch.nn.Sequential(
                   torch.nn.BatchNorm2d(img_dim + 2),
                   torch.nn.ReflectionPad2d(1),
                   torch.nn.Conv2d(img_dim + 2, img_dim + 2, (3, 3)),
                   torch.nn.ReLU(inplace=True),
                   torch.nn.ReflectionPad2d(1),
                   torch.nn.Conv2d(img_dim + 2, img_dim + 2, (3, 3)))
    conv = [networks.ResBlocks(n_res, (img_dim + 2)*1, 'in', 'relu', pad_type='reflect'),
            torch.nn.Conv2d((img_dim + 2)*1, 1, (1, 1))]
    self.conv = torch.nn.Sequential(*conv)
    self.apply(networks.weights_init)

  def get_coordinate(self, img_feat, batch_size, h, w):
    device = img_feat.device
    x_coor = torch.linspace(-1, 1, w, device=device).unsqueeze(dim=0).expand(h, w).view(1, 1, h, w)
    x_coor = torch.cat([x_coor for _ in range(batch_size)], dim=0)
    y_coor = torch.linspace(-1, 1, h, device=device).unsqueeze(dim=1).expand(h, w).view(1, 1, h, w)
    y_coor = torch.cat([y_coor for _ in range(batch_size)], dim=0)
    return torch.cat([x_coor, y_coor], dim=1)

  def forward(self, img_feat, text_feat):
    batch_size = img_feat.size(0)
    h = img_feat.size(2)
    w = img_feat.size(3)
     
    # expand text_feat
    text_feat = self.text_fc(text_feat).view(batch_size, -1, 1, 1)
    text_feat = text_feat.expand(batch_size, text_feat.size(1), h, w)
    
    # get coordinate feat
    with torch.no_grad():
      coor_feat = self.get_coordinate(img_feat, batch_size, h, w)
    
    # image feat
    img_feat = self.img_conv(torch.cat([img_feat, coor_feat], dim=1))
    
    inp = text_feat*img_feat

    # get attention mask
    attn = torch.sigmoid(self.conv(inp))
    return attn
