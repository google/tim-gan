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
import torch.nn.functional as F

class Adaptive_Routing(torch.nn.Module):
  def __init__(self, n_res, dim, text_dim, temperature=1.):
    super(AdaIN, self).__init__()
    #layer1
    self.res = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp = networks.MLP(text_dim, self.get_num_adain_params(self.res), text_dim, 3, norm='none', activ='relu')
    self.res2 = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp2 = networks.MLP(text_dim, self.get_num_adain_params(self.res2), text_dim, 3, norm='none', activ='relu')
    self.res3 = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp3 = networks.MLP(text_dim, self.get_num_adain_params(self.res3), text_dim, 3, norm='none', activ='relu')
    #layer2
    self.res_2 = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp_2 = networks.MLP(text_dim, self.get_num_adain_params(self.res_2), text_dim, 3, norm='none', activ='relu')
    self.res2_2 = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp2_2 = networks.MLP(text_dim, self.get_num_adain_params(self.res2_2), text_dim, 3, norm='none', activ='relu')
    self.res3_2 = networks.ResBlocks(n_res, dim, 'adain', 'relu', pad_type='reflect')
    self.mlp3_2 = networks.MLP(text_dim, self.get_num_adain_params(self.res3_2), text_dim, 3, norm='none', activ='relu')
    self.mlpencoder = networks.MLP(text_dim, 3*2, text_dim, 3, norm='none', activ='relu')
    #Temperature
    self.T = temperature


  def forward(self, img_feat, text_feat, temperature_rate):
    gates = self.mlpencoder(text_feat.detach())
    gates = gates.view(-1,2,3)
    gates = F.gumbel_softmax(gates, self.T, hard=False)
    gates = gates.view(-1,2,3,1,1,1)

    adain_params = self.mlp(text_feat)
    self.assign_adain_params(adain_params, self.res)
    out1 = self.res(img_feat)

    adain_params2 = self.mlp2(text_feat)
    self.assign_adain_params(adain_params2, self.res2)
    out2 = self.res2(img_feat)

    adain_params3 = self.mlp3(text_feat)
    self.assign_adain_params(adain_params3, self.res3)
    out3 = self.res3(img_feat)

    feat = out1*gates[:,0,0]+out2*gates[:,0,1]+out3*gates[:,0,2]

    adain_params_2 = self.mlp_2(text_feat)
    self.assign_adain_params(adain_params_2, self.res_2)
    out1 = self.res_2(feat)

    adain_params2_2 = self.mlp2_2(text_feat)
    self.assign_adain_params(adain_params2_2, self.res2_2)
    out2 = self.res2_2(feat)

    adain_params3_2 = self.mlp3_2(text_feat)
    self.assign_adain_params(adain_params3_2, self.res3_2)
    out3 = self.res3_2(feat)
    return out1*gates[:,1,0]+out2*gates[:,1,1]+out3*gates[:,1,2], gates

  def assign_adain_params(self, adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
      if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
        mean = adain_params[:, :m.num_features]
        std = adain_params[:, m.num_features:2*m.num_features]
        m.bias = mean.contiguous().view(-1)
        m.weight = std.contiguous().view(-1)
        if adain_params.size(1) > 2*m.num_features:
          adain_params = adain_params[:, 2*m.num_features:]

  def get_num_adain_params(self, model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
      if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
        num_adain_params += 2*m.num_features
    return num_adain_params
