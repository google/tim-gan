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

"""Models for Text and Image Composition."""

import math
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig

class BertTextEncoder(torch.nn.Module):
  """Base class for image + text composition."""

  def __init__(self, pretrained=True, img_dim=256):
    super(BertTextEncoder, self).__init__()
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    self.pretrained = pretrained
    ### Define an attention module using concat and additive
    self.query1 = torch.nn.Linear(768+img_dim, 512)
    self.key1 = torch.nn.Linear(768, 512)
    self.value1 = torch.nn.Linear(768, 512)
    self.query2 = torch.nn.Linear(768+img_dim, 512)
    self.key2 = torch.nn.Linear(768, 512)
    self.value2 = torch.nn.Linear(768, 512)
    if not pretrained:
      config = BertConfig.from_pretrained('bert-base-cased') 
      config.hidden_size = 768
      config.num_attention_heads = 12
      self.textmodel = BertModel(config)
    else:
      self.textmodel = BertModel.from_pretrained('bert-base-cased')
      #self.downsample = torch.nn.Linear(768,512)

  def extract_text_feature(self, texts, img1d):
    x = []
    xlen = []
    mask = []
    attmask = []
    text_tokens = []
    for text in texts:
      t = '[CLS] '+text[0].upper()+text[1:]+' [SEP]'
      tokenized_text = self.tokenizer.tokenize(t)
      text_tokens.append(tokenized_text) 
      indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
      x.append(indexed_tokens)
      xlen.append(len(indexed_tokens))
    maxlen = max(xlen)
    for i in range(len(x)):
      mask.append([0]+[1]*(xlen[i]-2)+[0]*(maxlen-xlen[i]+1))
      attmask.append([1]*(xlen[i])+[0]*(maxlen-xlen[i]))
      x[i] = x[i]+[0]*(maxlen-xlen[i])
    x = torch.tensor(x)
    mask = torch.tensor(mask, dtype=torch.float).unsqueeze(2)
    attmask = torch.tensor(attmask, dtype=torch.float).unsqueeze(2)
    itexts = torch.autograd.Variable(x).cuda()
    mask = torch.autograd.Variable(mask).cuda()
    attmask = torch.autograd.Variable(attmask).cuda()
    out = self.textmodel(itexts, attention_mask = attmask)
    xlen = (torch.tensor(xlen, dtype = torch.float)-2).view(-1,1).data.cuda()#remove special token
    assert tuple(out[0].shape) == (x.size()[0], maxlen, self.textmodel.config.hidden_size)
    
    puretext = torch.div(torch.sum(torch.mul(out[0],mask), dim=1),xlen)
    comb = torch.cat((puretext, img1d),dim=1).unsqueeze(1)
    masked_out = torch.mul(out[0],mask)
    mask_sfmax = (1-mask)*-10000.0

    query1 = self.query1(comb)
    key1 = self.key1(masked_out)
    value1 = self.value1(masked_out)
    logit1 = torch.sum(query1*key1,dim=2)/math.sqrt(512)+mask_sfmax[:,:,0]
    attn1 = torch.nn.functional.softmax(logit1,dim=1)
    out1 = torch.sum(attn1.unsqueeze(2)*value1,dim=1)

    query2 = self.query2(comb)
    key2 = self.key2(masked_out)
    value2 = self.value2(masked_out)
    logit2 = torch.sum(query2*key2,dim=2)/math.sqrt(512)+mask_sfmax[:,:,0]
    attn2 = torch.nn.functional.softmax(logit2,dim=1)
    out2 = torch.sum(attn2.unsqueeze(2)*value2,dim=1)

    rawtext = torch.cat((puretext, torch.zeros((puretext.shape[0],1024-768),device=puretext.device)),dim=1)
    return (out1, attn1), (out2, attn2), text_tokens,  rawtext
  