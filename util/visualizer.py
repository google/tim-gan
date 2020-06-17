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

import numpy as np
import os
import ntpath
import time
from . import util
from . import html
import scipy.misc
from io import BytesIO

def save_images(webpage, visuals, image_path, win_size=512):
  image_dir = webpage.get_image_dir()
  short_path = ntpath.basename(image_path[0])
  name = os.path.splitext(short_path)[0]

  webpage.add_header(name)
  ims = []
  txts = []
  links = []

  for label, image_numpy in visuals.items():
    if label.startswith('output'):
      fulllabel = label
      label = 'output'
    else:
      fulllabel = label
    image_name = '%s_%s.jpg' % (name, label)
    save_path = os.path.join(image_dir, image_name)
    util.save_image(image_numpy, save_path)

    ims.append(image_name)
    txts.append(fulllabel)
    links.append(image_name)
  webpage.add_images(ims, txts, links, width=win_size)
