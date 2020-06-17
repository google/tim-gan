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
import random

IMG_EXTENSIONS = [
  '.jpg', '.JPG', '.jpeg', '.JPEG',
  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff']
def image_file(filenames):
  out = []
  for name in filenames:
    if any(name.endswith(extension) for extension in IMG_EXTENSIONS):
      out.append(name)
  return out

def get_crop_params(img_size, output_size):
  w = img_size[1]
  h = img_size[0]
  tw = output_size[1]
  th = output_size[0]
  if w == tw and h == th:
    return 0, 0, h, w
 
  i = random.randint(0, h - th)
  j = random.randint(0, w - tw)
  return (i, j, th, tw)
