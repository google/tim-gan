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

GPU=0
EP=60

#clevr
BS=16
DATAROOT=../datasets/Clevr
CROPSIZE=192,144
NAME=clevr
OP=adaroute
BLOCK=1
T=1
W=5
PRETRAIN=clevr_recon
N_LAYER=2

# codraw
# BS=16
# DATAROOT=../datasets/CoDraw
# CROPSIZE=160,128
# NAME=codraw
# OP=adaroute
# BLOCK=1
# T=1
# W=5
# PRETRAIN=codraw_recon
# N_LAYER=2

CUDA_VISIBLE_DEVICES=$GPU python train.py --batch_size $BS --dataroot $DATAROOT --crop_size $CROPSIZE --name $NAME --operator $OP --pretrain $PRETRAIN --n_ep $EP --n_downsampling $N_LAYER --num_adablock $BLOCK --temperature $T --w_gate $W
