# Copyright 2021 Zhongyang Zhang
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.utils.data as data
import pandas as pd
import os


class StandardData(data.Dataset):
    def __init__(self, feat_data_dir=r'data/ref',
                 train=True,
                 csv_dir='',
                 model_name='',
                 label_dicts={},
                 num_shots=None,
                 seed=42,
                 base_model='plip',
                 dataset_name='Lung'):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.feat_data_dir = feat_data_dir
        self.index_col = 'slide_id'
        self.target_col = 'OncoTreeCode'
        self.project_id = 'project_id'
        self.num_shots = num_shots
        self.seed = seed
        self.csv_dir = csv_dir
        self.label_dicts = label_dicts
        self.dataset_name = dataset_name
        self.train = train
        self.model_name = model_name
        self.base_model = base_model
        self.check_files()

    def check_files(self):
        # This part is the core code block for load your own dataset.
        # You can choose to scan a folder, or load a file list pickle
        # file, or any other formats. The only thing you need to gua-
        # rantee is the `self.path_list` must be given a valid value.

        train_df = pd.read_csv(f'./numshots/{self.dataset_name}/{self.dataset_name}_train_{self.num_shots}_{self.seed}.csv')
        val_df = pd.read_csv(f'./numshots/{self.dataset_name}/{self.dataset_name}_val_{self.num_shots}_{self.seed}.csv')
        self.data =  train_df if self.train else val_df 


    def __len__(self):
        return len(self.data)
    
    def get_ids(self, ids):
        return self.data.loc[ids, self.index_col]

    def get_labels(self, ids):
        return self.data.loc[ids, self.target_col]

    def get_project_id(self, ids):
        project_id = self.data.loc[ids, self.project_id]
        if 'tcga' in project_id or 'TCGA' in project_id:
            return project_id.split('-')[-1]
        return project_id

    def __getitem__(self, idx):
        slide_id = self.get_ids(idx)
        if type(slide_id) != str:
            slide_id = str(slide_id)
        label = self.get_labels(idx)
        project_id = self.get_project_id(idx)
        pt_path = os.path.join(self.feat_data_dir, project_id, self.base_model + '_5', slide_id+'.pt')
        pt_path = pt_path.replace('clip', 'ViT-B-16')
        cla_img = torch.load(pt_path)
        pt_path_20 = os.path.join(self.feat_data_dir, project_id, self.base_model + '_20', slide_id+'.pt')
        pt_path_20 = pt_path_20.replace('clip', 'ViT-B-16')
        cla_img_20 = torch.load(pt_path_20)

        if self.label_dicts is not None:
            label = self.label_dicts[label]

        return (cla_img, cla_img_20), label, slide_id
