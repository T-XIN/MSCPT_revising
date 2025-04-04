import pandas as pd
import numpy as np
import os
import torch
import h5py
from torch.utils.data import  Dataset
from torchvision import transforms
import torch.nn.functional as F
from transformers import AutoTokenizer
import json
from tqdm import tqdm
import os
import openslide
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_pretrained_tokenizer(model_name):
    if model_name == 'plip':
        model_name = 'vinid/plip'
        # tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
        tokenizer = AutoTokenizer.from_pretrained("path/to/weights", use_fast=True)
    elif model_name == 'clip':
        model_name = 'openai/clip-vit-base-patch16'
        tokenizer = AutoTokenizer.from_pretrained('path/to/weights', use_fast=True, TOKENIZERS_PARALLELISM=True)
    elif model_name == 'conch':
        from conch.open_clip_custom import get_tokenizer
        tokenizer = get_tokenizer()
    else:
        raise NotImplementedError
    return tokenizer

def get_transforms_ctranspath(img_size=224, 
                            mean = (0.485, 0.456, 0.406), 
                            std = (0.229, 0.224, 0.225)):
    trnsfrms = transforms.Compose(
                    [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean = mean, std = std)
                    ]
                )
    return trnsfrms
# load visual encoder weights and transforms
def load_ctranspath_clip(model_name, ckpt_path, img_size = 224, return_trsforms = True):

    if model_name == 'plip':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained("path/to/weights")
        if return_trsforms:
            trsforms = get_transforms_ctranspath(img_size = img_size)
            return model, trsforms
    elif model_name == 'clip':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained('path/to/weights')
        if return_trsforms:
            trsforms = get_transforms_ctranspath(img_size = img_size)
            return model, trsforms
    elif model_name == 'conch':
        from conch.open_clip_custom import create_model_from_pretrained
        model, trsforms = create_model_from_pretrained("conch_ViT-B-16", 
                                checkpoint_path="path/to/weights", 
                                force_image_size=img_size)
        if return_trsforms:
            return model, trsforms
    return model


def file_exists(df, root, ext = '.h5'):
    file_id = df['slide_id']
    if type(file_id) != str:
        file_id = str(file_id)
    df['has_h5'] = os.path.isfile(os.path.join(root, file_id + ext))
    return df


def read_assets_from_h5(h5_path):
    assets = {}
    attrs = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            assets[key] = f[key][:]
            if f[key].attrs is not None:
                attrs[key] = dict(f[key].attrs)
    return assets, attrs

def compute_patch_level(level0_mag, target_mag = 20, patch_size = 256):
    custom_downsample = int(level0_mag / target_mag)
    if custom_downsample == 1:
        target_level = 0
        target_patch_size = patch_size
    else:
        target_level = 0
        target_patch_size = int(patch_size * custom_downsample)
    return target_level, target_patch_size

def compute_patch_args(df, target_mag = 20, patch_size = 256):
    df['patch_level'], df['patch_size'] = compute_patch_level(df['level0_mag'], target_mag, patch_size)
    return df

class Whole_Slide_Bag_FP(Dataset):
    def __init__(self, pt_features):
        """
        Args:
            coords (string): coordinates to extract patches from w.r.t. level 0.
            custom_transforms (callable, optional): Optional transform to be applied on a sample
            target_patch_size (int): Custom defined image size before embedding
        """
        self.coords = coords
        self.pt_features = pt_features


    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        coord = self.coords[idx]
        feature = self.pt_features[idx]
        return {'feature': feature, 'coords': coord}
    
@torch.no_grad()
def extract_features(df, model_name, model, wsi_ext = '.svs', pt_path='', device = 'cuda:0'):
    model.to(device)
    model.eval()
    slide_id = df['slide_id']
    if type(slide_id) != str:
        slide_id = str(slide_id)
    if 'UBC' in df['project_id'] or 'PANDA' in df['project_id']:
        wsi_path = os.path.join(wsi_source, slide_id + wsi_ext)
        pt_file_path = os.path.join(pt_path, df['project_id'], model_name+'_5', slide_id + '.pt')
    else:
        wsi_path = os.path.join(wsi_source, df['project_id'].split('-')[-1], slide_id + wsi_ext)
        pt_file_path = os.path.join(pt_path, df['project_id'].split('-')[-1], model_name+'_5', slide_id + '.pt')
    wsi = openslide.open_slide(wsi_path)
    pt_file_path = pt_file_path.replace('clip', 'ViT-B-16')
    features = torch.load(pt_file_path).to(device)
    patch_level = df['patch_level']
    patch_size = df['patch_size']
    h5_path = os.path.join(h5_source, slide_id + '.h5')
    assets, _ = read_assets_from_h5(h5_path)
    return_coords = assets['coords']
    print(f'slide_id: {slide_id}, n_patches: {len(return_coords)}')

    with torch.no_grad():
        if model_name == 'conch':
            features = features @ model.visual.proj_contrast
        else:
            features = model.visual_projection(features)
        features = F.normalize(features, dim=-1) 
    return features, return_coords, wsi, patch_level, patch_size
            


import argparse
parser = argparse.ArgumentParser(description='Extract features using patch coordinates')
parser.add_argument('--h5_source', type=str,default = 'path/to/h5/files', help='path to dir containing patch h5s')
parser.add_argument('--wsi_source', type=str,default = 'path/to/wsi', help='path to dir containing wsis')
parser.add_argument('--pt_path', type=str, default='path/to/pt/files', help='path to features')
parser.add_argument('--save_dir', type=str, default='path/to/save/dir', help='path to save extracted features')
parser.add_argument('--wsi_ext', type=str, default='.svs', help='extension for wsi')
parser.add_argument('--ckpt_path', type=str, help='path to clip ckpt')
parser.add_argument('--device', type=str, default='cuda:0', help='device cuda:n')
parser.add_argument('--model_name', type=str, default='conch')
parser.add_argument('--gpt_data', type=str, default='./train_data/gpt/description')
parser.add_argument('--top_k', type=int, default=10) # 10 for UBC_OCEAN, 30 for others
parser.add_argument('--split', type=str, default='')
parser.add_argument('--dataset_name', type=str, default='UBC-OCEAN', choices=['Lung', 'RCC', 'BRCA', 'UBC-OCEAN', 'PANDA'])
args = parser.parse_args()


def tokenize(texts, tokenizer, model_name='plip'):
    if model_name == 'conch':
        tokens = tokenizer.batch_encode_plus(texts, 
                                    max_length = 127,
                                    add_special_tokens=True, 
                                    return_token_type_ids=False,
                                    truncation = True,
                                    padding = 'max_length',
                                    return_tensors = 'pt')
        tokens = F.pad(tokens['input_ids'], (0, 1), value=tokenizer.pad_token_id)
        return tokens, None
    else:
        tokens = tokenizer.batch_encode_plus(texts, 
                                            max_length = 64,
                                            add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                            return_token_type_ids=False,
                                            truncation = True,
                                            padding = 'max_length',
                                            return_attention_mask=True)
    return tokens['input_ids'], tokens['attention_mask']


label_dicts = {
    'RCC': {'CHRCC': 0, 'CCRCC': 1, 'PRCC': 2},
    'Lung': {'LUAD': 0, 'LUSC': 1},
    'BRCA': {'Low': 0, 'High': 1},
    'UBC-OCEAN': {'LGSC':0, 'HGSC':1, 'EC':2,'CC':3, 'MC':4},
    'PANDA': {'LowRisk':0, 'IntermediateRisk':1, 'HighRisk':2}
}

project_dic = {
    'RCC': ['KIRC', 'KIRP', 'KICH'],
    'Lung': ['LUAD', 'LUSC'],
    'BRCA': ['BRCA'],
    'UBC-OCEAN': ['UBC-OCEAN'],
    'PANDA': ['PANDA']
}

csv_dic = {
    'RCC': './tcga_rcc.csv',
    'Lung': './tcga_lung.csv',
    'COADREAD': './tcga_coadread.csv',
    'BRCA': './tcga_brca.csv',
    'UBC-OCEAN': './ubc_ocean.csv',
    'PANDA':'./panda.csv'
}

if __name__ == '__main__':
    if args.dataset_name in ['UBC-OCEAN']:
        args.wsi_ext = '.tif'
    elif args.dataset_name in ['PANDA']:
        args.wsi_ext = '.tiff'
    else:
        args.wsi_ext = '.svs'
    csv_path = csv_dic[args.dataset_name]
    h5_source = args.h5_source
    wsi_source = args.wsi_source
    ckpt_path = args.ckpt_path
    device = args.device 
    prompt_file = os.path.join(args.gpt_data,f'{args.dataset_name.upper()}_select_pic.json')
    with open(prompt_file, 'r') as pf: 
        prompts = json.load(pf)

    model, trsforms = load_ctranspath_clip(model_name=args.model_name,
                                ckpt_path=ckpt_path, 
                                img_size = 224, 
                                return_trsforms = True)
    model.to(args.device)
    # Load tokenizer
    tokenizer = load_pretrained_tokenizer(args.model_name)
    all_weights = []
    prompts = [prompts[str(prompt_idx)] for prompt_idx in range(100)]
    for prompt in prompts:
        # Your code here
        classnames = prompt['classnames']
        templates = prompt['templates']
        idx_to_class = {v:k for k,v in label_dicts[args.dataset_name].items()}
        n_classes = len(idx_to_class)
        classnames_text = [classnames[idx_to_class[idx]] for idx in range(n_classes)]

        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames_text:
                texts = [template.replace('CLASSNAME', classname) for template in templates]

                texts, attention_mask = tokenize(texts, tokenizer, args.model_name) # Tokenize with custom tokenizer
                if args.model_name == 'conch':
                    class_embeddings = model.encode_text(texts.to(device), normalize=False)
                else:
                    texts = torch.from_numpy(np.array(texts)).to(device)
                    attention_mask = torch.from_numpy(np.array(attention_mask)).to(device)
                    class_embeddings = model.get_text_features(texts, attention_mask=attention_mask)
                
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        text_feats = torch.stack(zeroshot_weights, dim=0).to(device)
        all_weights.append(text_feats)
            
    text_feats = torch.stack(all_weights, dim=0).mean(dim=0)
    text_feats = F.normalize(text_feats, dim=-1)

    if not args.split:
        split_list = project_dic[args.dataset_name]
    else:
        split_list = [args.split]
    args.save_dir = os.path.join(args.save_dir, args.model_name, args.dataset_name, 'K_100')
    for args.split in split_list:
        df = pd.read_csv(csv_path)
        if args.split == 'KIRC':
            df = df[df['project_id']=='TCGA-KIRC']
        elif args.split == 'KIRP':
            df = df[df['project_id']=='TCGA-KIRP']
        elif args.split == 'KICH':
            df = df[df['project_id']=='TCGA-KICH']
        elif args.split == 'LUAD':
            df = df[df['project_id']=='TCGA-LUAD']
        elif args.split == 'LUSC':
            df = df[df['project_id']=='TCGA-LUSC']
        assert 'level0_mag' in df.columns, 'level0_mag column missing'
        h5_source = os.path.join(args.h5_source, args.split + '_5/patches')
        df = df.apply(lambda x: file_exists(x, h5_source), axis=1)
        df['has_h5'].value_counts()
        # df['has_slide'] = df['slide_id'].apply(lambda x: file_exists(x, wsi_source, ext='.svs'))
        # df = df[df['has_slide']]
        df = df[df['has_h5']]
        df = df.reset_index(drop=True)
        assert df['has_h5'].sum() == len(df['has_h5'])
        # assert df['has_slide'].sum() == len(df['has_slide'])
        df['pred'] = np.nan 
        df = df.apply(lambda x: compute_patch_args(x, target_mag = 5, patch_size = 256), axis=1)
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)


        for idx in tqdm(range(len(df))):
            slide_id = df.iloc[idx]['slide_id']
            if type(slide_id) != str:
                slide_id = str(slide_id)
            save_path = os.path.join(args.save_dir, slide_id)
            os.makedirs(save_path, exist_ok=True)
            category = label_dicts[args.dataset_name][df.iloc[idx]['OncoTreeCode']]
            img_feats, coords, wsi, patch_level, patch_size = extract_features( df.iloc[idx], model_name=args.model_name,
                                                                                wsi_ext=args.wsi_ext, 
                                                                                pt_path=args.pt_path,
                                                                                model=model, 
                                                                                device=device)
            logits = text_feats @ img_feats.T
            logits = logits.cpu()
            if args.top_k > img_feats.shape[0]:
                topk_values, topk_indices = torch.topk(logits, img_feats.shape[0], dim=1)
            else:
                topk_values, topk_indices = torch.topk(logits, args.top_k, dim=1)
            pred = topk_values.sum(dim=1).argmax().cpu().item()
            select_id = topk_indices.flatten().cpu().numpy()
            coord = coords[select_id]
            scores = logits[pred][select_id].numpy()
            df.loc[idx, 'pred'] = 1 if pred == category else 0  # 进行赋值
            for idx, (x,y) in enumerate(coord):
                big_img = wsi.read_region((x,y), patch_level, (patch_size, patch_size)).convert('RGB')
                big_img = big_img.resize((224,224))
                big_img.save(os.path.join(save_path, f"score_{scores[idx]:.4f}_{idx}_{x}_{y}.png"))
        df.to_csv(f'./{args.split}_result_{args.model_name}.csv', index=False)
