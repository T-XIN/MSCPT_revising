import h5py
import openslide
import argparse
import numpy as np
from tqdm import tqdm
import os
import cv2
import pandas as pd
import json
import torch
from PIL import Image, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import torch.nn.functional as F


def tokenize(tokenizer, texts):
    tokens = tokenizer.batch_encode_plus(texts, 
                                        max_length = 64,
                                        add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                                        return_token_type_ids=False,
                                        truncation = True,
                                        padding = 'max_length',
                                        return_attention_mask=True)
    return tokens['input_ids'], tokens['attention_mask']

# ctranspath transformation
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

def load_pretrained_tokenizer(model_name):
    if model_name == 'plip':
        model_name = 'vinid/plip'
        # tokenizer = AutoTokenizer.from_pretrained(model_name, fast=True)
        tokenizer = AutoTokenizer.from_pretrained("/home/hmh/weights/plip_weight/", use_fast=True)
    elif model_name == 'clip':
        model_name = 'openai/clip-vit-base-patch16'
        tokenizer = AutoTokenizer.from_pretrained('/home/hmh/weights/ViT-B-16', use_fast=True, TOKENIZERS_PARALLELISM=True)
    else:
        raise NotImplementedError
    return tokenizer

# load visual encoder weights and transforms
def load_ctranspath_clip(model_name, img_size = 224, return_trsforms = True):

    if model_name == 'plip':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained("path/to/weights")
        if return_trsforms:
            trsforms = get_transforms_ctranspath(img_size = img_size)
            return model, trsforms
    elif model_name == 'clip':
        from transformers import CLIPModel
        model = CLIPModel.from_pretrained('/home/hmh/weights/ViT-B-16')
        if return_trsforms:
            trsforms = get_transforms_ctranspath(img_size = img_size)
            return model, trsforms
    return model

def visualize_categorical_heatmap(
        args,
        wsi,
        slide_id,
        coords, 
        score,
        vis_level=None,
        patch_size=(256, 256),
        alpha=0.4,
        verbose=True,
        cmap=None,
        save_topK=False
    ):

    topK = []
    if save_topK and not args.plot_select:
        topK_index = np.argpartition(score.flatten(), -10)[-10:]
        for idx in topK_index:
            coord = coords[idx]
            topK_score = score[idx]
            img_block = wsi.read_region(coord, 0, patch_size).convert("RGB")
            topK.append((coord, topK_score, img_block))
    # Scaling from 0 to desired level
    downsample = int(wsi.level_downsamples[vis_level])
    scale = [1/downsample, 1/downsample]

    if len(score.shape) == 1:
        score = score.reshape(-1, 1)

    top_left = (0, 0)
    bot_right = wsi.level_dimensions[0]
    region_size = tuple((np.array(wsi.level_dimensions[0]) * scale).astype(int))
    w, h = region_size
    patch_size_orig = patch_size
    patch_size = np.ceil(np.array(patch_size) * np.array(scale)).astype(int)


    coords = np.ceil(coords * np.array(scale)).astype(int)

    if verbose:
        print(f'\nCreating heatmap for: {slide_id}')
        print('Top Left: ', top_left, 'Bottom Right: ', bot_right)
        print('Width: {}, Height: {}'.format(w, h))
        print(f'Original Patch Size / Scaled Patch Size: {patch_size_orig} / {patch_size}')
    
    vis_level = wsi.get_best_level_for_downsample(downsample)
    img = wsi.read_region(top_left, vis_level, wsi.level_dimensions[vis_level]).convert("RGB")
    if img.size != region_size:
        img = img.resize(region_size, resample=Image.Resampling.BICUBIC)
    img = np.array(img)
    
    if verbose:
        print('vis_level: ', vis_level)
        print('downsample: ', downsample)
        print('region_size: ', region_size)
        print('total of {} patches'.format(len(coords)))
    
    for idx in range(len(coords)):
        coord = coords[idx]
        slide_score = score[idx]
        if slide_score == -1:
            color = np.array([[0,0,0,0]])
        else:
            color = cmap(score[idx])
        color = (int(color[0][0]*255), int(color[0][1]*255), int(color[0][2]*255))
        img_block = img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]].copy()
        color_block = (np.ones((img_block.shape[0], img_block.shape[1], 3)) * color).astype(np.uint8)
        blended_block = cv2.addWeighted(color_block, alpha, img_block, 1 - alpha, 0)
        blended_block = np.array(ImageOps.expand(Image.fromarray(blended_block), border=0, fill=(50,50,50)).resize((img_block.shape[1], img_block.shape[0])))
        img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = blended_block

    img = Image.fromarray(img)
    return img, topK

def plot_and_save(args, wsi, coords, heatmap_score, patch_size, slide_id, cmap, save_topK):
    cat_map, topK = visualize_categorical_heatmap(
        args,
        wsi,
        slide_id,
        coords, 
        heatmap_score, 
        vis_level=wsi.get_best_level_for_downsample(128),
        patch_size=(patch_size, patch_size),
        alpha=0.7,
        cmap=cmap,
        save_topK=save_topK
    )
    cat_map.resize((cat_map.width//2, cat_map.height//2)).save(os.path.join(args.save_dir,f'{slide_id}_scale_{args.scale}X.png'))
    if save_topK:
        for topK_img in topK:
            coord, topK_score, img_block = topK_img
            img_block.save(os.path.join(args.save_dir, f'{slide_id}_scale_{args.scale}X_{coord[0]}_{coord[1]}_{topK_score[0]:.4f}.png'))

parser = argparse.ArgumentParser(description='Configurations for WSI Training')
parser.add_argument('--base_model', type=str, default='clip')
parser.add_argument('--model_name', default='mscpt', type=str)
parser.add_argument('--project_id', type=str, default='Lung')
parser.add_argument('--h5_root', type=str, default='/home/dataset5/hmh_data/CLAM/')
parser.add_argument('--pt_path', type=str, default='/home/dataset6/hmh_data/features/', help='path to features')
parser.add_argument('--wsi_root', type=str, default='/home/dataset4/hmh_data/TCGA_raw/')
parser.add_argument('--heatmap_score_root', type=str, default='heatmap/BASE_MODEL/PROJECT_ID/MODEL_NAME')
parser.add_argument('--gpt_data', type=str, default='./train_data/gpt/description/')
parser.add_argument('--plot_select', default=True)
parser.add_argument('--save_dir', default='visualization/', type=str)
parser.add_argument('--save_topK', default=False)
parser.add_argument('--scale', default=5, type=int)

args = parser.parse_args()
args.device = 'cuda:0'
Sample_picture ={}
args.heatmap_score_root = args.heatmap_score_root.replace('BASE_MODEL', args.base_model).replace('PROJECT_ID', args.project_id).replace('MODEL_NAME', args.model_name)
label_dicts = {
    'Lung': {'LUAD': 0, 'LUSC': 1},
    'BRCA': {'Low': 0, 'High': 1},
    'RCC': {'CHRCC': 0, 'CCRCC': 1, 'PRCC': 2}
}
label_dicts = label_dicts[args.project_id]
### Loading PANTHER Encoder
slide_list = os.listdir(args.heatmap_score_root)
if args.plot_select:
    args.save_dir = os.path.join(args.save_dir, 'plot_select', args.base_model, args.project_id)
else:
    args.save_dir = os.path.join(args.save_dir, 'heatmap', args.base_model, args.project_id, args.model_name)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir, exist_ok=True)
cmap = plt.cm.get_cmap('coolwarm')
meta_df = pd.read_csv(os.path.join(f'tcga_{args.project_id.lower()}.csv'))
meta_df = meta_df.set_index('slide_id')

df = pd.read_csv(f'./tcga_{args.project_id.lower()}.csv')

if args.plot_select:
    args.scale = 5
    prompt_file = os.path.join(args.gpt_data,f'{args.project_id}_select_pic.json')
    with open(prompt_file, 'r') as pf: 
        prompts = json.load(pf)

    model, trsforms = load_ctranspath_clip(model_name=args.base_model,
                                img_size = 224, 
                                return_trsforms = True)
    model.to(args.device)
    # Load tokenizer
    tokenizer = load_pretrained_tokenizer(args.base_model)
    all_weights = []
    for prompt_idx in range(len(prompts)):
        prompt = prompts[str(prompt_idx)]
        classnames = prompt['classnames']
        templates = prompt['templates']
        idx_to_class = {v:k for k,v in label_dicts.items()}
        n_classes = len(idx_to_class)
        classnames_text = [classnames[idx_to_class[idx]] for idx in range(n_classes)]

        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames_text:
                texts = [template.replace('CLASSNAME', classname) for template in templates]

                texts, attention_mask = tokenize(tokenizer, texts) # Tokenize with custom tokenizer
                texts = torch.from_numpy(np.array(texts)).to(args.device)
                attention_mask = torch.from_numpy(np.array(attention_mask)).to(args.device)
                class_embeddings = model.get_text_features(texts, attention_mask=attention_mask)
                
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
        text_feats = torch.stack(zeroshot_weights, dim=0).to(args.device)
        all_weights.append(text_feats)
            
    text_feats = torch.stack(all_weights, dim=0).mean(dim=0)
    text_feats = F.normalize(text_feats, dim=-1)


for slide_id in tqdm(slide_list):
    if '.pkl' not in slide_id:
        continue
    slide_id = slide_id.split('.pkl')[0]
    ### open your WSI and features
    slide_fpath = os.path.join(args.wsi_root, df[df['slide_id']==slide_id]['project_id'].item().split('-')[-1], f'{slide_id}.svs')
    h5_feats_fpath = os.path.join(args.h5_root, df[df['slide_id']==slide_id]['project_id'].item().split('-')[-1]+'_' + str(args.scale), f'patches/{slide_id}.h5')
    wsi = openslide.open_slide(slide_fpath)
    h5 = h5py.File(h5_feats_fpath, 'r')
    coords = h5['coords'][:]
    patch_size = h5['coords'].attrs['patch_size']
    label = meta_df.loc[slide_id]['OncoTreeCode']
    label = label_dicts[label]
    if args.plot_select:
        pt_file_path = os.path.join(args.pt_path, df[df['slide_id']==slide_id]['project_id'].item().split('-')[-1], args.base_model+'_5', slide_id + '.pt')
        pt_file_path = pt_file_path.replace('clip', 'ViT-B-16')
        features = torch.load(pt_file_path).to(args.device)
        with torch.no_grad():
            features = model.visual_projection(features)
            features = F.normalize(features, dim=-1)
        sim_score = text_feats @ features.T
        heatmap_score = torch.mean(sim_score, dim=0).cpu().numpy()
        heatmap_score = (heatmap_score-heatmap_score.min()) / (heatmap_score.max()-heatmap_score.min())
        # heatmap_score = np.where(np.isin(np.arange(features.shape[0]), select_id), sim_score, -1)
        heatmap_score = heatmap_score.reshape(-1, 1)
    else:
        heatmap_score = np.load(os.path.join(args.heatmap_score_root, f'{slide_id}.npy'))
        if args.model_name == 'mscpt':
            heatmap_score = heatmap_score[:, label, :]
            heatmap_score = heatmap_score.max(axis=1, keepdims=True)

        else:
            heatmap_score = heatmap_score.reshape(-1,1)
        heatmap_score = (heatmap_score-heatmap_score.min()) / (heatmap_score.max()-heatmap_score.min())


    ## Visualize
    plot_and_save(args, wsi, coords, heatmap_score, patch_size, slide_id, cmap, args.save_topK)