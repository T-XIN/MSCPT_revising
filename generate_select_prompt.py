import argparse
import numpy as np
import json
import os

parser = argparse.ArgumentParser(description='generate a set of prompts for zeroshot eval')
parser.add_argument('--iters', type=int, default=50, help='num sampling iters')
parser.add_argument('--dataset', type=str, choices=['rcc', 'brca', 'nsclc', 'ubc_ocean'], default='brca', help='name of dataset')
parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite prompt file if already exists')
args = parser.parse_args()

def main(args):
    if args.dataset == 'rcc':
        classnames_pool = {
            'CHRCC': ['chromophobe renal cell carcinoma', 'renal cell carcinoma, chromophobe type', 'renal cell carcinoma of the chromophobe type', 'chromophobe RCC'],  
            'CCRCC': ['clear cell renal cell carcinoma', 'renal cell carcinoma, clear cell type', 'renal cell carcinoma of the clear cell type', 'clear cell RCC'],
            'PRCC': ['papillary renal cell carcinoma', 'renal cell carcinoma, papillary type', 'renal cell carcinoma of the papillary type', 'papillary RCC']
        }
    elif args.dataset == 'brca':
        classnames_pool = {
            'Low': [
                'low risk of recurrence', 
                'good prognosis', 
                'better prognosis', 
                'less likely to recurrence', 
                'favorable outcome', 
                'reduced likelihood of recurrence'
            ],
            'High': [
                'high risk of recurrence', 
                'poor prognosis', 
                'bad prognosis', 
                'more likely to recurrence', 
                'challenging long-term outlook', 
                'elevated likelihood of recurrence'
            ]
        }
    elif args.dataset == 'nsclc':
        classnames_pool = {
            'LUAD': ['adenocarcinoma', 'lung adenocarcinoma', 'adenocarcinoma of the lung', 'pulmonary adenocarcinoma', 'adenocarcinoma, lepidic pattern', 'adenocarcinoma, solid pattern', 'adenocarcinoma, micropapillary pattern', 'adenocarcinoma, acinar pattern', 'adenocarcinoma, papillary pattern'],
            'LUSC': ['squamous cell carcinoma', 'lung squamous cell carcinoma', 'squamous cell carcinoma of the lung', 'pulmonary squamous cell carcinoma']
        }
    elif args.dataset == 'ubc_ocean':
        classnames_pool = {
            'CC': [
                'clear cell carcinoma', 
                'ovarian clear cell carcinoma', 
                'clear cell adenocarcinoma of the ovary', 
                'ovarian clear cell adenocarcinoma', 
                'clear cell carcinoma, NOS'
            ],
            'EC': [
                'endometrioid carcinoma', 
                'ovarian endometrioid carcinoma', 
                'endometrioid adenocarcinoma of the ovary', 
                'endometrioid carcinoma, low-grade', 
                'endometrioid carcinoma, high-grade'
            ],
            'HGSC': [
                'high-grade serous carcinoma', 
                'ovarian high-grade serous carcinoma', 
                'high-grade serous adenocarcinoma of the ovary', 
                'serous carcinoma, high-grade', 
                'high-grade serous ovarian cancer'
            ],
            'LGSC': [
                'low-grade serous carcinoma', 
                'ovarian low-grade serous carcinoma', 
                'low-grade serous adenocarcinoma of the ovary', 
                'serous carcinoma, low-grade', 
                'low-grade serous ovarian cancer'
            ],
            'MC': [
                'mucinous carcinoma', 
                'ovarian mucinous carcinoma', 
                'mucinous adenocarcinoma of the ovary', 
                'mucinous carcinoma, intestinal type', 
                'mucinous carcinoma, endocervical type'
            ]
        }
    else:
        raise NotImplementedError

    templates_pool = [
    'CLASSNAME.',
    'a photomicrograph showing CLASSNAME.',
    'a photomicrograph of CLASSNAME.',
    'an image of CLASSNAME.',
    'an image showing CLASSNAME.', 
    'an Lung of CLASSNAME.',
    'CLASSNAME is shown.',
    'this is CLASSNAME.',
    'there is CLASSNAME.',
    'a histopathological image showing CLASSNAME.',
    'a histopathological image of CLASSNAME.',
    'a histopathological photograph of CLASSNAME.',
    'a histopathological photograph showing CLASSNAME.',
    'shows CLASSNAME.',
    'presence of CLASSNAME.',
    'CLASSNAME is present.'
    ]


    iters = args.iters
    sizerange = range(1, len(templates_pool)+1)


    path_to_prompts = f'./train_data/gpt/description/{args.dataset}_select_pic.json' 
    if not args.overwrite and os.path.isfile(path_to_prompts):
        return

    sampled_prompts = {}
    for i in range(iters):
        size = np.random.choice(sizerange)
        classnames_subset = {k: np.random.choice(v, size=1, replace=False)[0] for k, v in classnames_pool.items()}
        template_subset = np.random.choice(templates_pool, size=size, replace=False).tolist()
        sampled_prompts[i] = {
            'classnames': classnames_subset,
            'templates': template_subset
        }

    json.dump(sampled_prompts, open(path_to_prompts, 'w'), indent=4)

if __name__ == '__main__':
    main(args)
    
