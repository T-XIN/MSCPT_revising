import pandas as pd
import os

csv_dir = './tcga_brca.csv'
df = pd.read_csv(csv_dir)
categories = df['OncoTreeCode'].unique()
exist_pt_path = 'path/to/pt/files'
exist_pt_slides = []
for project_id in ['BRCA']:
    exist_pt_path_ = exist_pt_path.replace('PROJECT_ID', project_id)
    for file in os.listdir(exist_pt_path_):
        exist_pt_slides.append(file.split('.pt')[0])
df = df[df.slide_id.isin(exist_pt_slides)]
numshots = [0, 1, 2, 4, 8, 16]


for seed in range(1,6):
    val_df = df.sample(n=int(len(df)*0.8), random_state=seed)
    train_df_ = df.drop(val_df.index)
    val_df = val_df.reset_index(drop=True)
    train_df = []
    for idx in range(1,6):
        for categorie in categories:
            samples = train_df_[train_df_['OncoTreeCode']==categorie].sample(n=numshots[idx]-numshots[idx-1], random_state=seed)
            train_df.append(samples)
            train_df_ = train_df_.drop(samples.index)
        save_train_df = pd.concat(train_df, axis=0)
        save_train_df = save_train_df.reset_index(drop=True)
        
        save_train_df.to_csv(f'./numshots/BRCA/BRCA_train_{numshots[idx]}_{seed}.csv', index=False)
        val_df.to_csv(f'./numshots/BRCA/BRCA_val_{numshots[idx]}_{seed}.csv', index=False)

