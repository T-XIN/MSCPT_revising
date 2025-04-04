import pandas as pd

dataset_id = 'Lung'
for seed in range(1,6):
    df_16_train = pd.read_csv(f'numshots/{dataset_id}/{dataset_id}_train_16_{seed}.csv')
    df_16_val = pd.read_csv(f'numshots/{dataset_id}/{dataset_id}_val_16_{seed}.csv')
    num_class = int(len(df_16_train) / 16)
    seg_num = 16
    for shot in [1, 2, 4, 8]:
        df_shot_train = []
        for i in range(num_class):
            df_shot_train.append(df_16_train[i*seg_num:i*seg_num+shot])
        df_shot_train = pd.concat(df_shot_train, ignore_index=True)
        df_shot_val = df_16_val.copy()
        df_shot_train.to_csv(f'numshots/{dataset_id}/{dataset_id}_train_{shot}_{seed}.csv', index=False)
        df_shot_val.to_csv(f'numshots/{dataset_id}/{dataset_id}_val_{shot}_{seed}.csv', index=False)