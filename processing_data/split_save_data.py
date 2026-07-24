from sklearn.model_selection import train_test_split
import pandas as pd

# Split data
# Total training: 50% ALMG + 100% Câmara
# Effective training: 90% of total training
# Validation: 10% of total training
# Test: 50% ALMG
def split_save_data(data_path_output, seed):
    data_path_almg = f"{data_path_output}/sb2_almg_pairs.csv"
    data_path_camara = f"{data_path_output}/sb2_chamber_pairs.csv"

    df_almg_25_75 = pd.read_csv(data_path_almg)
    df_camara_25_75 = pd.read_csv(data_path_camara)

    df_almg_train, df_test = train_test_split(
        df_almg_25_75, test_size=0.5, random_state=seed, stratify=df_almg_25_75['label']
    )

    df_train_todos = pd.concat([df_almg_train, df_camara_25_75],ignore_index=True)
    
    df_train, df_eval = train_test_split(
        df_train_todos, test_size=0.1, random_state=seed, stratify=df_train_todos['label']
    )
    
    df_train.to_csv(f'{data_path_output}/train.csv', index=False)
    df_eval.to_csv(f'{data_path_output}/eval.csv', index=False)
    df_test.to_csv(f'{data_path_output}/test.csv', index=False)
