import pandas as pd
import numpy as np
import os, logging, json
from check_structure import check_existing_file
from sklearn.preprocessing import StandardScaler


with open('config.json', 'r') as f:
    data_folderpath = json.load(f)
    X_train = pd.read_csv(f"{data_folderpath['output_folderpath']}/X_train.csv")
    X_test  = pd.read_csv(f"{data_folderpath['output_folderpath']}/X_test.csv")
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

def normalize_data(data_folderpath):
    for file, filename in zip([X_train, X_test], ['X_train_scaled', 'X_test_scaled']):
        output_filepath = os.path.join(data_folderpath, f'{filename}.csv')
        if check_existing_file(output_filepath):
            file = pd.DataFrame(file)
            file.to_csv(output_filepath, index=False)

def main():
    normalize_data(data_folderpath['output_folderpath'])

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()