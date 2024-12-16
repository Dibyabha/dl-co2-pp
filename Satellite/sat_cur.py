"""
In this section, we will distribute the satellite data from 71 different power plants from the
Kingdom of Saudi Arabia (KSA) into train, valid and test sets. 
A note: This dataset is proprietary to IBM.
"""

# Necessary packages
import os
import shutil
import random

# directory filename is where you have stored the data originally
dir = 'file'

# creating train, valid and test directories
train_dir = 'file/train'
valid_dir = 'file/valid'
test_dir = 'file/test'
os.makedirs(train_dir, exist_ok = True)
os.makedirs(valid_dir, exist_ok = True)
os.makedirs(test_dir, exist_ok = True)

files = os.listdir(dir)
satfiles = [f for f in files if f.endswith('.nc')]

# shuffle the files and then distribute them
random.shuffle(satfiles)

# training set gets 65%, validation set gets 15% and testing set gets 20% 
total_files = len(satfiles)
train_size = int(total_files * 0.65)
valid_size = int(total_files * 0.15)
train_files = satfiles[:train_size]
valid_files = satfiles[train_size:train_size + valid_size]
test_files = satfiles[train_size + valid_size:]

# This function moves the file into its target directory
def files_to_dir(file_list, target_dir):
    for file in file_list:
        shutil.move(os.path.join(dir, file), os.path.join(target_dir, file))

files_to_dir(train_files, train_dir)
files_to_dir(valid_files, valid_dir)
files_to_dir(test_files, test_dir)

""""
In KSA dataset, the xco2, no2, u and v for each day were separated by 366 indices
which means for day 1, xco2 was 0th index, no2 was 366th index, u was 732th index and v was 1098th index.
The output emission was reported in an annual manner meaning 1464th index value was emission rate value.
"""

import numpy as np
from netCDF4 import Dataset as NetCDFDataset

class CustomDataset:
    
    def __init__(self, file_path):
        self.ncfile = NetCDFDataset(file_path, 'r')
        self.basename = os.path.basename(file_path).replace('.nc', '')
        self.data = self.ncfile.variables[self.basename]

    def __len__(self):
        return 366

    def __getitem__(self, idx):
        if 0 <= idx < 366:
            
            xco2_arr = self.data[idx]
            no2_arr = self.data[idx + 366]
            u_arr = self.data[idx + 732]
            v_arr = self.data[idx + 1098]
            emiss_arr = self.data[1464]
            outputs = np.mean(emiss_arr) # mean value is taken as representative for the entire
            inputs = np.stack([xco2_arr, no2_arr, u_arr, v_arr], axis = -1)

            # normalization
            min_val = inputs.min(axis = (0, 1), keepdims = True)
            max_val = inputs.max(axis = (0, 1), keepdims = True)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1  
            inputs = (inputs - min_val) / range_val
            
            return inputs, outputs

    def __del__(self):
        self.ncfile.close()
        del self.data
        del self.ncfile
    
def load_to_data(file_paths):
    all_inputs = []
    all_outputs = []
    for file_path in file_paths:
        dataset = CustomDataset(file_path)
        for idx in range(len(dataset)):
            inputs, outputs = dataset[idx]
            if inputs is not None and inputs.shape == (64, 64, 4) and outputs != 0:
                all_inputs.append(inputs)
                all_outputs.append(outputs)
        del dataset
    return np.array(all_inputs), np.array(all_outputs)

train_set = 'file/train'
valid_set = 'file/valid'
test_set = 'file/test'

train_files = [os.path.join(train_set, f) for f in os.listdir(train_set) if f.endswith('.nc')]
valid_files = [os.path.join(valid_set, f) for f in os.listdir(valid_set) if f.endswith('.nc')]
test_files = [os.path.join(test_set, f) for f in os.listdir(test_set) if f.endswith('.nc')]

train_inputs, train_outputs = load_to_data(train_files)
valid_inputs, valid_outputs = load_to_data(valid_files)
test_inputs, test_outputs = load_to_data(test_files)

np.save('sat_train_inputs.npy', train_inputs)
np.save('sat_valid_inputs.npy', valid_inputs)
np.save('sat_test_inputs.npy', test_inputs)
np.save('sat_train_outputs.npy', train_outputs)
np.save('sat_valid_outputs.npy', valid_outputs)
np.save('sat_test_outputs.npy', test_outputs)

