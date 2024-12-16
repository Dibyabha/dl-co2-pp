""" In this step, we will combine the datasets from Lippendorf and Boxberg
and create a simulated dataset. """

# necessary packages
import numpy as np
from netCDF4 import Dataset

class CustomDataset:
    def __init__(self, file_path, variable = None):
        self.ncfile = Dataset(file_path, 'r')
        self.variable = variable

    def __len__(self):
        return self.ncfile.variables['xco2_noisy'].shape[0]

    def __getitem__(self, idx):
        xco2_read = self.ncfile.variables['xco2_noisy'][idx]
        u_read = self.ncfile.variables['u'][idx]
        v_read = self.ncfile.variables['v'][idx]
        emiss_read = self.ncfile.variables['emiss'][idx]
        
        xco2_arr = np.array(xco2_read).astype('float32')
        u_arr = np.array(u_read).astype('float32')
        v_arr = np.array(v_read).astype('float32')
        emiss_arr = np.array(emiss_read).astype('float32')
        
        if self.variable:
            var_read = self.ncfile.variables[self.variable][idx]
            var_arr = np.array(var_read).astype('float32')
            inputs = np.stack([xco2_arr, u_arr, v_arr, var_arr], axis = -1)
        else:
            inputs = np.stack([xco2_arr, u_arr, v_arr], axis = -1)
        
        min_val = inputs.min(axis = (0, 1), keepdims = True)
        max_val = inputs.max(axis = (0, 1), keepdims = True)
        max_val[max_val == min_val] = min_val[max_val == min_val] + 1
        inputs = (inputs - min_val) / (max_val - min_val)
        
        weights = np.array([0.0, 1.0, 0.0])
        weighted = np.round(np.sum(weights * emiss_arr), 3)
        outputs = np.array([weighted], dtype = np.float32)
        
        return inputs, outputs

    def __del__(self):
        self.ncfile.close()

# save_data function is basically to take the inputs and outputs and save it in npy format for further use 
def save_data(dataset, input_file, output_file):
    num_samples = len(dataset)
    inputs_all = np.zeros((num_samples, 64, 64, 4 if dataset.variable else 3), dtype = np.float32) # we will have (N, 64, 64, 4) as we have 4 inputs
    outputs_all = np.zeros((num_samples, 1), dtype = np.float32) # we will have (N, 1)
    
    for idx in range(num_samples):
        inputs, outputs = dataset[idx]
        inputs_all[idx] = inputs
        outputs_all[idx] = outputs
    
    np.save(input_file, inputs_all)
    np.save(output_file, outputs_all)

# The file names will be your choice
# Another point to remember is the same process needs to repeated for the other location
# For eg., if first time you did it for Lippendorf then the same process needs to be repeated for Boxberg
train_set = 'train_file.nc'
valid_set = 'valid_file.nc'
test_set = 'test.nc'

train_dataset = CustomDataset(train_set, variable = 'no2_noisy')
valid_dataset = CustomDataset(valid_set, variable = 'no2_noisy')
test_dataset = CustomDataset(test_set, variable = 'no2_noisy')

# Instead of loc you can put lip or box depending on the location
save_data(train_dataset, 'train_inputs_loc.npy', 'train_outputs_loc.npy')
save_data(valid_dataset, 'valid_inputs_loc.npy', 'valid_outputs_loc.npy')
save_data(test_dataset, 'test_inputs_loc.npy', 'test_outputs_loc.npy')

"""
Once the inputs and outputs npy files for train, valid and test for each location is done
we will curate the simulated dataset
"""
# If your file names are different, change them accordingly
# Also, a point to keep in mind is that we have to repeat this
# for each type : train, valid and test
inputs_lip = np.load('train_inputs_lip.npy')
inputs_box = np.load('train_inputs_box.npy')
sim_inputs = np.concatenate([inputs_lip, inputs_box], axis = 0)
outputs_lip = np.load('train_outputs_lip.npy')
outputs_box = np.load('train_outputs_box.npy')
sim_outputs = np.concatenate([outputs_lip, outputs_box], axis = 0)
#print("Shape of the simulated set:", sim_inputs.shape)
#print("Shape of the simulated set:", sim_outputs.shape)
np.save('sim_type_inputs.npy', sim_inputs)
np.save('sim_type_outputs.npy', sim_outputs)

""" The simulated files are curated and the same training methodology will be used as shown in org_model_train_eval.py """


