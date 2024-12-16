"""
In this section, we will combine our satellite data with simulated data
"""

import numpy as np

# Load input simulated and satellite data
sim_train_inputs = np.load('sim_train_inputs.npy')
sat_train_inputs = np.load('sat_train_inputs.npy')
sim_valid_inputs = np.load('sim_valid_inputs.npy')
sat_valid_inputs = np.load('sat_valid_inputs.npy')
sim_test_inputs = np.load('sim_test_inputs.npy')
sat_test_inputs = np.load('sat_test_inputs.npy')

comb_train_inputs = np.concatenate([sim_train_inputs, sat_train_inputs], axis = 0)
comb_valid_inputs = np.concatenate([sim_valid_inputs, sat_valid_inputs], axis = 0)
comb_test_inputs = np.concatenate([sim_test_inputs, sat_test_inputs], axis = 0)

np.save('comb_train_inputs.npy', comb_train_inputs)
np.save('comb_valid_inputs.npy', comb_valid_inputs)
np.save('comb_test_inputs.npy', comb_test_inputs)

# Load output simulated and satellite data
sim_train_outputs = np.load('sim_train_outputs.npy')
sat_train_outputs = np.load('sat_train_outputs.npy')
sim_valid_outputs = np.load('sim_valid_outputs.npy')
sat_valid_outputs = np.load('sat_valid_outputs.npy')
sim_test_outputs = np.load('sim_test_outputs.npy')
sat_test_outputs = np.load('sat_test_outputs.npy')

comb_train_outputs = np.concatenate([sim_train_outputs, sat_train_outputs], axis = 0)
comb_valid_outputs = np.concatenate([sim_valid_outputs, sat_valid_outputs], axis = 0)
comb_test_outputs = np.concatenate([sim_test_outputs, sat_test_outputs], axis = 0)

np.save('comb_train_outputs.npy', comb_train_outputs)
np.save('comb_valid_outputs.npy', comb_valid_outputs)
np.save('comb_test_outputs.npy', comb_test_outputs)

