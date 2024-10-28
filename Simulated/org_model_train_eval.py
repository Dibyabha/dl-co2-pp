"""
We will start by replicating the results that are obtained by J. D. Brazidec et al. on 'Deep learning
applied to CO2 power plant emissions quantification using simulated satellite images'. A point for all
readers, we did not run the model for all existing cases. Rather our focus was only on the replication
of the results where the authors of the paper considered XCO2, NO2, u and v. We did not consider other
combinations and kept all the existing parameters same. This file is on the replication of the results
from the Lippendorf or Boxberg dataset.
"""

""" We will start by importing the necessary packages """

import h5py
import os
import numpy as np
from netCDF4 import Dataset as NetCDFDataset
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, Dropout, BatchNormalization, MaxPool2D, Flatten, Dense, GaussianNoise
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
import tensorflow as tf

""" 
Custom dataset to take the XCO2, NO2, u and v data and normalize them before using and analyzing.
Shapes of each variable are (64, 64) and after stacking the input shape is (64, 64, 4). A note to
all readers, before stacking we normalized all the variables. As for the output varibale (emiss),
we only considered the middle value and hence extracted the middle value and rounded it off to 3
decimal places.
""" 
class CustomDataset:
    
    def __init__(self, file_path, variable = None):
        self.ncfile = NetCDFDataset(file_path, 'r')                                     
        self.variable = variable                                                        
    
    def __len__(self):
        return self.ncfile.variables['xco2_noisy'].shape[0]
    
    def __getitem__(self, idx):
        xco2_read = self.ncfile.variables['xco2_noisy'][idx] 
        u_read = self.ncfile.variables['u'][idx] 
        v_read = self.ncfile.variables['v'][idx] 
        emiss_read = self.ncfile.variables['emiss'][idx] 
        xco2_arr = np.array(xco2_read.data).astype('float32') 
        u_arr = np.array(u_read.data).astype('float32')
        v_arr = np.array(v_read.data).astype('float32')
        emiss_arr = np.array(emiss_read.data).astype('float32')
        
        if self.variable:
            var_read = self.ncfile.variables[self.variable][idx] 
            var_arr = np.array(var_read.data).astype('float32')
            inputs = np.stack([xco2_arr, u_arr, v_arr, var_arr], axis = -1)
        else: 
            inputs = np.stack([xco2_arr, u_arr, v_arr], axis = -1) 
        
        mean = inputs.mean(axis = (0, 1), keepdims = True)
        std = inputs.std(axis = (0, 1), keepdims = True) 
        std[std==0] = 1 
        inputs = (inputs - mean)/std 
        weights = np.array([0.0, 1.0, 0.0]) 
        weighted = np.round(np.sum(weights * emiss_arr), 3)
        outputs = np.array([weighted], dtype = np.float32)
        
        return inputs, outputs 

    def __del__(self):
        self.ncfile.close() 

"""
The CNN model architecure where the convolution operation has a kernel of (3, 3) with activation fn
as Exponential Linear Unit (elu) and stride as 1 with padding = 0. Also, batchnorm and dropout are
added with maxpool operations and a fully connected layer with leaky relu activation.
"""
def cnn_mod(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation = "elu", strides = 1, input_shape = input_shape)) 
    model.add(Dropout(0.1))
    model.add(Conv2D(32, (3, 3), activation = "elu", strides = 1)) 
    model.add(MaxPool2D(pool_size = (2, 2), padding = "valid", strides = 2))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation = "elu", strides = 1))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation = "elu", strides = 1)) 
    model.add(BatchNormalization()) 
    model.add(Conv2D(64, (3, 3), activation = "elu", strides = 1)) 
    model.add(Dropout(0.2)) 
    model.add(Conv2D(64, (3, 3), activation = "elu", strides = 1)) 
    model.add(MaxPool2D(pool_size = (2, 2), padding = "valid", strides = 2)) 
    model.add(Conv2D(64, (3, 3), activation = "elu", strides = 1)) 
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation = "elu", strides = 1))
    model.add(MaxPool2D(pool_size = (2, 2), padding = "valid", strides = 2)) 
    model.add(Flatten()) 
    model.add(Dense(1)) 
    model.add(LeakyReLU(alpha = 0.3))
    
    return model

"""
The U-Net regression model architecture which progressively doubles the feature maps
starting from 64 to 512 at the bottleneck layer. Using droput of 0.2 and activation
fn as Rectified Linear Unit (ReLU), the output from final decoder layer is taken in a
fullt connected layer to predict the emission rate.
"""
def unet_mod(input_shape, dropout_rate = 0.2):
    inputs = Input(input_shape)
    c1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(inputs) 
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(c1)
    c1 = BatchNormalization()(c1)
    d1 = Dropout(dropout_rate)(c1) 
    p1 = MaxPool2D(pool_size = (2, 2))(d1) 
    c2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(p1) 
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(c2)
    c2 = BatchNormalization()(c2)
    d2 = Dropout(dropout_rate)(c2)
    p2 = MaxPool2D(pool_size = (2, 2))(d2)
    c3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(p2) 
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(c3)
    c3 = BatchNormalization()(c3)
    d3 = Dropout(dropout_rate)(c3)
    p3 = MaxPool2D(pool_size = (2, 2))(d3) 
    c4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(p3) 
    c4 = BatchNormalization()(c4)
    c4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(c4) 
    c4 = BatchNormalization()(c4)
    d4 = Dropout(dropout_rate)(c4)
    u5 = Conv2DTranspose(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(d4) 
    u5 = UpSampling2D((2, 2))(u5) 
    concat5 = Concatenate()([u5, c3]) 
    c5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(concat5) 
    c5 = BatchNormalization()(c5)
    c5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(c5)
    c5 = BatchNormalization()(c5)
    d5 = Dropout(dropout_rate)(c5)
    u6 = Conv2DTranspose(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(d5)
    u6 = UpSampling2D((2, 2))(u6)
    concat6 = Concatenate()([u6, c2])
    c6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(concat6)
    c6 = BatchNormalization()(c6)
    c6 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(c6)
    c6 = BatchNormalization()(c6)
    d6 = Dropout(dropout_rate)(c6)
    u7 = Conv2DTranspose(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(d6) 
    u7 = UpSampling2D((2, 2))(u7)
    concat7 = Concatenate()([u7, c1]) 
    c7 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_uniform')(concat7)
    c7 = BatchNormalization()(c7)
    c7 = Conv2D(1, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(c7)
    c7 = BatchNormalization()(c7)
    
    flat = Flatten()(c7)
    output = Dense(1, activation = 'linear')(flat) 
    
    model = Model(inputs = inputs, outputs = output)
    
    return model
"""
Augmentation applied to only training data.
"""
data_augmentation = ImageDataGenerator(rotation_range = 180, width_shift_range = 0.0, height_shift_range = 0.0, shear_range = 90, zoom_range = 0.2, horizontal_flip = True, vertical_flip = True, fill_mode = 'nearest')

"""
Checkpoints
"""
checkpoint_dir = './saved_models/' 
os.makedirs(checkpoint_dir, exist_ok = True)
model_filepath = os.path.join(checkpoint_dir, 'filepath.h5') # instead of filepath put your own filepath to make sure you save the model weights
checkpoint_callback = ModelCheckpoint(filepath = model_filepath, save_weights_only = True, verbose = 1) # save the weights after every epoch

"""
Training and Evaluation
"""
# We saved the datasets from Lippendorf and Boxberg as name_type_dataset.nc where name will be lip or box
# and type will be valid, test or train. So if you want to run the model for Boxberg, just change the name
# to box and do not change the rest.
test_set = 'lip_test_dataset.nc'
valid_set = 'lip_valid_dataset.nc'
train_set = 'lip_train_dataset.nc'

# We have used the no2_noisy as the variable with XCO2, u and v
train_dataset = CustomDataset(train_set, variable = 'no2_noisy')
valid_dataset = CustomDataset(valid_set, variable = 'no2_noisy')
test_dataset = CustomDataset(test_set, variable = 'no2_noisy')

# We wanted to store the inputs and outputs and hence, we stored them as
# numpy arrays and fed them into the model
train_inputs = []
train_outputs = []
val_inputs = []
val_outputs = []
test_inputs = []
test_outputs = []

for inputs, outputs in train_dataset:
    train_inputs.append(inputs)
    train_outputs.append(outputs)
for inputs, outputs in valid_dataset:
    val_inputs.append(inputs)
    val_outputs.append(outputs)
for inputs, outputs in test_dataset:
    test_inputs.append(inputs)
    test_outputs.append(outputs)

train_inputs = np.array(train_inputs) # train_inputs shape : (25152, 64, 64, 4)
train_outputs = np.array(train_outputs) # train_outputs shape : (25152, 1)
val_inputs = np.array(val_inputs) # val_inputs shape : (4608, 64, 64, 4)
val_outputs = np.array(val_outputs) # val_outputs shape : (4608, 1)
test_inputs = np.array(test_inputs) # test_inputs shape : (6289, 64, 64, 4)
test_outputs = np.array(test_outputs) # test_outputs shape : (6289, 1)

model = cnn_mod(input_shape = (64, 64, 4)) # Declaring the model as cnn model. If you want to use the unet model, change it to unet_mod(input_shape = (64, 64, 4), dropout_rate = 0.2)
# model.load_weights('saved_models/filepath.h5') # If we have already stored the weights, then uncomment this line to load
optimizer = Adam(learning_rate = 1e-3) # Taking Adam optimizer with learning rate as 1e-3

# We will update our learning rate if we reach a plateau. 
# For this we used ReduceLROnPlateau which will monitor val_loss.
learning_callback = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 20, verbose = 1, min_delta = 5e-3, cooldown = 0, min_lr = 5e-5)

# instead of Mean Absolute Error (mae) as the loss function we can also use Mean Absolute Percentage Error (mape), Mean Squared Error (mse)
model.compile(optimizer = optimizer, loss = 'mae')

# Around 186000 trainable paramaters are there
model.summary()

# Hyper-parameters are taken from the paper and not changed
history = model.fit(data_augmentation.flow(train_inputs, train_outputs, batch_size = 32, shuffle = True), epochs = 500, steps_per_epoch = len(train_dataset)//32,
validation_data = (val_inputs, val_outputs), validation_batch_size = 32, validation_steps = None, callbacks = [learning_callback, checkpoint_callback])

# Model's performance on test set
test_loss = model.evaluate(test_inputs, test_outputs, batch_size = None) # Uncomment these lines to infer model's performance.
print(f'Test Loss: {test_loss}')

# To store the true emissions and predicted emissions in a .csv format.
true_emissions = []
predicted_emissions = []
for inputs, targets in test_dataset:
    inputs = np.expand_dims(inputs, axis = 0)
    outputs = model.predict(inputs) # Predicting the output based on the model's learning.
    print("True value : ", targets)
    print("Predicted value : ", outputs)
    true_emissions.append(targets) # Append the true emissions into true_emissions list
    predicted_emissions.append(outputs) # Append the predicted emissions into predicted_emissions list

# Converting into numpy arrays
true_emissions = np.array(true_emissions) # Shape : 6289, 3
predicted_emissions = np.array(predicted_emissions) # Shape : 6289, 1, 3
true_emissions = true_emissions.reshape(predicted_emissions.shape) # After reshaping : 6289, 1, 3

# Storing the true emissions and predicted emissions in a csv format. 
df = pd.DataFrame({'True Emissions': true_emissions.flatten(),'Predicted Emissions': predicted_emissions.flatten()}, index = np.arange(1, len(true_emissions) + 1))
df.to_csv('file.csv') # change file to your own desired name
