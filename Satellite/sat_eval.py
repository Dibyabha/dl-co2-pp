""""
This section deals with the training and inference of the satellite data
"""

# Necessary packages
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, BatchNormalization, MaxPool2D, Flatten, Dense, GaussianNoise, Concatenate, ConvTranspose2D, Upsampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

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

# Load all files
train_inputs = np.load('sat_train_inputs.npy')
train_outputs = np.load('sat_train_outputs.npy')
val_inputs = np.load('sat_valid_inputs.npy')
val_outputs = np.load('sat_valid_outputs.npy')
test_inputs = np.load('sat_test_inputs.npy')
test_outputs = np.load('sat_test_outputs.npy')

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

model = cnn_mod(input_shape = (64, 64, 4)) # Declaring the model as cnn model. If you want to use the unet model, change it to unet_mod(input_shape = (64, 64, 4), dropout_rate = 0.2)
# model.load_weights('saved_models/filepath.h5') # If we have already stored the weights, then uncomment this line to load
optimizer = Adam(learning_rate = 1e-3) # Taking Adam optimizer with learning rate as 1e-3

# We will update our learning rate if we reach a plateau. 
# For this we used ReduceLROnPlateau which will monitor val_loss.
learning_callback = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.5, patience = 20, verbose = 1, min_delta = 5e-3, cooldown = 0, min_lr = 5e-5)

# Along with Mean Absolute Error (mae) as the loss function we have also used Mean Absolute Percentage Error (mape), Mean Squared Error (mse)
model.compile(optimizer = optimizer, loss = 'mae')
model.summary()

history = model.fit(data_augmentation.flow(train_inputs, train_outputs, batch_size = 32, shuffle = True), epochs = 500, steps_per_epoch = len(train_inputs)//32,
validation_data = (val_inputs, val_outputs), validation_batch_size = 32, validation_steps = None, callbacks = [learning_callback, checkpoint_callback])

# Model's performance on test set
test_loss = model.evaluate(test_inputs, test_outputs, batch_size = None) # Uncomment these lines to infer model's performance.
print(f'Test Loss: {test_loss}')

# To store the true emissions and predicted emissions in a .csv format.
true_emissions = []
predicted_emissions = []
for i in range(len(test_inputs)):
    inputs = np.expand_dims(test_inputs[i], axis = 0)
    outputs = model.predict(inputs)
    true_emissions.append(test_outputs[i])
    predicted_emissions.append(outputs)

# Converting into numpy arrays
true_emissions = np.array(true_emissions) 
predicted_emissions = np.array(predicted_emissions)
true_emissions = true_emissions.reshape(predicted_emissions.shape) 

# Storing the true emissions and predicted emissions in a csv format. 
# For different loss function, there will be different evaluations, so make sure to store them in the same data structure
df = pd.DataFrame({'True Emissions': true_emissions.flatten(),'Predicted Emissions': predicted_emissions.flatten()}, index = np.arange(1, len(true_emissions) + 1))
df.to_csv('file.csv') # change file to your own desired name
