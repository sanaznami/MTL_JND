import tensorflow as tf
import numpy as np
import glob
import os
import cv2
import argparse
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Lambda, Add, multiply, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate, LeakyReLU, Reshape, Permute, Conv2D, Conv2DTranspose, MaxPooling2D, Flatten, Dense, Activation, Input, BatchNormalization, Dropout  
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras import metrics
from PIL import Image, ImageOps
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.preprocessing.image import load_img, array_to_img
from scipy import misc
from tensorflow.keras import backend as K

###################################################################################
def load_data(data_dir, base_command, jnd_value):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')
    jnd_train_dir = os.path.join(data_dir, '{}train'.format(jnd_value.lower()))
    jnd_valid_dir = os.path.join(data_dir, '{}valid'.format(jnd_value.lower()))
    jnd_test_dir = os.path.join(data_dir, '{}test'.format(jnd_value.lower()))

    # Load and preprocess data
    def load_images_and_labels(folder_path, target_shape=(352, 640, 3)):
        images = []
        labels = []
        for subdir, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    # Load label from the text file
                    label_path = os.path.join(subdir, file)
                    labels = np.loadtxt(label_path)
                elif file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.bmp'):
                    # Load image
                    image_path = os.path.join(subdir, file)
                    image = cv2.imread(image_path)
                    # Resize the image to the target shape
                    image_resized = cv2.resize(image, (target_shape[1], target_shape[0]))
                    images.append(image_resized)
        return images, labels

    if base_command == 'train':
        train_images, train_labels = load_images_and_labels(train_dir)
        valid_images, valid_labels = load_images_and_labels(valid_dir)
        jnd_train_images, _ = load_images_and_labels(jnd_train_dir)  # Load images 
        jnd_valid_images, _ = load_images_and_labels(jnd_valid_dir)  # Load images 
        
        # Convert the lists of images and labels to numpy arrays
        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        valid_images = np.array(valid_images)
        valid_labels = np.array(valid_labels)
        jnd_train_images = np.array(jnd_train_images)
        jnd_valid_images = np.array(jnd_valid_images)
        
        return train_images, train_labels, valid_images, valid_labels, jnd_train_images, jnd_valid_images
    
    elif base_command == 'test':
        test_images, test_labels = load_images_and_labels(test_dir)
        jnd_test_images, _ = load_images_and_labels(jnd_test_dir)  # Load images 
        
        # Convert the lists of images and labels to numpy arrays
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)
        jnd_test_images = np.array(jnd_test_images)
        
        return test_images, test_labels, jnd_test_images

########################################################################################################
"""
Contains the implementation of Squeeze-and-Excitation Networks paper.
"""

def attach_attention_module(net, attention_module):
  if attention_module == 'se_block': # SE_block
    net = se_block(net)
  elif attention_module == 'cbam_block': # CBAM_block
    net = cbam_block(net)
  else:
    raise Exception("'{}' is not supported attention module!".format(attention_module))

  return net
  
def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature.shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature
    
def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature
    
def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature.shape[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool.shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool.shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool.shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])
    
def spatial_attention(input_feature):
	kernel_size = 7
	
	if K.image_data_format() == "channels_first":
		channel = input_feature.shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature.shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool.shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool.shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat.shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	assert cbam_feature.shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

###################################################################################

def ssim_metric(y_true, y_pred):
    return tf.image.ssim(y_true, y_pred, max_val=255.0)
    
def psnr_metric(y_true, y_pred):
    return tf.image.psnr(y_true, y_pred, max_val=255.0)
    
###################################################################################

# Define the function for creating a single encoder-decoder block
def create_encoder_decoder(input_layer, num_filters, latent_dim):
    # Encoder layers
    x = Conv2D(num_filters, (5, 5), strides=(2, 2), padding='same')(input_layer)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv2D(num_filters, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv2D(num_filters, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.05)(x)
    encoded = Conv2D(latent_dim, (5, 5), strides=(2, 2), padding='same')(x)
    
    # Decoder layers
    x = Conv2DTranspose(num_filters, (5, 5), strides=(2, 2), padding='same')(encoded)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv2DTranspose(num_filters, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = Conv2DTranspose(num_filters, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.05)(x)
    decoded = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(x)
    
    return Model(input_layer, decoded)

# Define the LAT block
def LATBlock(input_layer, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = channel_attention(x)
    
    return x

# Define the main function for creating the entire model
def create_model(ImgReconstrution_Model_Path):
    # Input layer
    input_img = Input(shape=(352, 640, 3))
    
    # Create encoder-decoder blocks for each branch
    ModelLAT = create_encoder_decoder(input_img, num_filters=128, latent_dim=192)
    
    # Load weights and freeze layers
    ModelName = 'ImgReconstruction.h5'
    ModelLAT.load_weights(os.path.join(ImgReconstrution_Model_Path, ModelName))
    
    for layer in ModelLAT.layers[0:]:
        layer.trainable = True
            
    ImgRecon = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', name='JNDImg')(ModelLAT.layers[13].output)
    LATJND = ModelLAT.layers[7].output

    LATJND = BatchNormalization()(LATJND)
    
    # Apply LATBlock
    LATJND = LATBlock(LATJND, num_filters=64)
    
    # Further processing and architecture
    SecondLATBlock = LATBlock(LATJND, num_filters=128)
    MP1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(SecondLATBlock)
    
    ThirdLATBlock = LATBlock(MP1, num_filters=256)
    
    ForthLATBlock = LATBlock(ThirdLATBlock, num_filters=512)
    MP3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(ForthLATBlock)
    
    Flat = Flatten()(MP3)
    
    jnd_out = Dense(256, activation='relu')(Flat)
    jnd_out = Dense(128, activation='relu')(jnd_out)
    jnd_out = Dense(1, activation='relu', name='JND_output')(jnd_out)
    
    # Create the final model that outputs the three JND values
    LAT = Model(inputs=input_img, outputs=[ImgRecon, jnd_out])
    return LAT

##################################################################################################################
def train_model(LAT, X_Train, train_labels, X_Valid, valid_labels, jnd_train_images, jnd_valid_images, checkpoint_path, csv_log_path, learning_rate, batch_size, epochs, jnd_value):
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    LAT.compile(optimizer=optimizer, loss={'JNDImg': 'mean_squared_error' , 'JND_output': 'mean_absolute_error'}, metrics={'JNDImg': [psnr_metric, ssim_metric, tf.keras.metrics.MeanAbsoluteError()]}, loss_weights={'JNDImg': 0.01, 'JND_output': 0.99})

    checkpoint_filename = 'BestModel{}.h5'.format(jnd_value)
    checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoint_path, checkpoint_filename),
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')

    csv_log_filename = 'logs{}.csv'.format(jnd_value)
    csv_logger = CSVLogger(os.path.join(csv_log_path, csv_log_filename), append=True, separator=';')


    jnd_index = int(jnd_value[-1]) - 1  # Extract the index of the selected JND value

    history = LAT.fit(X_Train,
                      {'JNDImg': jnd_train_images, 'JND_output': train_labels[:, jnd_index]},
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(X_Valid, {'JNDImg': jnd_valid_images, 'JND_output': valid_labels[:, jnd_index]}),
                      callbacks=[checkpoint, csv_logger],
                      shuffle=True)

##################################################################################################################

def test_model(LAT, X_Test, test_labels, jnd_test_images, result_path, model_weights_path, jnd_value):

    optimizer = keras.optimizers.Adam(lr=0.00001)
    LAT.compile(optimizer=optimizer, loss={'JNDImg': 'mean_squared_error' , 'JND_output': 'mean_absolute_error'}, metrics={'JNDImg': [psnr_metric, ssim_metric, tf.keras.metrics.MeanAbsoluteError()]}, loss_weights={'JNDImg': 0.01, 'JND_output': 0.99})
    
    # Test the model and save results
    jnd_index = int(jnd_value[-1]) - 1
    model_weights_file = os.path.join(model_weights_path, f'E2ELAT{jnd_value}.h5')
    LAT.load_weights(model_weights_file)

    results = LAT.predict(X_Test)
    jnd_out_values = results[1]  # Extract the 'jnd_out' values

    save_test_results(jnd_out_values, result_path, jnd_value)


def save_test_results(results, result_path, jnd_value):
    results_array = np.array(results)

    test_results_filename = '{}test_results.csv'.format(jnd_value)
    test_results_path = os.path.join(result_path, test_results_filename)
    np.savetxt(test_results_path, results_array, delimiter=',')

##################################################################################################################

def main(base_command, jnd_value, data_dir, checkpoint_path, csv_log_path, result_path, learning_rate, batch_size, epochs, model_weights_path, ImgReconstrution_Model_Path):
    
    if base_command == 'train':
        train_images, train_labels, valid_images, valid_labels, jnd_train_images, jnd_valid_images = load_data(data_dir, base_command, jnd_value)
        LAT = create_model(ImgReconstrution_Model_Path)
        
        train_model(LAT, train_images, train_labels, valid_images, valid_labels, jnd_train_images, jnd_valid_images, checkpoint_path, csv_log_path, learning_rate, batch_size, epochs, jnd_value)
            
    elif base_command == 'test':
        test_images, test_labels, jnd_test_images = load_data(data_dir, base_command, jnd_value)
        LAT = create_model(ImgReconstrution_Model_Path)
        
        test_model(LAT, test_images, test_labels, jnd_test_images, result_path, model_weights_path, jnd_value)
            
    else:
        print("Invalid base command. Please use 'train' or 'test' followed by JND values.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the LAT model.')
    parser.add_argument('base_command', choices=['train', 'test'], help='Base command for either training or testing.')
    parser.add_argument('--jnd_value', choices=['JND1', 'JND2', 'JND3'], help='JND value to predict')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the folder containing train, valid, and test subfolders.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to save checkpoints during training.')
    parser.add_argument('--csv_log_path', type=str, help='Path to save CSV logs during training.')
    parser.add_argument('--result_path', type=str, help='Path to save test results.')
    parser.add_argument('--model_weights_path', type=str, help='Path to the pre-trained model for testing.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--ImgReconstrution_Model_Path', type=str, help='Path to the pre-trained JND Reconstruction models')

    args = parser.parse_args()
    main(args.base_command, args.jnd_value, args.data_dir, args.checkpoint_path, args.csv_log_path, args.result_path, args.learning_rate, args.batch_size, args.epochs, args.model_weights_path, args.ImgReconstrution_Model_Path)
    
