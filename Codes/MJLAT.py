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

def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

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

    train_images, train_labels = load_images_and_labels(train_dir)
    valid_images, valid_labels = load_images_and_labels(valid_dir)
    test_images, test_labels = load_images_and_labels(test_dir)

    # Convert the lists of images and labels to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    valid_images = np.array(valid_images)
    valid_labels = np.array(valid_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

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
def create_model(JND_Recon_Models_Path):
    # Input layer
    input_img = Input(shape=(352, 640, 3))
    
    # Create encoder-decoder blocks for each branch
    ModelLAT1 = create_encoder_decoder(input_img, num_filters=128, latent_dim=192)
    ModelLAT2 = create_encoder_decoder(input_img, num_filters=128, latent_dim=192)
    ModelLAT3 = create_encoder_decoder(input_img, num_filters=128, latent_dim=192)
    
    # Load weights and freeze layers
    ModelLAT1.load_weights(os.path.join(JND_Recon_Models_Path, 'JND1Reconstruction.h5'))
    ModelLAT2.load_weights(os.path.join(JND_Recon_Models_Path, 'JND2Reconstruction.h5'))
    ModelLAT3.load_weights(os.path.join(JND_Recon_Models_Path, 'JND3Reconstruction.h5'))
    
    for branch in [ModelLAT1, ModelLAT2, ModelLAT3]:
        for layer in branch.layers[0:]:
            layer.trainable = False
    
    # Extract LAT features
    LATJND1 = ModelLAT1.layers[7].output
    LATJND2 = ModelLAT2.layers[7].output
    LATJND3 = ModelLAT3.layers[7].output
    
    LATJND1 = BatchNormalization()(LATJND1)
    LATJND2 = BatchNormalization()(LATJND2)
    LATJND3 = BatchNormalization()(LATJND3)
    
    # Apply LATBlock
    LATJND1 = LATBlock(LATJND1, num_filters=64)
    LATJND2 = LATBlock(LATJND2, num_filters=64)
    LATJND3 = LATBlock(LATJND3, num_filters=64)
    
    # Combine outputs
    combined = multiply([LATJND1, LATJND2, LATJND3])
    
    # Further processing and architecture
    SecondLATBlock = LATBlock(combined, num_filters=128)
    MP1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(SecondLATBlock)
    
    ThirdLATBlock = LATBlock(MP1, num_filters=256)
    
    ForthLATBlock = LATBlock(ThirdLATBlock, num_filters=512)
    MP3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(ForthLATBlock)
    
    Flat = Flatten()(MP3)
    
    jnd_output_1 = Dense(256, activation='relu')(Flat)
    jnd_output_1 = Dense(128, activation='relu')(jnd_output_1)
    jnd_output_1 = Dense(1, activation='relu', name='JND1_output')(jnd_output_1)
    
    jnd_output_2 = Dense(256, activation='relu')(Flat)
    jnd_output_2 = Dense(128, activation='relu')(jnd_output_2)
    jnd_output_2 = Dense(1, activation='relu', name='JND2_output')(jnd_output_2)
    
    jnd_output_3 = Dense(256, activation='relu')(Flat)
    jnd_output_3 = Dense(128, activation='relu')(jnd_output_3)
    jnd_output_3 = Dense(1, activation='relu', name='JND3_output')(jnd_output_3)
    
    # Create the final model that outputs the three JND values
    MJLAT = Model(inputs=input_img, outputs=[jnd_output_1, jnd_output_2, jnd_output_3])
    return MJLAT

##################################################################################################################

def train_model(MJLAT, X_Train, train_labels, X_Valid, valid_labels, checkpoint_path, csv_log_path, learning_rate, batch_size, epochs):
    # Compile and train the model
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    MJLAT.compile(optimizer=optimizer,
                    loss={'JND1_output': 'mean_absolute_error', 'JND2_output': 'mean_absolute_error', 'JND3_output': 'mean_absolute_error'},
                    loss_weights={'JND1_output': 1., 'JND2_output': 1., 'JND3_output': 1.})

    checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoint_path, 'BestModel.h5'),
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')
                             
    csv_logger = CSVLogger(os.path.join(csv_log_path, 'logs.csv'), append=True, separator=';')


    history = MJLAT.fit(X_Train,
                           {'JND1_output': train_labels[:,0], 'JND2_output': train_labels[:,1], 'JND3_output': train_labels[:,2]},
                           epochs=epochs,
                           batch_size=batch_size,
                           validation_data=(X_Valid, {'JND1_output': valid_labels[:,0], 'JND2_output': valid_labels[:,1], 'JND3_output': valid_labels[:,2]}),
                           callbacks=[checkpoint, csv_logger],
                           shuffle=True)


##################################################################################################################

def test_model(MJLAT, X_Test, test_labels, result_path, model_weights_path):
    optimizer = keras.optimizers.Adam(lr=0.00001)
    MJLAT.compile(optimizer=optimizer,
                    loss={'JND1_output': 'mean_absolute_error', 'JND2_output': 'mean_absolute_error', 'JND3_output': 'mean_absolute_error'},
                    loss_weights={'JND1_output': 1., 'JND2_output': 1., 'JND3_output': 1.})
    # Test the model and save results
    model_weights_file = os.path.join(model_weights_path, 'MJLAT.h5')
    MJLAT.load_weights(model_weights_file)

    results = MJLAT.predict(X_Test)
    print(results)

    save_test_results(results, result_path)

def save_test_results(results, result_path):
    results_array = np.array(results)
    flattened_results = results_array.reshape(results_array.shape[0], -1).T

    test_results_path = os.path.join(result_path, 'test_results.csv')
    np.savetxt(test_results_path, flattened_results, delimiter=',')
##################################################################################################################

def main(base_command, data_dir, checkpoint_path, csv_log_path, result_path, learning_rate, batch_size, epochs, model_weights_path, JND_Recon_Models_Path):
    X_Train, train_labels, X_Valid, valid_labels, X_Test, test_labels = load_data(data_dir)
    MJLAT = create_model(JND_Recon_Models_Path)

    if base_command == 'train':
        train_model(MJLAT, X_Train, train_labels, X_Valid, valid_labels, checkpoint_path, csv_log_path, learning_rate, batch_size, epochs)
    elif base_command == 'test':
        test_model(MJLAT, X_Test, test_labels, result_path, model_weights_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the MJLAT model.')
    parser.add_argument('base_command', choices=['train', 'test'], help='Base command for either training or testing.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the folder containing train, valid, and test subfolders.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to save checkpoints during training.')
    parser.add_argument('--csv_log_path', type=str, help='Path to save CSV logs during training.')
    parser.add_argument('--result_path', type=str, help='Path to save test results.')
    parser.add_argument('--model_weights_path', type=str, help='Path to the pre-trained model for testing.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    parser.add_argument('--JND_Recon_Models_Path', required=True, type=str, help='Path to the pre-trained JND Reconstruction models')


    args = parser.parse_args()
    main(args.base_command, args.data_dir, args.checkpoint_path, args.csv_log_path, args.result_path, args.learning_rate, args.batch_size, args.epochs, args.model_weights_path, args.JND_Recon_Models_Path)


































#
#
#def create_model():
#    input_img = Input(shape=(352, 640, 3))
#    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(input_img)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    encoded1 = Conv2D(192, (5, 5), strides=(2, 2), padding='same')(x)
#    
#    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(encoded1)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    decoded1 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(x)
#    
#    LATJND1 = Model(input_img, decoded1)   
#    
#    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(input_img)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    encoded2 = Conv2D(192, (5, 5), strides=(2, 2), padding='same')(x)
#    
#    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(encoded2)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    decoded2 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(x)
#    
#    LATJND2 = Model(input_img, decoded2)
#    
#    
#    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(input_img)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    encoded3 = Conv2D(192, (5, 5), strides=(2, 2), padding='same')(x)
#    
#    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(encoded3)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#    x = LeakyReLU(alpha=0.05)(x)
#    decoded3 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(x)
#    
#    LATJND3 = Model(input_img, decoded3)
#        
#    LATJND1.load_weights('FTIRVS.h5')   
#    LATJND2.load_weights('FTIRVSJ2.h5')   
#    LATJND3.load_weights('FTIRVSJ3.h5')
#    
#    
#    for layer in LATJND1.layers[0:]:
#        layer.trainable = False
#    for layer in LATJND2.layers[0:]:
#        layer.trainable = False
#    for layer in LATJND3.layers[0:]:
#        layer.trainable = False
#        
#    JNDLS1 = LATJND1.layers[7].output
#    JNDLS2 = LATJND2.layers[7].output
#    JNDLS3 = LATJND3.layers[7].output
#    
#    
#    JNDLS1 = keras.layers.BatchNormalization()(JNDLS1)
#    
#    JNDLS1 = Conv2D(64,(3,3),padding="same")(JNDLS1)
#    JNDLS1 = keras.layers.BatchNormalization()(JNDLS1)
#    JNDLS1 = keras.activations.relu(JNDLS1)
#    JNDLS1 = channel_attention(JNDLS1)
#    
#    
#    
#    JNDLS2 = keras.layers.BatchNormalization()(JNDLS2)
#    
#    JNDLS2 = Conv2D(64,(3,3),padding="same")(JNDLS2)
#    JNDLS2 = keras.layers.BatchNormalization()(JNDLS2)
#    JNDLS2 = keras.activations.relu(JNDLS2)
#    JNDLS2 = channel_attention(JNDLS2)
#    
#    
#    
#    JNDLS3 = keras.layers.BatchNormalization()(JNDLS3)
#    
#    JNDLS3 = Conv2D(64,(3,3),padding="same")(JNDLS3)
#    JNDLS3 = keras.layers.BatchNormalization()(JNDLS3)
#    JNDLS3 = keras.activations.relu(JNDLS3)
#    JNDLS3 = channel_attention(JNDLS3)
#    
#    
#    
#    J3L = multiply([JNDLS1, JNDLS2, JNDLS3])
#    
#    JNDLS = Conv2D(128,(3,3),padding="same")(J3L)
#    JNDLS = keras.layers.BatchNormalization()(JNDLS)
#    JNDLS = keras.activations.relu(JNDLS)
#    JNDLS = channel_attention(JNDLS)
#    JNDLS = MaxPooling2D(pool_size=(2,2),strides=(2,2))(JNDLS)
#    
#    JNDLS = Conv2D(256,(3,3),padding="same")(JNDLS)
#    JNDLS = keras.layers.BatchNormalization()(JNDLS)
#    JNDLS = keras.activations.relu(JNDLS)
#    JNDLS = channel_attention(JNDLS)
#    
#    JNDLS = Conv2D(512,(3,3),padding="same")(JNDLS)
#    JNDLS = keras.layers.BatchNormalization()(JNDLS)
#    JNDLS = keras.activations.relu(JNDLS)
#    JNDLS = channel_attention(JNDLS)
#    JNDLS = MaxPooling2D(pool_size=(2,2),strides=(2,2))(JNDLS)
#    
#    JNDLS = Flatten()(JNDLS)
#    
#    JND1FC = Dense(256, activation='relu')(JNDLS)
#    JND1FC = Dense(128, activation='relu')(JND1FC)
#    JND1FC = Dense(1, activation='relu', name='JND1')(JND1FC)
#    
#    JND2FC = Dense(256, activation='relu')(JNDLS)
#    JND2FC = Dense(128, activation='relu')(JND2FC)
#    JND2FC = Dense(1, activation='relu', name='JND2')(JND2FC)
#    
#    JND3FC = Dense(256, activation='relu')(JNDLS)
#    JND3FC = Dense(128, activation='relu')(JND3FC)
#    JND3FC = Dense(1, activation='relu', name='JND3')(JND3FC)
#    
#    MJLATModel = Model(inputs=input_img,outputs=[JND1FC, JND2FC, JND3FC])
#    return MJLATModel
#    
#
#
#input_img = Input(shape=(352, 640, 3))
#x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(input_img)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#encoded1 = Conv2D(192, (5, 5), strides=(2, 2), padding='same')(x)
#
#x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(encoded1)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#decoded1 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(x)
#
#model1 = Model(input_img, decoded1)   
#
#x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(input_img)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#encoded2 = Conv2D(192, (5, 5), strides=(2, 2), padding='same')(x)
#
#x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(encoded2)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#decoded2 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(x)
#
#model2 = Model(input_img, decoded2)
#
#
#x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(input_img)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#encoded3 = Conv2D(192, (5, 5), strides=(2, 2), padding='same')(x)
#
#x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(encoded3)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#x = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same')(x)
#x = LeakyReLU(alpha=0.05)(x)
#decoded3 = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same')(x)
#
#model3 = Model(input_img, decoded3)
#    
#model1.load_weights('FTIRVS.h5')   
#model2.load_weights('FTIRVSJ2.h5')   
#model3.load_weights('FTIRVSJ3.h5')
#
#
#for layer in model1.layers[0:]:
#    layer.trainable = False
#for layer in model2.layers[0:]:
#    layer.trainable = False
#for layer in model3.layers[0:]:
#    layer.trainable = False
#    
#JNDLS1 = model1.layers[7].output
#JNDLS2 = model2.layers[7].output
#JNDLS3 = model3.layers[7].output
#
#
#JNDLS1 = keras.layers.BatchNormalization()(JNDLS1)
#
#JNDLS1 = Conv2D(64,(3,3),padding="same")(JNDLS1)
#JNDLS1 = keras.layers.BatchNormalization()(JNDLS1)
#JNDLS1 = keras.activations.relu(JNDLS1)
#JNDLS1 = channel_attention(JNDLS1)
#
#
#
#JNDLS2 = keras.layers.BatchNormalization()(JNDLS2)
#
#JNDLS2 = Conv2D(64,(3,3),padding="same")(JNDLS2)
#JNDLS2 = keras.layers.BatchNormalization()(JNDLS2)
#JNDLS2 = keras.activations.relu(JNDLS2)
#JNDLS2 = channel_attention(JNDLS2)
#
#
#
#JNDLS3 = keras.layers.BatchNormalization()(JNDLS3)
#
#JNDLS3 = Conv2D(64,(3,3),padding="same")(JNDLS3)
#JNDLS3 = keras.layers.BatchNormalization()(JNDLS3)
#JNDLS3 = keras.activations.relu(JNDLS3)
#JNDLS3 = channel_attention(JNDLS3)
#
#
#
#J3L = multiply([JNDLS1, JNDLS2, JNDLS3])
#
#JNDLS = Conv2D(128,(3,3),padding="same")(J3L)
#JNDLS = keras.layers.BatchNormalization()(JNDLS)
#JNDLS = keras.activations.relu(JNDLS)
#JNDLS = channel_attention(JNDLS)
#JNDLS = MaxPooling2D(pool_size=(2,2),strides=(2,2))(JNDLS)
#
#JNDLS = Conv2D(256,(3,3),padding="same")(JNDLS)
#JNDLS = keras.layers.BatchNormalization()(JNDLS)
#JNDLS = keras.activations.relu(JNDLS)
#JNDLS = channel_attention(JNDLS)
#
#JNDLS = Conv2D(512,(3,3),padding="same")(JNDLS)
#JNDLS = keras.layers.BatchNormalization()(JNDLS)
#JNDLS = keras.activations.relu(JNDLS)
#JNDLS = channel_attention(JNDLS)
#JNDLS = MaxPooling2D(pool_size=(2,2),strides=(2,2))(JNDLS)
#
#JNDLS = Flatten()(JNDLS)
#
#JND1TC = Dense(256, activation='relu')(JNDLS)
#JND1TC = Dense(128, activation='relu')(JND1TC)
#JND1TC = Dense(1, activation='relu', name='JND1')(JND1TC)
#
#JND2TC = Dense(256, activation='relu')(JNDLS)
#JND2TC = Dense(128, activation='relu')(JND2TC)
#JND2TC = Dense(1, activation='relu', name='JND2')(JND2TC)
#
#JND3TC = Dense(256, activation='relu')(JNDLS)
#JND3TC = Dense(128, activation='relu')(JND3TC)
#JND3TC = Dense(1, activation='relu', name='JND3')(JND3TC)
#
#JNDTCVS = Model(inputs=input_img,outputs=[JND1TC, JND2TC, JND3TC])
#JNDTCVS.summary()
#    
#checkpoint = ModelCheckpoint(filepath='MJLAT.h5',
#                                 monitor='val_loss',
#								 verbose=1,
#                                 save_best_only=False,
#                                 save_weights_only=True,
#                                 period=10)
#
#csv_logger = CSVLogger('/lustre/sgn-data/Sanaz/Results/TCSVTDataset/MT3LJND/Latent/3LatentSpace/V19/MM/MT_TCV19.csv', append=True, separator=';')
#
#opt = keras.optimizers.Adam(lr=0.00001)
#JNDTCVS.compile(optimizer=opt, loss={'JND1': 'mean_absolute_error', 'JND2': 'mean_absolute_error', 'JND3': 'mean_absolute_error'}, loss_weights={'JND1': 1., 'JND2': 1., 'JND3': 1.})
#
#
#history=JNDTCVS.fit(X_Train,
#            {'JND1': trainlabel1, 'JND2': trainlabel2, 'JND3': trainlabel3},
#            epochs=300,
#            batch_size=8,
#            validation_data=(X_Valid, {'JND1': validlabel1, 'JND2': validlabel2, 'JND3': validlabel3}),
#            callbacks=[checkpoint, csv_logger],
#            shuffle=True)
#
#
####################################################################################
#