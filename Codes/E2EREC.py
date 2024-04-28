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

# Define REC block
def RECBlock(input_layer, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same")(input_layer)
    x = BatchNormalization()(x)
    x = keras.activations.relu(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    
    return x

# Define the main function for creating the entire model
def create_model(ImgReconstrution_Model_Path):
    # Input layer
    input_img = Input(shape=(352, 640, 3))
    
    # Create encoder-decoder blocks for each branch
    ModelREC = create_encoder_decoder(input_img, num_filters=128, latent_dim=192)
    
    # Load weights and freeze layers
    ModelName = 'best_model_MSEB8.h5'
    ModelREC.load_weights(os.path.join(ImgReconstrution_Model_Path, ModelName))
    
    for layer in ModelREC.layers[0:]:
        layer.trainable = True
            
    ImgRecon = Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', name='JNDImg')(ModelREC.layers[13].output)

    RECJND = BatchNormalization()(ImgRecon)
    
    # Apply RECBlock
    RECJND = RECBlock(RECJND, num_filters=512)
    RECJND = RECBlock(RECJND, num_filters=256)
    RECJND = RECBlock(RECJND, num_filters=128)
    RECJND = RECBlock(RECJND, num_filters=64)
    RECJND = RECBlock(RECJND, num_filters=32)
    
    Flat = Flatten()(RECJND)
    
    jnd_out = Dense(512, activation='relu')(Flat)
    jnd_out = Dense(1, activation='relu', name='JND_output')(jnd_out)
    
    # Create the final model that outputs the three JND values
    REC = Model(inputs=input_img, outputs=[ImgRecon, jnd_out])
    return REC

##################################################################################################################
def train_model(REC, X_Train, train_labels, X_Valid, valid_labels, jnd_train_images, jnd_valid_images, checkpoint_path, csv_log_path, learning_rate, batch_size, epochs, jnd_value):
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    REC.compile(optimizer=optimizer, loss={'JNDImg': 'mean_squared_error' , 'JND_output': 'mean_absolute_error'}, metrics={'JNDImg': [psnr_metric, ssim_metric, tf.keras.metrics.MeanAbsoluteError()]}, loss_weights={'JNDImg': 0.01, 'JND_output': 0.99})

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

    history = REC.fit(X_Train,
                      {'JNDImg': jnd_train_images, 'JND_output': train_labels[:, jnd_index]},
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(X_Valid, {'JNDImg': jnd_valid_images, 'JND_output': valid_labels[:, jnd_index]}),
                      callbacks=[checkpoint, csv_logger],
                      shuffle=True)

##################################################################################################################

def test_model(REC, X_Test, test_labels, jnd_test_images, result_path, model_weights_path, jnd_value):

    optimizer = keras.optimizers.Adam(lr=0.00001)
    REC.compile(optimizer=optimizer, loss={'JNDImg': 'mean_squared_error' , 'JND_output': 'mean_absolute_error'}, metrics={'JNDImg': [psnr_metric, ssim_metric, tf.keras.metrics.MeanAbsoluteError()]}, loss_weights={'JNDImg': 0.01, 'JND_output': 0.99})
    
    # Test the model and save results
    jnd_index = int(jnd_value[-1]) - 1
    model_weights_file = os.path.join(model_weights_path, f'E2EREC{jnd_value}.h5')
    REC.load_weights(model_weights_file)

    results = REC.predict(X_Test)
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
        REC = create_model(ImgReconstrution_Model_Path)
        
        train_model(REC, train_images, train_labels, valid_images, valid_labels, jnd_train_images, jnd_valid_images, checkpoint_path, csv_log_path, learning_rate, batch_size, epochs, jnd_value)
            
    elif base_command == 'test':
        test_images, test_labels, jnd_test_images = load_data(data_dir, base_command, jnd_value)
        REC = create_model(ImgReconstrution_Model_Path)
        
        test_model(REC, test_images, test_labels, jnd_test_images, result_path, model_weights_path, jnd_value)
            
    else:
        print("Invalid base command. Please use 'train' or 'test' followed by JND values.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the REC model.')
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
    