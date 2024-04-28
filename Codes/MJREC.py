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
def create_model(JND_Recon_Models_Path):
    # Input layer
    input_img = Input(shape=(352, 640, 3))
    
    # Create encoder-decoder blocks for each branch
    ModelREC1 = create_encoder_decoder(input_img, num_filters=128, latent_dim=192)
    ModelREC2 = create_encoder_decoder(input_img, num_filters=128, latent_dim=192)
    ModelREC3 = create_encoder_decoder(input_img, num_filters=128, latent_dim=192)
    
    # Load weights and freeze layers
    ModelREC1.load_weights(os.path.join(JND_Recon_Models_Path, 'JND1Reconstruction.h5'))
    ModelREC2.load_weights(os.path.join(JND_Recon_Models_Path, 'JND2Reconstruction.h5'))
    ModelREC3.load_weights(os.path.join(JND_Recon_Models_Path, 'JND3Reconstruction.h5'))
    
    for branch in [ModelREC1, ModelREC2, ModelREC3]:
        for layer in branch.layers[0:]:
            layer.trainable = False
    
    # Extract LAT features
    RECJND1 = ModelREC1.layers[14].output
    RECJND2 = ModelREC2.layers[14].output
    RECJND3 = ModelREC3.layers[14].output
    
    RECJND1 = BatchNormalization()(RECJND1)
    RECJND2 = BatchNormalization()(RECJND2)
    RECJND3 = BatchNormalization()(RECJND3)
    
    # Apply RECBlock
    RECJND1 = RECBlock(RECJND1, num_filters=512)
    RECJND2 = RECBlock(RECJND2, num_filters=512)
    RECJND3 = RECBlock(RECJND3, num_filters=512)
    
    # Combine outputs
    combined = multiply([RECJND1, RECJND2, RECJND3])
    
    # Further processing and architecture
    CRECBlock = RECBlock(combined, num_filters=256)
    CRECBlock = RECBlock(CRECBlock, num_filters=128)
    CRECBlock = RECBlock(CRECBlock, num_filters=64)
    CRECBlock = RECBlock(CRECBlock, num_filters=32)
    
    Flat = Flatten()(CRECBlock)
    
    FC = Dense(512, activation='relu')(Flat)
    
    jnd_output_1 = Dense(1, activation='relu', name='JND1_output')(FC)
    
    jnd_output_2 = Dense(1, activation='relu', name='JND2_output')(FC)
    
    jnd_output_3 = Dense(1, activation='relu', name='JND3_output')(FC)
    
    # Create the final model that outputs the three JND values
    MJREC = Model(inputs=input_img, outputs=[jnd_output_1, jnd_output_2, jnd_output_3])
    return MJREC

##################################################################################################################

def train_model(MJREC, X_Train, train_labels, X_Valid, valid_labels, checkpoint_path, csv_log_path, learning_rate, batch_size, epochs):
    # Compile and train the model
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    MJREC.compile(optimizer=optimizer,
                    loss={'JND1_output': 'mean_absolute_error', 'JND2_output': 'mean_absolute_error', 'JND3_output': 'mean_absolute_error'},
                    loss_weights={'JND1_output': 1., 'JND2_output': 1., 'JND3_output': 1.})

    checkpoint = ModelCheckpoint(filepath=os.path.join(checkpoint_path, 'BestModel.h5'),
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min')
                             
    csv_logger = CSVLogger(os.path.join(csv_log_path, 'logs.csv'), append=True, separator=';')


    history = MJREC.fit(X_Train,
                           {'JND1_output': train_labels[:,0], 'JND2_output': train_labels[:,1], 'JND3_output': train_labels[:,2]},
                           epochs=epochs,
                           batch_size=batch_size,
                           validation_data=(X_Valid, {'JND1_output': valid_labels[:,0], 'JND2_output': valid_labels[:,1], 'JND3_output': valid_labels[:,2]}),
                           callbacks=[checkpoint, csv_logger],
                           shuffle=True)


##################################################################################################################

def test_model(MJREC, X_Test, test_labels, result_path, model_weights_path):
    optimizer = keras.optimizers.Adam(lr=0.00001)
    MJREC.compile(optimizer=optimizer,
                    loss={'JND1_output': 'mean_absolute_error', 'JND2_output': 'mean_absolute_error', 'JND3_output': 'mean_absolute_error'},
                    loss_weights={'JND1_output': 1., 'JND2_output': 1., 'JND3_output': 1.})
    # Test the model and save results
    model_weights_file = os.path.join(model_weights_path, 'MJREC.h5')
    MJREC.load_weights(model_weights_file)

    results = MJREC.predict(X_Test)
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
    MJREC = create_model(JND_Recon_Models_Path)

    if base_command == 'train':
        train_model(MJREC, X_Train, train_labels, X_Valid, valid_labels, checkpoint_path, csv_log_path, learning_rate, batch_size, epochs)
    elif base_command == 'test':
        test_model(MJREC, X_Test, test_labels, result_path, model_weights_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the MJREC model.')
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
