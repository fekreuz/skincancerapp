import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adamax
from keras.callbacks import EarlyStopping
from keras.models import load_model
import tensorflow.keras.callbacks as callbacks
import keras_tuner
from keras_tuner.tuners import RandomSearch
import pickle


#class to save training history
class SaveHistoryCallback(callbacks.Callback):
    def __init__(self, filepath):
        super(SaveHistoryCallback, self).__init__()
        self.filepath = filepath
        self.history_data = {}

    def on_epoch_end(self, epoch, logs=None):
        for key, value in logs.items():
            self.history_data.setdefault(key, []).append(value)

    def on_train_end(self, logs=None):
        df_history = pd.DataFrame(self.history_data)
        df_history.to_csv(self.filepath, index=False)
        print(f"Training history saved to {self.filepath}")

#data processing
def random_under_sampling(df, categories, sample_size):
    """
    Function to random undersample datasets by given categories and sample size
    --------------
    Parameter: df: dataframe, categories: categories of samples, sample_size: size of sample per category.
    --------------
    Return: under_sampled_data
    """
    
    # Separate the data based on categories
    category_data = {category: df[df[category] == 1] for category in categories}

    # Randomly under-sample each category to sample_size samples
    under_sampled_data = pd.DataFrame()

    for category in categories:
        if len(category_data[category]) > sample_size:
            under_sampled_data = pd.concat([under_sampled_data, category_data[category].sample(sample_size, random_state=42)])
        else:
            under_sampled_data = pd.concat([under_sampled_data, category_data[category]])

    # Shuffle the final under-sampled data
    under_sampled_data = under_sampled_data.sample(frac=1).reset_index(drop=True)

    return under_sampled_data

#shape the images
def process_images(base_path_image, df, image_size=(224, 224)):
    """
    Function that loads images, reshape them to image_size (224, 224) and transform them into numpy arrays.
    -----------
    Variables: base_path_image: directory path of image file e.g.'Desktop/to/your/image/directory',
               df: dataframe that contains image file names (incl. datatype suffix), image_size: reshaping size.
    -----------
    Returns:   <images>, list of image arrays 
    """
    images = []
    files = os.listdir(base_path_image)
    
    for file in df['image']:
        if not file.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            continue
        img = Image.open(os.path.join(base_path_image, file))
        name = file
        new_width, new_height = image_size
        resized_image = img.resize((new_width, new_height))
        img_array = np.array(resized_image)
        images.append({'image': name, 'array': img_array})
    
    return images

#check shape of images
def shape_check(df, shape):
    """
    Checks if the shape of all arrays within a dataframe is equal to defined shape.
    ------------------
    Parameter: 
    df: dataframe, shape: tuple of dimensions e.g. (450, 600, 3)
    ------------------
    Returns: None,
    Prints the shape of each array or "Shape is not <shape>. Shape is <shape_>".

    """
    for i in range(len(df)):
        shape_= df[i].shape
        if shape_ != shape:
            print(f'Shape is not {shape}. Shape is {shape_}')
        else:
            print(shape_)

    return None

#CNN model
def create_model(best_model='checkpoints/best_model_Adam_lr_2_backup.h5',
                 checkpoint_filepath='checkpoints/best_model_Adam_lr_2_backup_friday1.h5', 
                 history_filepath='checkpoints/best_model_Adam_lr_2_backup_friday1.csv',
                 learning_rate=0.00026712, epochs=50, batch_size=40, validation_split=0.4, freeze=True): #r:0.001
    """
    Function to create a DenseNet201-based model based on previous <best_model>.
    -----------
    Parameter: best_model= path of model, checkpoint_filepath= path of checkpoint model, history_filepath= path of full trainig history, 
                learning_rate= Adamax learning rate, batch_size= batch size, validation_split= validation split.
    -----------
    Returns:
    new_model
    """
    K.clear_session()
    new_model = load_model(best_model)
    # best_model = pickle.load(open("DenseNet_s150_batch_size20_e10.sav", "rb"))
    
     # Freezing the first 150 layers
    if freeze:
        for layer in new_model.layers[:150]:
            layer.trainable = False

    model_check = ModelCheckpoint(checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    save_history = SaveHistoryCallback(history_filepath)

    new_model.compile(Adamax(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    history = new_model.fit(X_train,
                             y_train,
                             epochs=epochs,
                             batch_size=batch_size, 
                             validation_split=validation_split, class_weight=class_weights_dict, 
                             callbacks=[model_check, save_history])

    return new_model

#keras hyperparameter tuner 
def create_model_hp(hp):
    """
    Function to create a DenseNet201-based model with hyperparameter tuning.

    Parameters:
    -----------
    best_model: str
    The path of the pre-trained model that serves as a starting point for further fine-tuning.

    checkpoint_filepath: str
    The path where the best model will be saved during training based on validation loss.

    history_filepath: str
    The path where the full training history will be saved in a CSV file.

    learning_rate: float
    The learning rate for the Adamax optimizer used during model compilation.

    batch_size: int
    The batch size for training the model.

    validation_split: float
    The fraction of the training data to be used as validation data during training.

    Returns:
    --------
    new_model: TensorFlow Keras model   
    The trained DenseNet201-based model with optimized hyperparameters.
    """

    learning_rate = hp.Float('learning_rate', min_value=0.0001, max_value=0.1, sampling='log')
    batch_size = hp.Choice('batch_size', values=[20, 40, 60])

    tf.keras.backend.clear_session()
    new_model = tf.keras.models.load_model('checkpoints/best_model_Adam_lr.h5')
    
    model_check = tf.keras.callbacks.ModelCheckpoint('checkpoints/best_model_freeze_layer150_lr01.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
    save_history = SaveHistoryCallback('checkpoints/best_model_freeze_layer150_lr01.csv')

    new_model.compile(optimizer=Adamax(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    history = new_model.fit(X_train,
                            y_train,
                            epochs=50,
                            batch_size=batch_size, 
                            validation_split=0.4, class_weight=class_weights_dict, 
                            callbacks=[model_check, save_history])

    return new_model

#load reference csv
df=pd.read_csv('archive/GroundTruth.csv', sep=',')
df['image']=df['image'].apply(lambda x: x+ '.jpg')

#data processing
#undersample the dataframe
categories= ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
sample_size = 150
under_sampled_data = random_under_sampling(df, categories, sample_size)

# load pictures and preprocessing
base_path_image = '/Users/nando_macbook/Desktop/final_project/archive/images/'
#classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
images= process_images(base_path_image, under_sampled_data)
df_pic = pd.DataFrame(images)

#merge to final df
merged_df = df_pic.merge(under_sampled_data[['image', 'MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']], on='image', how='left')
merged_df

#checking for Nas
merged_df.isna().sum()

#shape check
shape_check(merged_df['array'], (224, 224, 3))

#defining arrays as X and binary encoded lesions as y
X=np.stack(merged_df['array'].values) #stacking possible since arrays!
y=merged_df[['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']].values

#train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)

#reverse OnehotEncoded y_train with argmax to use <compute_class_weigth>
y_train_index= np.argmax(y_train, axis=1, out=None)
class_labels=np.array([0, 1, 2, 3, 4, 5, 6]) #['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
number_class_labels=len(class_labels)
class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train_index)
class_weights_dict = dict(zip(class_labels, class_weights))

#retrain model
create_model(best_model='checkpoints/best_model_Adam_lr_2_backup_friday1.h5',
                 checkpoint_filepath='checkpoints/best_model_freeze150_friday.h5', 
                 history_filepath='checkpoints/best_model_freeze150_friday.csv', epochs=200)




#tuner = RandomSearch(
 #   create_model_hp,
#   objective='val_accuracy',
 #   max_trials=5,  # Adjust this value based on the search space and computational resources
#    executions_per_trial=1,
 #   directory='my_tuner_dir',  # Directory to store tuning results
#   project_name='my_model_tuning'  # Name of the tuning project
#)
# Perform hyperparameter tuning
#tuner.search(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Get the best model
#best_model = tuner.get_best_models(num_models=1)[0]
#pickle.dump(best_model, open("my_tuner_dir/DenseNet_s150_batch_size40_tuner1.sav", "wb"))