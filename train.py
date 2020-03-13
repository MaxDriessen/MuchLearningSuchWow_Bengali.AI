import os

from tqdm.auto import tqdm
import cv2
import pandas as pd
import keras
from models import resnext, simplenet
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import psutil

import gc
import wandb
from wandb.keras import WandbCallback

# Preprocessing stuff
def resize(df, size=64, need_progress_bar=True):
    resized = {}
    resize_size = 64
    if need_progress_bar:
        for i in tqdm(range(df.shape[0])):
            image = df.loc[df.index[i]].values.reshape(137, 236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x, y, w, h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax, xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    else:
        for i in range(df.shape[0]):
            # image = cv2.resize(df.loc[df.index[i]].values.reshape(137,236),(size,size),None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
            image = df.loc[df.index[i]].values.reshape(137, 236)
            _, thresh = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]

            idx = 0
            ls_xmin = []
            ls_ymin = []
            ls_xmax = []
            ls_ymax = []
            for cnt in contours:
                idx += 1
                x, y, w, h = cv2.boundingRect(cnt)
                ls_xmin.append(x)
                ls_ymin.append(y)
                ls_xmax.append(x + w)
                ls_ymax.append(y + h)
            xmin = min(ls_xmin)
            ymin = min(ls_ymin)
            xmax = max(ls_xmax)
            ymax = max(ls_ymax)

            roi = image[ymin:ymax, xmin:xmax]
            resized_roi = cv2.resize(roi, (resize_size, resize_size), interpolation=cv2.INTER_AREA)
            resized[df.index[i]] = resized_roi.reshape(-1)
    resized = pd.DataFrame(resized).T
    return resized

# Load data
# print('Loading data...')
train_df_ = pd.read_csv('bengaliai-cv19/train.csv')
train_df_ = train_df_.drop(['grapheme'], axis=1)
#
# train_df, classes_df = None, None
# for i in tqdm(range(1)):
#     if train_df is not None:
#         train_df_new = pd.merge(pd.read_feather(f'train_image_data_{i}.feather'), train_df_, on='image_id', copy=False).drop(['image_id'], axis=1)
#         # train_df_new = train_df_new.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)
#         train_df_new, classes_df_new = train_df_new.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1), train_df_new[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
#         train_df_new = resize(train_df_new) / 255
#         gc.collect()
#         train_df = pd.concat([train_df, train_df_new])
#         classes_df = pd.concat([classes_df, classes_df_new])
#
#     else:
#         train_df = pd.merge(pd.read_feather(f'train_image_data_{i}.feather'), train_df_, on='image_id', copy=False).drop(['image_id'], axis=1)
#         train_df, classes_df = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1), train_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
#         train_df = resize(train_df) / 255
#     gc.collect()
#     print(psutil.virtual_memory().percent)
#
#
# # X_train, classes_df = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1), train_df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]
# # del train_df
# gc.collect()
# # X_train = resize(X_train) / 255
#
# X_train = train_df.values.reshape(-1, 64, 64, 1)
# del train_df
# gc.collect()
#
# print('gen_y')
# Y_train_root = pd.get_dummies(classes_df['grapheme_root']).values
# Y_train_vowel = pd.get_dummies(classes_df['vowel_diacritic']).values
# Y_train_consonant = pd.get_dummies(classes_df['consonant_diacritic']).values
#
# print('split')
# x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(
#             X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)
#
# del X_train
# del Y_train_root, Y_train_vowel, Y_train_consonant


# Wandb
run = wandb.init(project='bengali')
config = run.config

# Config
config.blocks = [
    {
        'width': 48,
        'output_width': 64,
        'cardinality': 4,
        'count': 4
    },
    {
        'width': 64,
        'output_width': 128,
        'cardinality': 4,
        'count': 4
    },
    {
        'width': 128,
        'output_width': 256,
        'cardinality': 4,
        'count': 4
    },
    {
        'width': 256,
        'output_width': 512,
        'cardinality': 4,
        'count': 4
    }
]
config.iChannels = 48
config.epochs = 120
config.max_lr = 0.0016
config.min_lr = 0.0004
config.n_cycles = 8
config.dropout = 0.3
config.batch_size = 200
config.validation_split = 0.08
config.net_type = 'simple'
config.steps_per_epoch = int(200840*(1-config.validation_split))//config.batch_size//4
validation_steps = int(200840*config.validation_split)//config.batch_size//4
# config.meta_epochs = 5


# Initialize model and model callbacks/generators
class MultiOutputDataGenerator(keras.preprocessing.image.ImageDataGenerator):

    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):

        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)

        for flowx, flowy in super().flow(x, targets, batch_size=batch_size,
                                         shuffle=shuffle):
            target_dict = {}
            i = 0
            for output in ordered_outputs:
                target_length = target_lengths[output]
                target_dict[output] = flowy[:, i: i + target_length]
                i += target_length

            yield flowx, target_dict

# datagen = MultiOutputDataGenerator(
#             featurewise_center=False,  # set input mean to 0 over the dataset
#             samplewise_center=False,  # set each sample mean to 0
#             featurewise_std_normalization=False,  # divide inputs by std of the dataset
#             samplewise_std_normalization=False,  # divide each input by its std
#             zca_whitening=False,  # apply ZCA whitening
#             rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
#             zoom_range=0.15,  # Randomly zoom image
#             width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
#             height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
#             horizontal_flip=False,  # randomly flip images
#             vertical_flip=False)  # randomly flip images
#
# datagen.fit(x_train)


if config.net_type is None:
    model = resnext(config.blocks, config.iChannels)
elif config.net_type == 'simple':
    model = simplenet(config.blocks, config.iChannels, config.dropout)
else:
    raise ValueError(f'Invalid net type: {config.net_type}')

model.compile(optimizer='adam', loss='categorical_crossentropy', loss_weights=[2, 1, 1], metrics=['accuracy', keras.metrics.Recall()])
# print(model.summary())

learning_rate_reduction_root = ReduceLROnPlateau(monitor='dense_a_accuracy',
                                                 patience=3,
                                                 verbose=1,
                                                 factor=0.5,
                                                 min_lr=0.00001)
learning_rate_reduction_vowel = ReduceLROnPlateau(monitor='dense_b_accuracy',
                                                  patience=3,
                                                  verbose=1,
                                                  factor=0.5,
                                                  min_lr=0.00001)
learning_rate_reduction_consonant = ReduceLROnPlateau(monitor='dense_c_accuracy',
                                                      patience=3,
                                                      verbose=1,
                                                      factor=0.5,
                                                      min_lr=0.00001)
train_df_ = pd.read_csv('bengaliai-cv19/train.csv')
train_df_ = train_df_.drop(['grapheme'], axis=1, inplace=False)


# Train
print('Starting training...')
# histories = []
# for _ in range(config.meta_epochs):
#     for i in range(4):
#         train_df = pd.merge(pd.read_feather('train_image_data_%s.feather' % str(i)), train_df_,
#                             on='image_id').drop(['image_id'], axis=1)
#
#
#         X_train = train_df.drop(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic'], axis=1)
#         X_train = X_train.astype(np.uint8)
#         X_train = resize(X_train) / 255
#
#         # CNN takes images in shape `(batch_size, h, w, channels)`, so reshape the images
#         X_train = X_train.values.reshape(-1, 64, 64, 1)
#
#         Y_train_root = pd.get_dummies(train_df['grapheme_root']).values
#         Y_train_vowel = pd.get_dummies(train_df['vowel_diacritic']).values
#         Y_train_consonant = pd.get_dummies(train_df['consonant_diacritic']).values
#         # Y_train_root = Y_train_root.astype(np.float32)
#         # Y_train_vowel = Y_train_vowel.astype(np.float32)
#         # Y_train_consonant = Y_train_consonant.astype(np.float32)
#
#         print(f'Training images: {X_train.shape}')
#         print(f'Training labels root: {Y_train_root.shape}')
#         print(f'Training labels vowel: {Y_train_vowel.shape}')
#         print(f'Training labels consonants: {Y_train_consonant.shape}')
#
#         # Divide the data into training and validation set
#         x_train, x_test, y_train_root, y_test_root, y_train_vowel, y_test_vowel, y_train_consonant, y_test_consonant = train_test_split(
#             X_train, Y_train_root, Y_train_vowel, Y_train_consonant, test_size=0.08, random_state=666)
#         del train_df
#         del X_train
#         del Y_train_root, Y_train_vowel, Y_train_consonant
#
#         # Data augmentation for creating more training data
#         datagen = MultiOutputDataGenerator(
#             featurewise_center=False,  # set input mean to 0 over the dataset
#             samplewise_center=False,  # set each sample mean to 0
#             featurewise_std_normalization=False,  # divide inputs by std of the dataset
#             samplewise_std_normalization=False,  # divide each input by its std
#             zca_whitening=False,  # apply ZCA whitening
#             rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
#             zoom_range=0.15,  # Randomly zoom image
#             width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
#             height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
#             horizontal_flip=False,  # randomly flip images
#             vertical_flip=False,   # randomly flip images
#             rescale=1.0/255.0,
#             validation_split=0.055)
#
#         # This will just calculate parameters required to augment the given data. This won't perform any augmentations
#         print('fitting datagen')
#         datagen.fit(x_train)
#
#         # Fit the model
#         print('starting fitting')
#         history = model.fit_generator(
#             datagen.flow(x_train, {'dense_a': y_train_root, 'dense_b': y_train_vowel, 'dense_c': y_train_consonant},
#                          batch_size=config.batch_size),
#             epochs=config.epochs, validation_data=(x_test, [y_test_root, y_test_vowel, y_test_consonant]),
#             steps_per_epoch=x_train.shape[0] // config.batch_size,
#             callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant,
#                         WandbCallback(data_type='image')])
#
#         histories.append(history)
#
#         # Delete to reduce memory usage
#         del x_train
#         del x_test
#         del y_train_root
#         del y_test_root
#         del y_train_vowel
#         del y_test_vowel
#         del y_train_consonant
#         del y_test_consonant
#         gc.collect()

datagen = MultiOutputDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=8,  # randomly rotate images in the range (degrees, 0 to 180)
            zoom_range=0.15,  # Randomly zoom image
            width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False,   # randomly flip images
            rescale=1.0/255.0,
            validation_split=config.validation_split)

from keras.callbacks import Callback
from keras import backend

class SnapshotEnsemble(Callback):
    # constructor
    def __init__(self, n_epochs, n_cycles, lrate_min, lrate_max, verbose=0):
        super().__init__()
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_min = lrate_max
        self.lr_max = lrate_max
        self.lrates = list()

    # calculate learning rate for an epoch
    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_min, lrate_max):
        epochs_per_cycle = np.floor(n_epochs / n_cycles)
        cos_inner = (np.pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return (lrate_max-lrate_min) / 2 * (np.cos(cos_inner) + 1) + lrate_min

    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs=None):
        # calculate learning rate
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_min, self.lr_max)
        # set learning rate
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)

    def on_epoch_end(self, epoch, logs={}):
        # check if we can save model
        epochs_per_cycle = np.floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            # save model to file
            filename = os.path.join(wandb.run.dir, "snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle))
            self.model.save(filename)
            print('>saved snapshot %s, epoch %d' % (filename, epoch))


train_df_['image_id'] = train_df_['image_id'].astype(str)+'.png'
# train_df_['grapheme_root'] = 'root_' + train_df_['grapheme_root'].astype(np.str)
# train_df_['vowel_diacritic'] = 'vowel_' + train_df_['vowel_diacritic'].astype(np.str)
# train_df_['consonant_diacritic'] = 'consonant_' + train_df_['consonant_diacritic'].astype(np.str)
# train_df_['class'] = train_df_[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']].values.tolist()

# groupby_dict = {'grapheme_root': 'class',
#                 'vowel_diacritic': 'class',
#                 'consonant_diacritic': 'class'}

# Set the index of df as Column 'id'
# train_df_ = train_df_.set_index('image_id')

# Groupby the groupby_dict created above
# train_df_ = train_df_.groupby(groupby_dict, axis=1).min()

# for i, item in enumerate(train_df_['class']):
#     if not isinstance(item, list):
#         print(i, item)
#         print(train_df_.groupby(['image_id'])[['grapheme_root', 'vowel_diacritic']].apply(list))
#         error()
# print(train_df_['class'].head)
# print(train_df_['class'].dtype)
    # [train_df_['grapheme_root'], train_df_['vowel_diacritic'], train_df_['consonant_diacritic']]


def one_hot(arr: np.ndarray, num_classes):
    arr_one_hot = np.zeros((arr.size, num_classes))
    arr_one_hot[np.arange(arr.size), arr] = 1
    return arr_one_hot

Y_train_root = pd.get_dummies(train_df_.set_index('image_id')['grapheme_root'], dtype=np.float32)
Y_train_vowel = pd.get_dummies(train_df_.set_index('image_id')['vowel_diacritic'], dtype=np.float32)
Y_train_consonant = pd.get_dummies(train_df_.set_index('image_id')['consonant_diacritic'], dtype=np.float32)

def generator_wrapper(gen: MultiOutputDataGenerator, df: pd.DataFrame, subset: str, batch_size: int):
    import tensorflow as tf
    for flowx, flowy in gen.flow_from_dataframe(df, color_mode='grayscale', directory='data', x_col='image_id',
                                                  y_col='image_id', class_mode='raw', target_size=(64, 64), subset=subset, batch_size=batch_size, shuffle=True):
        # batch_records = df[df['image_id'].isin(flowy)]
        # grapheme_roots = batch_records['grapheme_root'].values.astype(np.int)
        # grapheme_roots = one_hot(grapheme_roots, 168)
        # vowel_diacritics = batch_records['vowel_diacritic'].values.astype(np.int)
        # vowel_diacritics = one_hot(vowel_diacritics, 11)
        # consonant_diacritics = batch_records['consonant_diacritic'].values.astype(np.int)
        # consonant_diacritics = one_hot(consonant_diacritics, 7)

        # yield flowx, {'dense_a': grapheme_roots, 'dense_b': vowel_diacritics, 'dense_c': consonant_diacritics}
        #
        # import matplotlib.pyplot as plt
        # plt.imshow(flowx[0].reshape(64, 64))
        # plt.show()
        # error()

        yield flowx, {
            'dense_a': Y_train_root.loc[flowy].values,
            'dense_b': Y_train_vowel.loc[flowy].values,
            'dense_c': Y_train_consonant.loc[flowy].values,
        }


# model.fit_generator(datagen.flow_from_dataframe(train_df_, directory='data', class_mode='categorical', x_col='image_id', y_col='class',
#                      batch_size=config.batch_size, color_mode='grayscale', target_size=(64, 64)),
validation_generator = generator_wrapper(datagen, train_df_, 'validation', config.batch_size)
model.fit_generator(generator_wrapper(datagen, train_df_, 'training', config.batch_size), validation_data=validation_generator, validation_steps=validation_steps,
                    epochs=config.epochs, steps_per_epoch=config.steps_per_epoch,
        callbacks=[learning_rate_reduction_root, learning_rate_reduction_vowel, learning_rate_reduction_consonant,
                    WandbCallback(data_type='image')])#, SnapshotEnsemble(config.epochs, config.n_cycles, config.min_lr, config.max_lr)])

# Save things
print('Saving things...')
model.save(os.path.join(wandb.run.dir, "model.hdf5"))
wandb.save('train.py')
wandb.save('models.py')
