import numpy as np
from keras.layers import Conv2D, BatchNormalization, Activation, Add, MaxPool2D, \
    Dense, Dropout, GlobalAveragePooling2D, Concatenate, Input, Flatten
from keras import Model
from keras.regularizers import l2


WEIGHT_DECAY = 5e-4



def resnext_block(x, width, output_width, cardinality):
    x = Conv2D(filters=width, padding='same', kernel_size=3)(x)
    inp = x
    inp = Conv2D(output_width, padding='same', kernel_size=1)(inp)
    subblocks = []

    for i in range(cardinality):
        y = Conv2D(filters=width, kernel_size=1)(x)
        # y = BatchNormalization()(y)#name=f'bn_1_{np.random.random()}')(y)
        y = Activation('relu')(y)
        y = Conv2D(filters=width, kernel_size=3, padding='same')(y)
        # y = BatchNormalization()(y)#name=f'bn_3_{np.random.random()}')(y)
        y = Activation('relu')(y)
        subblocks.append(y)

    x = Concatenate()(subblocks)
    x = Conv2D(output_width, kernel_size=1)(x)
    x = BatchNormalization()(x)
    x = Add()([x, inp])

    x = Activation('relu')(x)

    return x


def init_block(x, iChannels):
    x = Conv2D(iChannels, (7, 7), strides=2, padding='same', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    return x


def resnext(blocks, iChannels, input_size=(64, 64, 1)):
    x = Input(shape=input_size)
    inp = x

    x = init_block(x, iChannels)

    for b in blocks:
        for i in range(b['count']):
            x = resnext_block(x, b['width'], b['output_width'], b['cardinality'])
        x = MaxPool2D()(x)

    x = GlobalAveragePooling2D()(x)

    x = Dense(2048, activation="relu")(x)
    x = Dropout(rate=0.12)(x)
    x = Dense(1024, activation="relu")(x)

    head_root = Dense(168, activation='softmax', name='dense_a')(x)
    head_vowel = Dense(11, activation='softmax', name='dense_b')(x)
    head_consonant = Dense(7, activation='softmax', name='dense_c')(x)

    model = Model(inputs=inp, outputs=[head_root, head_vowel, head_consonant])

    return model

def simplenet(blocks, iChannels, dropout, input_size=(64, 64, 1)):
    inputs = Input(shape=input_size)
    model = Conv2D(filters=iChannels, kernel_size=(7, 7), padding='SAME', activation='relu', input_shape=input_size)(inputs)
    for b in blocks:
        for i in range(b['count']):
            model = Conv2D(filters=b['width'], kernel_size=(3, 3), padding='SAME', activation='relu')(model)
        model = BatchNormalization(momentum=0.15)(model)
        model = MaxPool2D(pool_size=(2, 2))(model)
        model = Conv2D(filters=b['output_width'], kernel_size=(5, 5), padding='SAME', activation='relu')(model)
        model = Dropout(rate=dropout)(model)

    model = Flatten()(model)
    model = Dense(1024, activation="relu")(model)
    model = Dropout(rate=dropout)(model)
    dense = Dense(512, activation="relu")(model)

    head_root = Dense(168, activation='softmax', name='dense_a')(dense)
    head_vowel = Dense(11, activation='softmax', name='dense_b')(dense)
    head_consonant = Dense(7, activation='softmax', name='dense_c')(dense)

    model = Model(inputs=inputs, outputs=[head_root, head_vowel, head_consonant])
    return model

# %%

# model = resnext([
#     {
#         'width': 128,
#         'output_width': 256,
#         'cardinality': 32,
#         'count': 3
#     },
#     {
#         'width': 256,
#         'output_width': 512,
#         'cardinality': 32,
#         'count': 4
#     },
#     {
#         'width': 512,
#         'output_width': 1024,
#         'cardinality': 32,
#         'count': 6
#     },
#     {
#         'width': 1024,
#         'output_width': 2048,
#         'cardinality': 32,
#         'count': 3
#     }
# ], 64)
