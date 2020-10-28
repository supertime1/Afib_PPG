import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import io
import itertools
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv1D, BatchNormalization, Input, Add, Activation, \
    MaxPooling1D, Dropout, Flatten, TimeDistributed, Bidirectional, Dense, LSTM, ZeroPadding1D, \
    AveragePooling1D, Conv1DTranspose, GlobalMaxPooling1D, Concatenate, Permute, Dot, Multiply, RepeatVector, \
    Lambda, Average

from tensorflow.keras.initializers import glorot_uniform

cnn = tf.keras.Sequential([
    # 1st Conv1D
    tf.keras.layers.Conv1D(8, 1, strides=1,
                           activation='relu', input_shape=(3750, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Dropout(0.2),
    # 2nd Conv1D
    tf.keras.layers.Conv1D(16, 3, strides=1,
                           activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Dropout(0.2),
    # 3rd Conv1D
    tf.keras.layers.Conv1D(32, 3, strides=1,
                           activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Dropout(0.2),
    # 4th Conv1D
    tf.keras.layers.Conv1D(64, 3, strides=1,
                           activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=2, strides=2),
    tf.keras.layers.Dropout(0.2),
    # 5th Conv1D
    tf.keras.layers.Conv1D(16, 1, strides=1,
                           activation='relu'),
    tf.keras.layers.BatchNormalization(),
    # Full connection layer
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(3, activation='softmax')
])


class Simple_CNN(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super(Simple_CNN, self).__init__()

        self.convA = TimeDistributed(Conv1D(8, 1, strides=1, activation='relu'), input_shape=input_shape)
        self.batchA = TimeDistributed(BatchNormalization())
        self.maxpoolA = TimeDistributed(MaxPooling1D(pool_size=2, strides=2))
        self.dropA = TimeDistributed(Dropout(0.2))

        self.convB = TimeDistributed(Conv1D(16, 3, strides=1, activation='relu'))
        self.batchB = TimeDistributed(BatchNormalization())
        self.maxpoolB = TimeDistributed(MaxPooling1D(pool_size=2, strides=2))
        self.dropB = TimeDistributed(Dropout(0.2))

        self.convC = TimeDistributed(Conv1D(32, 3, strides=1, activation='relu'))
        self.batchC = TimeDistributed(BatchNormalization())
        self.maxpoolC = TimeDistributed(MaxPooling1D(pool_size=2, strides=2))
        self.dropC = TimeDistributed(Dropout(0.2))

        self.convD = TimeDistributed(Conv1D(64, 3, strides=1, activation='relu'))
        self.batchD = TimeDistributed(BatchNormalization())
        self.maxpoolD = TimeDistributed(MaxPooling1D(pool_size=2, strides=2))
        self.dropD = TimeDistributed(Dropout(0.2))

        self.convE = TimeDistributed(Conv1D(16, 1, strides=1, activation='relu'))
        self.batch_normE = TimeDistributed(BatchNormalization())
        self.flatE = TimeDistributed(Flatten())

    def call(self, inputs):
        x = self.convA(inputs)
        x = self.batchA(x)
        x = self.maxpoolA(x)
        x = self.dropA(x)

        x = self.convB(x)
        x = self.batchB(x)
        x = self.maxpoolB(x)
        x = self.dropB(x)

        x = self.convC(x)
        x = self.batchC(x)
        x = self.maxpoolC(x)
        x = self.dropC(x)

        x = self.convD(x)
        x = self.batchD(x)
        x = self.maxpoolD(x)
        x = self.dropD(x)

        x = self.convE(x)
        x = self.batch_normE(x)
        x = self.flatE(x)

        return x


class CNN_LSTM(Model):
    def __init__(self, input_shape, classes):
        super(CNN_LSTM, self).__init__()
        self.cnn = Simple_CNN(input_shape=input_shape)
        self.bi_lstmA = Bidirectional(LSTM(32, return_sequences=True))
        self.bi_lstmB = Bidirectional(LSTM(16))
        self.dense = Dense(classes, activation='softmax')

    def call(self, inputs):
        x = self.cnn(inputs)
        x = self.bi_lstmA(x)
        x = self.bi_lstmB(x)
        x = self.dense(x)

        return x


def identity_block_18(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters

    X_shortcut = X

    # First component of main path
    X = Conv1D(filters=F1, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=2, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=2, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block_18(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv1D(filters=F1, kernel_size=f, strides=s, padding='valid', name=conv_name_base + '2a',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=2, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv1D(filters=F2, kernel_size=f, strides=1, padding='same', name=conv_name_base + '2b',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=2, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv1D(filters=F1, kernel_size=f, strides=s, padding='valid', name=conv_name_base + '1',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=2, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def ResNet18(input_shape=(750, 1), classes=1, as_model=False):
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding1D(3)(X_input)

    # Stage 1
    X = Conv1D(64, 7, strides=2, name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=2, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(3, strides=2)(X)

    # Stage 2
    X = identity_block_18(X, 3, [64, 64], stage=2, block='a')
    X = identity_block_18(X, 3, [64, 64], stage=2, block='b')

    # Stage 3
    X = convolutional_block_18(X, f=3, filters=[128, 128], stage=3, block='a', s=2)
    X = identity_block_18(X, 3, [128, 128], stage=3, block='b')

    # Stage 4
    X = convolutional_block_18(X, f=3, filters=[256, 256], stage=4, block='a', s=2)
    X = identity_block_18(X, 3, [256, 256], stage=4, block='b')

    # Stage 5
    X = convolutional_block_18(X, f=3, filters=[512, 512], stage=5, block='a', s=2)
    X = identity_block_18(X, 3, [512, 512], stage=5, block='b')

    # AVGPOOL
    X = AveragePooling1D(2, name="avg_pool")(X)

    # output layer
    X = Flatten()(X)

    if as_model:
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet18')

    return model


def Resnet18_LSTM(Tx, n_a, n_s, input_image_size, classes=1):
    # define resnet
    resnet = ResNet18(input_shape=(input_image_size, 1), classes=classes, as_model=False)

    X_input = Input(shape=(Tx, input_image_size, 1))

    X = tf.keras.layers.TimeDistributed(resnet)(X_input)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_a, return_sequences=True))(X)
    X = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(n_s))(X)
    X = tf.keras.layers.Dense(classes, activation='softmax')(X)

    model = Model(inputs=[X_input], outputs=X)

    return model


def plot_confusion_matrix(cm, class_names, normalize=False):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    plt.ylim(bottom=-0.5, top=2.5)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 1.5

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image


def decay(epoch):
    if epoch < 100:
        return 1e-3
    elif 100 <= epoch < 200:
        return 1e-4
    else:
        return 1e-5


class Autoencoder(Model):
    def __init__(self, input_shape):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            Input(shape=input_shape),
            Conv1D(128, 3, activation='relu', padding='same', strides=1),
            Conv1D(64, 3, activation='relu', padding='same', strides=1),
            Conv1D(32, 3, activation='relu', padding='same', strides=1),
        ])

        self.decoder = tf.keras.Sequential([
            Conv1DTranspose(32, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv1DTranspose(64, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv1DTranspose(128, kernel_size=3, strides=1, activation='relu', padding='same'),
            Conv1D(1, kernel_size=3, activation='sigmoid', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
