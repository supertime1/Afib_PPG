from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import io
import itertools
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv1D, BatchNormalization, Input, Add, Activation, \
    MaxPooling1D, Dropout, Flatten, TimeDistributed, Bidirectional, Dense, LSTM, ZeroPadding1D, \
    AveragePooling1D, GlobalAveragePooling1D, Concatenate, Permute, Dot, Multiply, RepeatVector, \
    Lambda, Average, GlobalAveragePooling2D, DepthwiseConv2D
from tensorflow.keras.initializers import glorot_uniform

#import dependencies to define DepthwiseConv1D class


from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.convolutional import Conv1D
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.layers.DepthwiseConv1D')
class DepthwiseConv1D(Conv1D):
  def __init__(self,
               kernel_size,
               strides=1,
               padding='valid',
               depth_multiplier=1,
               data_format=None,
               activation=None,
               use_bias=True,
               depthwise_initializer='glorot_uniform',
               bias_initializer='zeros',
               depthwise_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               depthwise_constraint=None,
               bias_constraint=None,
               **kwargs):
    super(DepthwiseConv1D, self).__init__(
        filters=None,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        activation=activation,
        use_bias=use_bias,
        bias_regularizer=bias_regularizer,
        activity_regularizer=activity_regularizer,
        bias_constraint=bias_constraint,
        # autocast=False,
        **kwargs)
    self.depth_multiplier = depth_multiplier
    self.depthwise_initializer = initializers.get(depthwise_initializer)
    self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
    self.depthwise_constraint = constraints.get(depthwise_constraint)
    self.bias_initializer = initializers.get(bias_initializer)

  def build(self, input_shape):
    if len(input_shape) < 3:
      raise ValueError('Inputs to `DepthwiseConv1D` should have rank 3. '
                       'Received input shape:', str(input_shape))
    input_shape = tensor_shape.TensorShape(input_shape)

    #TODO(pj1989): replace with channel_axis = self._get_channel_axis()
    if self.data_format == 'channels_last':
        channel_axis = -1
    elif self.data_format == 'channels_first':
        channel_axis = 1

    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs to '
                       '`DepthwiseConv1D` '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    depthwise_kernel_shape = (self.kernel_size[0],
                              input_dim,
                              self.depth_multiplier)

    self.depthwise_kernel = self.add_weight(
        shape=depthwise_kernel_shape,
        initializer=self.depthwise_initializer,
        name='depthwise_kernel',
        regularizer=self.depthwise_regularizer,
        constraint=self.depthwise_constraint)

    if self.use_bias:
      self.bias = self.add_weight(shape=(input_dim * self.depth_multiplier,),
                                  initializer=self.bias_initializer,
                                  name='bias',
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)
    else:
      self.bias = None
    # Set input spec.
    self.input_spec = InputSpec(ndim=3, axes={channel_axis: input_dim})
    self.built = True

  def call(self, inputs):
    if self.padding == 'causal':
      inputs = array_ops.pad(inputs, self._compute_causal_padding())
    if self.data_format == 'channels_last':
      spatial_start_dim = 1
    else:
      spatial_start_dim = 2

    # Explicitly broadcast inputs and kernels to 4D.
    # TODO(fchollet): refactor when a native depthwise_conv2d op is available.
    strides = self.strides * 2
    inputs = array_ops.expand_dims(inputs, spatial_start_dim)
    depthwise_kernel = array_ops.expand_dims(self.depthwise_kernel, 0)
    dilation_rate = (1,) + self.dilation_rate

    outputs = backend.depthwise_conv2d(
        inputs,
        depthwise_kernel,
        strides=strides,
        padding=self.padding,
        dilation_rate=dilation_rate,
        data_format=self.data_format)

    if self.use_bias:
      outputs = backend.bias_add(
          outputs,
          self.bias,
          data_format=self.data_format)

    outputs = array_ops.squeeze(outputs, [spatial_start_dim])

    if self.activation is not None:
      return self.activation(outputs)

    return outputs

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if self.data_format == 'channels_first':
      length = input_shape[2]
      out_filters = input_shape[1] * self.depth_multiplier
    elif self.data_format == 'channels_last':
      length = input_shape[1]
      out_filters = input_shape[2] * self.depth_multiplier

    length = conv_utils.conv_output_length(length, self.kernel_size,
                                           self.padding,
                                           self.strides)
    if self.data_format == 'channels_first':
      return (input_shape[0], out_filters, length)
    elif self.data_format == 'channels_last':
      return (input_shape[0], length, out_filters)

  def get_config(self):
    config = super(DepthwiseConv1D, self).get_config()
    config.pop('filters')
    config.pop('kernel_initializer')
    config.pop('kernel_regularizer')
    config.pop('kernel_constraint')
    config['depth_multiplier'] = self.depth_multiplier
    config['depthwise_initializer'] = initializers.serialize(
        self.depthwise_initializer)
    config['depthwise_regularizer'] = regularizers.serialize(
        self.depthwise_regularizer)
    config['depthwise_constraint'] = constraints.serialize(
        self.depthwise_constraint)
    return config

def MobileNet(input_shape=None, dropout=0.2, alpha=1, classes=3):

    signal_input = Input(shape=input_shape)

    x = Conv2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(signal_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(64 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(128 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(256 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(512 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv2D((3, 3), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(int(1024 * alpha), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout)(x)
    out = Dense(classes, activation='softmax')(x)

    model = Model(signal_input, out, name='mobilenet')
    return model

def MobileNet_1D(input_shape=None, dropout=0.2, alpha=1, classes=3):

    signal_input = Input(shape=input_shape)

    x = Conv1D(int(32 * alpha), 3, strides=2, padding='same', use_bias=False)(signal_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(64 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(128 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(128 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(256 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(256 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(512 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=2, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(1024 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = DepthwiseConv1D(3, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(int(1024 * alpha), 1, strides=1, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout)(x)
    out = Dense(classes, activation='softmax')(x)

    model = Model(signal_input, out, name='mobilenet')
    return model

def cnn_lstm(input_shape=(3,1250,1), classes=3):

    cnn = tf.keras.Sequential([
        # 1st Conv1D
        Conv1D(8, 1, strides=1, activation='relu', input_shape=(1250, 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        # 2nd Conv1D
        Conv1D(16, 3, strides=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        # 3rd Conv1D
        Conv1D(32, 3, strides=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        # 4th Conv1D
        Conv1D(64, 3, strides=1, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2),
        Dropout(0.2),
        # 5th Conv1D
        Conv1D(16, 1, strides=1, activation='relu'),
        BatchNormalization(),
        # Full connection layer
        Flatten()
    ])

    X_input = Input(shape=input_shape)
    X = TimeDistributed(cnn)(X_input)
    X = Bidirectional(LSTM(32, return_sequences=True))(X)
    X = Bidirectional(LSTM(16))(X)
    X = Dense(classes, activation='softmax')(X)

    model = Model(inputs=[X_input], outputs=X)

    return model

def identity_block_18_1d(X, f, filters, stage, block):
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


def convolutional_block_18_1d(X, f, filters, stage, block, s=2):
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


def resnet18_1d(input_shape=(750, 1), classes=1, as_model=False):
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
    X = identity_block_18_1d(X, 3, [64, 64], stage=2, block='a')
    X = identity_block_18_1d(X, 3, [64, 64], stage=2, block='b')

    # Stage 3
    X = convolutional_block_18_1d(X, f=3, filters=[128, 128], stage=3, block='a', s=2)
    X = identity_block_18_1d(X, 3, [128, 128], stage=3, block='b')

    # Stage 4
    X = convolutional_block_18_1d(X, f=3, filters=[256, 256], stage=4, block='a', s=2)
    X = identity_block_18_1d(X, 3, [256, 256], stage=4, block='b')

    # Stage 5
    X = convolutional_block_18_1d(X, f=3, filters=[512, 512], stage=5, block='a', s=2)
    X = identity_block_18_1d(X, 3, [512, 512], stage=5, block='b')

    # AVGPOOL
    X = AveragePooling1D(2, name="avg_pool")(X)

    # output layer
    X = Flatten()(X)

    if as_model:
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet18')

    return model


def resnet18_lstm(Tx, n_a, n_s, input_image_size, classes=3):
    # define resnet
    resnet = resnet18_1d(input_shape=(input_image_size, 1), classes=classes, as_model=False)

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
    if epoch < 50:
        return 1e-4
    elif 50 <= epoch < 100:
        return 1e-5
    else:
        return 1e-6


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
