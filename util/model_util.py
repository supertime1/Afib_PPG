import tensorflow as tf
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import io
import itertools
import keras

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
    tf.keras.layers.Dense(3, activation='sigmoid')
])


def confusion_matrix_callback(log_dir, model, test_signals, test_labels, class_names=['NO Afib', 'Afib', 'Other']):

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
        plt.ylim(bottom=1.5, top=-0.5)

        if normalize:
            cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 1.5

        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     verticalalignment='center',
                     color="white" if cm[i, j] > threshold else "black")

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

    def log_confusion_matrix(epoch, logs):
        # Use the model to predict the values from the test_images.
        test_pred_raw = model.predict(test_signals)

        test_pred = np.where(test_pred_raw > 0.5, 1, 0)

        # Calculate the confusion matrix using sklearn.metrics
        cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)

        figure = plot_confusion_matrix(cm, class_names=class_names, normalize=True)
        cm_image = plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)

    file_writer_cm = tf.summary.create_file_writer(log_dir + '/cm')
    cm_callback = keras.callbacks.LambdaCallback(on_epoch_end=log_confusion_matrix)

    return cm_callback


def decay(epoch):
    if epoch < 100:
        return 1e-3
    elif 100 <= epoch < 200:
        return 1e-4
    else:
        return 1e-5








