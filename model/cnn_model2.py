import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

def cnn_model2(num_classes, input_shape):
    input_layer = Input(shape=input_shape)

    # Initial Convolution
    x = Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual Blocks
    num_blocks_list = [2, 2, 2, 2]  
    filter_sizes = [32, 64, 128, 256]
    for i, num_blocks in enumerate(num_blocks_list):
        for _ in range(num_blocks):
            # Residual path
            residual = x

            # First convolutional layer
            x = Conv2D(filter_sizes[i], (3, 3), padding='same')(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)

            # Second convolutional layer
            x = Conv2D(filter_sizes[i], (3, 3), padding='same')(x)
            x = BatchNormalization()(x)

            # Shortcut connection
            if residual.shape[-1] != filter_sizes[i]:
                residual = Conv2D(filter_sizes[i], (1, 1), padding='same')(residual)
            x = tf.keras.layers.add([x, residual])
            x = ReLU()(x)

    # Global Average Pooling and Dense layers
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model