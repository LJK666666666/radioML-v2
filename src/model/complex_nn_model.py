import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization, Activation, Permute
from keras.optimizers import Adam

def build_complex_nn_model(input_shape, num_classes):
    """
    Build a Complex-like NN model for radio signal classification.
    Processes I/Q data using 1D convolutions.
    
    Args:
        input_shape: Input shape of the data (channels, sequence_length)
        num_classes: Number of classes to classify
        
    Returns:
        A compiled Keras model
    """
    model = Sequential()
    # Input shape (2, 128) -> Permute to (128, 2)
    model.add(Permute((2, 1), input_shape=input_shape))

    model.add(Conv1D(filters=128, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=256, kernel_size=5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.6)) 
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model
