from keras import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPool1D, MaxPool2D, CuDNNLSTM

def nn_classifier(nb_classes, in_dimensions, two_d):

    if two_d:
        kernels = [map(int, item) for item in[(in_dimensions[0]/4, in_dimensions[1]/4), (in_dimensions[0]/8, in_dimensions[1]/8),
                   (1, in_dimensions[1]/16)]]
    else:
        kernels = [int(item) for item in [in_dimensions[0]/8, in_dimensions[0]/16, in_dimensions[0]/32]]
    filters = [8, 16, 32]

    nnet = Sequential(name='RCNN')
    nnet.add(Dense(units=200, input_shape=(in_dimensions), activation='relu'))
    for idx in range(len(filters)):
        if two_d:
            nnet.add(Conv2D(filters=filters[idx], kernel_size=kernels[idx], padding='same', activation='relu'))
            nnet.add(MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
        else:
            nnet.add(Conv1D(filters=filters[idx], kernel_size=kernels[idx], padding='same', activation='relu'))
            nnet.add(MaxPool1D(pool_size=3, strides=2))
    nnet.add(CuDNNLSTM(units=100, unit_forget_bias=True))
    nnet.add(Dense(units=1000, activation='relu'))
    nnet.add(Dense(units=nb_classes, activation='softmax'))

    return nnet
