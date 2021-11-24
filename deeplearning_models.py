import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Input, MaxPool2D, BatchNormalization, GlobalAvgPool2D, Flatten
from tensorflow.keras import Model

def streetsign_model(no_classes):

    my_input = Input(shape=(60,60,3))

    x = Conv2D(32, (3,3), activation='relu')(my_input)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3,3), activation='relu')(x)
    x = MaxPool2D()(x)
    x = BatchNormalization()(x)

    #x= Flatten()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(no_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

if __name__=="__main__":
    model = streetsign_model(10)
    model.summary()