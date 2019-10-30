# To make a CNN capable of classifying images from the Cifar 10 Dataset

from keras.datasets import cifar10
from keras.models import Sequential
from keras.applications import VGG16
from keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

batch_size = 35
epochs = 111
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.reshape(X_train.shape[0], 32, 32, 3)
X_test = X_test.reshape(X_test.shape[0], 32, 32, 3)
#one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]  # 10

#define a CNN model based off of the VGG16 CNN from Keras

def extended_model():
    #create extended CNN model with VGG16
    pretrained = VGG16(include_top=False, input_shape=(32, 32, 3), pooling="max", weights='imagenet')
    pretrained.trainable = False
    model = Sequential()
    model.add(pretrained)
    model.add(Dropout(0.25))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile Model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#build the model
model = extended_model()
model.summary()
#Fit the model with checkpoint
filepath = "weights.{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)
#Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
