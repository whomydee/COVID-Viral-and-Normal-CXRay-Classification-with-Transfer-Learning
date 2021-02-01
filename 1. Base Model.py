from initializer import *
from load_dump import *

def create_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(image_size, image_size, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(12, activation='relu'))

    model.add(Dense(3, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])
    return model


basic_model = create_model()
# basic_model.summary()


tensorboard = TensorBoard \
    (log_dir='C:/Users/shad_/Desktop/2. Fall 2020/CSE 6211 (Deep Learning)/Submission/Project/Covid Detection with UNet/logs/{}'.format
        (int(time.time())))

train_x = np.asarray(train_x).reshape(train_x.shape[0], image_size, image_size, 1)
train_y = np.asarray(train_y)

train_y = to_categorical(train_y)
basic_model.fit(train_x, train_y, batch_size=32, epochs=20, validation_split=0.1, callbacks=[tensorboard])

basic_model.save \
    ('C:/Users/shad_/Desktop/2. Fall 2020/CSE 6211 (Deep Learning)/Submission/Project/Covid Detection with UNet/models/basic_model')