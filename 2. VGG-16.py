from initializer import *
from load_dump import *

new_input = Input(shape=(image_size, image_size, 3))
model = VGG16(include_top=False, input_shape=(image_size, image_size, 3), weights='imagenet')

for layer in model.layers:
    layer.trainable = False

output_layer = Flatten()(model.output)
prediction = Dense(3, activation='softmax')(output_layer)

model = Model(inputs=model.inputs, outputs=prediction)

model.summary()


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


tensorboard = TensorBoard \
    (log_dir='logs/{}'.format
        (int(time.time())))



train_x = np.asarray(train_x).reshape(train_x.shape[0], image_size, image_size, 1)
train_x = tf.image.grayscale_to_rgb(tf.convert_to_tensor(train_x))

train_y = np.asarray(train_y)
train_y = to_categorical(train_y)

tensorboard = TensorBoard(log_dir='logs/VGG16'.format(int(time.time())))

model.fit(train_x, train_y, epochs=10, batch_size=128, validation_split=0.1, callbacks=[tensorboard])

#model.summary()
model.save('models/vgg/')