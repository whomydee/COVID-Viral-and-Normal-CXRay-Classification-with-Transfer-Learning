from load_dump import *

test_x = np.asarray(test_x).reshape(test_x.shape[0], image_size, image_size, 1)
test_x = tf.image.grayscale_to_rgb(tf.convert_to_tensor(test_x))
test_y = to_categorical(test_y)

loss = []
accuracy = []
x_labels = ['VGG16', 'Inception V3', 'EfficientNetB7', 'ResNet152', 'MobileNet']


vgg = tf.keras.models.load_model('models/vgg')
tmp = np.round(vgg.evaluate(test_x, test_y), 2)
loss.append(tmp[0])
accuracy.append(tmp[1])


inception = tf.keras.models.load_model('models/inceptionv3')
tmp = np.round(inception.evaluate(test_x, test_y), 2)
loss.append(tmp[0])
accuracy.append(tmp[1])


resnet = tf.keras.models.load_model('models/resnet152')
tmp = np.round(resnet.evaluate(test_x, test_y), 2)
loss.append(tmp[0])
accuracy.append(tmp[1])

efficientNet = tf.keras.models.load_model('models/efficientnetb7')
tmp = np.round(efficientNet.evaluate(test_x, test_y), 2)
loss.append(tmp[0])
accuracy.append(tmp[1])

mobileNet = tf.keras.models.load_model('models/mobilenet')
tmp = np.round(mobileNet.evaluate(test_x, test_y), 2)
loss.append(tmp[0])
accuracy.append(tmp[1])

print(loss)
print(accuracy)
print(x_labels)