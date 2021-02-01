from initializer import *

tmp = open("dumps/128/training_x_dump.pickle", "rb")
train_x = pickle.load(tmp)

tmp = open("dumps/128/training_y_dump.pickle", "rb")
train_y = pickle.load(tmp)

tmp = open("dumps/128/testing_x_dump.pickle", "rb")
test_x = pickle.load(tmp)

tmp = open("dumps/128/testing_y_dump.pickle", "rb")
test_y = pickle.load(tmp)

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

"""### Normalizing the Data and other Initialization"""

train_x /= 255.0
test_x /= 255.0