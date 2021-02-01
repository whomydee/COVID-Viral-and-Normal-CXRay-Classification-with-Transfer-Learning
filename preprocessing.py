from initializer import *

rows = []
for i in range(image_size * image_size):
    rows.append(i)

rows.append('Label')
training_data = pd.DataFrame()
testing_data = pd.DataFrame()


def append_in_dataframe(df, image, label, ):
    image = np.append(image, label)
    image = image.flatten()
    image = image.T
    df = pd.concat([df, pd.Series(image)], axis=1)
    return df

"""### Loading all the images into the DataFrame."""


# Pre-processing for COVID

covid_filenames = os.listdir(location_covid_images)

for image in progressbar(covid_filenames):
  tmp_image = cv2.imread((location_covid_images + image), cv2.IMREAD_GRAYSCALE)
  tmp_image = cv2.resize(tmp_image, (image_size, image_size))
  training_data = append_in_dataframe(training_data, tmp_image, 0)


# Pre-processing for VIRAL

viral_filenames = os.listdir(location_viral_images)

for image in progressbar(viral_filenames):
  tmp_image = cv2.imread((location_viral_images + image), cv2.IMREAD_GRAYSCALE)
  tmp_image = cv2.resize(tmp_image, (image_size, image_size))
  training_data = append_in_dataframe(training_data, tmp_image, 1)


# Pre-processing for NORMAL

normal_filenames = os.listdir(location_normal_images)

for image in progressbar(normal_filenames):
  tmp_image = cv2.imread((location_normal_images + image), cv2.IMREAD_GRAYSCALE)
  tmp_image = cv2.resize(tmp_image, (image_size, image_size))
  training_data = append_in_dataframe(training_data, tmp_image, 2)

print("Pre-processing Done: Training")

training_data.tail()

"""###Shuffling the DataFrame"""

shuffled_training_data = copy.deepcopy(training_data.T)
shuffled_training_data = shuffled_training_data.sample(frac=1).reset_index(drop=True)

shuffled_training_data.shape

"""### Partitioning for Training data and Label"""

training_x_df = shuffled_training_data.iloc[:, :-1]
training_y_df = shuffled_training_data.iloc[:, -1:]

"""### Doing the same things for Testing Images"""

location_test_covid_images = location_dataset + "test/" + "covid/"
location_test_viral_images = location_dataset  + "test/" + "viral/"
location_test_normal_images = location_dataset  + "test/" + "normal/"


# Pre-processing for COVID

covid_test_filenames = os.listdir(location_test_covid_images)

for image in progressbar(covid_test_filenames):
  tmp_image = cv2.imread((location_test_covid_images + image), cv2.IMREAD_GRAYSCALE)
  tmp_image = cv2.resize(tmp_image, (image_size, image_size))
  testing_data = append_in_dataframe(testing_data, tmp_image, 0)


# Pre-processing for VIRAL

viral_test_filenames = os.listdir(location_test_viral_images)

for image in progressbar(viral_test_filenames):
  tmp_image = cv2.imread((location_test_viral_images + image), cv2.IMREAD_GRAYSCALE)
  tmp_image = cv2.resize(tmp_image, (image_size, image_size))
  testing_data = append_in_dataframe(testing_data, tmp_image, 1)


# Pre-processing for NORMAL

normal_filenames = os.listdir(location_test_normal_images)

for image in progressbar(normal_filenames):
  tmp_image = cv2.imread((location_test_normal_images + image), cv2.IMREAD_GRAYSCALE)
  tmp_image = cv2.resize(tmp_image, (image_size, image_size))
  testing_data = append_in_dataframe(testing_data, tmp_image, 2)

print("Pre-processing Done: Testing")

shuffled_testing_data = testing_data.T

testing_x_df = shuffled_testing_data.iloc[:, :-1]
testing_y_df = shuffled_testing_data.iloc[:, -1:]

"""###Creating dump after Preprocessing
I will start working with the pre-processed data directly from now on.
"""

training_x_dump = open("dumps/128/training_x_dump.pickle", "wb")
pickle.dump(training_x_df, training_x_dump)

training_y_dump = open("dumps/128/training_y_dump.pickle", "wb")
pickle.dump(training_y_df, training_y_dump)

testing_x_dump = open("dumps/128/testing_x_dump.pickle", "wb")
pickle.dump(testing_x_df, testing_x_dump)

testing_y_dump = open("dumps/128/testing_y_dump.pickle", "wb")
pickle.dump(testing_y_df, testing_y_dump)