import os
import shutil
import pandas as pd


class PrepareImageDirectory:
    def __init__(self, train_ratio, data_path, images_path, train_path, val_path):
        self.train_ratio = train_ratio
        self.data_path = data_path
        self.images_path = images_path
        self.train_path = train_path
        self.val_path = val_path


    def build_train_val_directory(self):
        '''
        Split images from original download into new train-validation split directory with paths:
        parent/train/class1, parent/train/class2
        parent/val/class1, parent/val/class2

        Inputs kept generic so directory names can be chosen higher up.
        '''
        paths = self.define_paths()

        # create directories
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

        covid_image_path, other_image_path, images_path, train_path, val_path = paths

        self.populate_class_directory(covid_image_path, other_image_path, images_path)

        class_labels = os.listdir(images_path)

        self.populate_train_val_directory(class_labels, images_path, train_path, val_path)


    def define_paths(self):
        # paths to store images after being split by class
        covid_image_path = os.path.join(self.data_path, 'split_images/covid')
        other_image_path = os.path.join(self.data_path, 'split_images/other')

        # image paths for train and validation split
        full_images_path = os.path.join(self.data_path, self.images_path)
        full_train_path = os.path.join(self.data_path, self.train_path)
        full_val_path = os.path.join(self.data_path, self.val_path)

        return [covid_image_path, other_image_path, full_images_path, full_train_path, full_val_path]



    def populate_class_directory(self, covid_image_path, other_image_path, images_path):

        print('Spliting images into "covid" and "other"')

        metadata = pd.read_csv(os.path.sep.join([self.data_path, 'metadata.csv']))
        print(metadata.head(3))

        for i, row in metadata.iterrows():
            if (row['finding'] == "COVID-19") & (row['view'] == "PA") & (row['modality'] == 'X-ray'):

                image_filename = row['filename']

                # build the path to the input image file
                individual_image_path = os.path.sep.join([self.data_path, 'images', image_filename])

                # pass if image path doesn't exist, assuming dataset error
                if not os.path.exists(individual_image_path):
                    continue

                # path to copy images to
                output_path = os.path.sep.join([covid_image_path, image_filename])
                print(individual_image_path, output_path)
                # copy the image across
                shutil.copy(individual_image_path, output_path)

        # building 'other' image directory
        for i, row in metadata.iterrows():
            if (row['finding'] != "COVID-19") & (row['view'] == "PA") & (row['modality'] == 'X-ray'):

                # build the path to the input image file
                individual_image_path = os.path.sep.join([self.data_path, 'images', row["filename"]])

                # pass if image path doesn't exist, assuming dataset error
                if not os.path.exists(individual_image_path):
                    continue

                # path to copy images to
                image_filename = row['filename']
                output_path = os.path.sep.join([other_image_path, image_filename])

                # copy the image across
                shutil.copy(individual_image_path, output_path)



    def populate_train_val_directory(self, classes, images_path, train_path, val_path):

        for c in classes:

            class_path = os.path.join(images_path, c)
            images = os.listdir(class_path)
            n_train = int(len(images) * self.train_ratio)

            train_images = images[:n_train]
            test_images = images[n_train:]

            os.makedirs(os.path.join(train_path, c), exist_ok = True)
            os.makedirs(os.path.join(val_path, c), exist_ok = True)

            for image in train_images:
                image_src = os.path.join(class_path, image)
                image_destination = os.path.join(train_path, c, image)
                shutil.copyfile(image_src, image_destination)

            for image in test_images:
                image_src = os.path.join(class_path, image)
                image_destination = os.path.join(val_path, c, image)
                shutil.copyfile(image_src, image_destination)
