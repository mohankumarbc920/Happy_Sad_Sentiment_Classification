import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def load_custom_data(data_path, image_size):
    
    images = []
    labels = []

    for category in os.listdir(data_path):
        category_path = os.path.join(data_path, category)
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (image_size, image_size))
            images.append(image)
            labels.append(category)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

def preprocess_custom_data(images, labels, num_classes):
   
    images = images.astype('float32') / 255.0
    labels = to_categorical(labels, num_classes)

    return images, labels

def split_custom_data(images, labels, test_size):
   
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    data_directory = "data/"
    image_dimension = 64
    num_classes = 2
    test_proportion = 0.2

    # Load and preprocess the custom image data
    images, labels = load_custom_data(data_directory, image_dimension)
    images, labels = preprocess_custom_data(images, labels, num_classes)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = split_custom_data(images, labels, test_proportion)

    print("Custom data preprocessing complete.")
