import cv2
import numpy as np
from tensorflow.keras.models import load_model
from data_preprocessing import preprocess_custom_data

def load_image(file_path, img_size):
    image = cv2.imread(file_path)
    image = cv2.resize(image, (img_size, img_size))
    return image

def predict_emotion(model, image):
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    if prediction[0][0] > prediction[0][1]:
        return "Happy"
    else:
        return "Sad"

if __name__ == "__main__":
    model_path = "models/happy_sad_cnn_model.h5"
    img_size = 64
    example_image_path = "data/test/happy/happy_test_image1.jpg"

    # Load the trained model
    model = load_model(model_path)

    # Load an example image for inference
    image = load_image(example_image_path, img_size)

    # Preprocess the image
    preprocessed_image = preprocess_custom_data(np.array([image]), np.array([0]), 2)[0]

    # Perform inference
    emotion = predict_emotion(model, preprocessed_image)

    print("Predicted emotion:", emotion)
