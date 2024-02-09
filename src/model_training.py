import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_custom_data, preprocess_custom_data, split_custom_data

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

if __name__ == "__main__":
    data_directory = "data/"
    img_size = 64
    num_classes = 2
    test_proportion = 0.2
    epochs = 20
    batch_size = 32

    # Load and preprocess data
    images, labels = load_custom_data(data_directory, img_size)
    images, labels = preprocess_custom_data(images, labels, num_classes)
    X_train, X_test, y_train, y_test = split_custom_data(images, labels, test_proportion)

    # Build the model
    input_shape = (img_size, img_size, 3)
    model = build_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

    # Save the model
    model.save("models/happy_sad_cnn_model.h5")

    print("Model training complete.")
