from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
from data_preprocessing import preprocess_custom_data, split_custom_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    precision = precision_score(y_true_classes, y_pred_classes)
    recall = recall_score(y_true_classes, y_pred_classes)
    f1 = f1_score(y_true_classes, y_pred_classes)

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    model_path = "models/happy_sad_cnn_model.h5"
    data_directory = "data/"
    img_size = 64
    num_classes = 2
    test_proportion = 0.2

    # Load and preprocess data
    images, labels = load_custom_data(data_directory, img_size)
    X, y = preprocess_custom_data(images, labels, num_classes)
    _, X_test, _, y_test = split_custom_data(X, y, test_proportion)

    # Load the trained model
    model = load_model(model_path)

    # Evaluate the model
    accuracy, precision, recall, f1 = evaluate_model(model, X_test, y_test)

    print("Evaluation Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
