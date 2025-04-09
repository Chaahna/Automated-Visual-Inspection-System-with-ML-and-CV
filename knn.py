import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Define dataset paths
data_dir_train = "dataset/train"
data_dir_test = "dataset/test"
categories = ["freshapples", "freshbanana", "freshoranges", "rottenapples", "rottenbanana", "rottenoranges"]

# Image size for resizing
img_size = 100

# Load data function
def load_data(data_dir):
    data = []
    labels = []
    for category in categories:
        path = os.path.join(data_dir, category)
        label = 0 if "fresh" in category else 1
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img_resized = cv2.resize(img_array, (img_size, img_size))
                data.append(img_resized)
                labels.append(label)
            except Exception as e:
                pass
    return np.array(data), np.array(labels)

# Load the training dataset
X_train, y_train = load_data(data_dir_train)

# Load the testing dataset
X_test, y_test = load_data(data_dir_test)

# Feature extraction using color and texture analysis
def extract_features(image):
    # Convert image to HSV for color analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Calculate histogram for hue channel
    hist_hue = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    # Normalize histogram
    hist_hue = cv2.normalize(hist_hue, hist_hue).flatten()

    # Texture analysis using Laplacian variance (sharpness)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Combine features
    features = np.concatenate([hist_hue, [laplacian_var]])
    return features

# Extract features for all training images
train_features = [extract_features(img) for img in X_train]
train_features = np.array(train_features)

# Extract features for all testing images
test_features = [extract_features(img) for img in X_test]
test_features = np.array(test_features)

# Train a K-Nearest Neighbors classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(train_features, y_train)

# Test the classifier
y_pred = knn.predict(test_features)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classifier Accuracy: {accuracy * 100:.2f}%")

# Function to classify new fruit images
def classify_fruit(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_features = extract_features(img_resized).reshape(1, -1)
    prediction = knn.predict(img_features)
    return "Fresh" if prediction[0] == 0 else "Rotten"

# Example usage
new_image_path = "dataset/dataset/test/rottenapples/apple1.png"  # Update with your actual path
print("image:", new_image_path)
result = classify_fruit(new_image_path)
print(f"The fruit is: {result}")

# Using the mechArm 270Pi to pick fresh or rotten fruit
