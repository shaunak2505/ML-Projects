import streamlit as st
import numpy as np
import cv2
from PIL import Image
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist

st.title("Multi-Digit Recognition using HOG + RandomForest")

# Load and train model
@st.cache_resource
def load_model():
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train, y_train = X_train[:10000], y_train[:10000]

    def preprocess_image(img):
        img = cv2.resize(img, (32, 32))
        img = img.astype(np.float32) / 255.0
        return img

    def extract_hog_features(img):
        return hog(img, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)

    X_train_prep = [preprocess_image(img) for img in X_train]
    X_train_feat = np.array([extract_hog_features(img) for img in X_train_prep])

    scaler = StandardScaler()
    X_train_feat = scaler.fit_transform(X_train_feat)

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train_feat, y_train)

    return clf, scaler

clf, scaler = load_model()

# Digit segmentation function
def segment_digits(image):
    """Segment digits using contours (assumes dark digits on light background)."""
    # Resize to make sure it handles multiple digits
    img = cv2.resize(image, (128, 32))

    # Convert to 8-bit image (values between 0 and 255)
    img = np.uint8(img * 255)

    # Thresholding the image to get binary representation (inverted black on white)
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours (individual digits)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_imgs = []
    for cnt in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 5 and h > 10:  # Filtering small noise or invalid contours
            digit = img[y:y + h, x:x + w]
            digit = cv2.resize(digit, (32, 32))  # Resize to match model input size
            digit_imgs.append(digit)
    return digit_imgs

# Upload image
uploaded_file = st.file_uploader("Upload an image with digits", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    img_np = np.array(image).astype(np.float32) / 255.0

    st.image(image, caption="Uploaded Image", width=200)

    try:
        digits = segment_digits(img_np)  # Segment the digits
        predictions = []
        for digit in digits:
            # Extract HOG features for each digit
            features = hog(digit, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
            features_scaled = scaler.transform([features])
            
            # Predict the digit using the trained classifier
            pred = clf.predict(features_scaled)[0]
            predictions.append(str(pred))

        if predictions:
            st.success(f"Predicted Number: {''.join(predictions)}")
        else:
            st.warning("No digits detected. Try a clearer image.")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
