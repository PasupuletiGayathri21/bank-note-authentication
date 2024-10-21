import streamlit as st
import joblib
import numpy as np
from scipy.stats import kurtosis, skew
from PIL import Image
import io
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('banknote_auth_model.joblib')

# Load the scaler used during training (you need to save this during model training)
scaler = joblib.load('feature_scaler.joblib')

st.title('Bank Note Authenticator')

st.write("""
This app uses machine learning to predict whether a bank note is authentic or counterfeit based on its image features.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a bank note image...", type=["jpg", "jpeg", "png"])

def extract_features(image):
    # Convert image to grayscale
    gray_image = image.convert('L')
    
    # Resize image to ensure consistent processing
    gray_image = gray_image.resize((200, 400))
    
    # Convert to numpy array
    img_array = np.array(gray_image)
    
    # Flatten the image
    img_flat = img_array.flatten()
    
    # Calculate features
    variance = np.var(img_flat)
    skewness_value = skew(img_flat)
    kurtosis_value = kurtosis(img_flat)
    
    # Calculate entropy
    histogram, _ = np.histogram(img_flat, bins=256, range=(0, 255))
    histogram = histogram / float(np.sum(histogram))
    entropy_value = -np.sum(histogram * np.log2(histogram + 1e-7))
    
    return np.array([variance, skewness_value, kurtosis_value, entropy_value])

if uploaded_file is not None:
    try:
        # Read the image
        image = Image.open(io.BytesIO(uploaded_file.read()))
        
        # Display the uploaded image
        st.image(image, caption='Uploaded Bank Note Image', use_column_width=True)
        
        # Extract features
        features = extract_features(image)
        
        # Scale features
        scaled_features = scaler.transform(features.reshape(1, -1))
        
        # Display calculated features
        st.write("Calculated Features (Before Scaling):")
        feature_names = ['Variance', 'Skewness', 'Kurtosis', 'Entropy']
        for name, value in zip(feature_names, features):
            st.write(f"{name}: {value:.6f}")
        
        st.write("\nScaled Features:")
        for name, value in zip(feature_names, scaled_features[0]):
            st.write(f"{name}: {value:.6f}")
        
        # Make prediction
        prediction = model.predict(scaled_features)
        proba = model.predict_proba(scaled_features)
        
        # Display the prediction
        st.write("## Prediction")
        if prediction[0] == 1:
            st.success(f'The bank note is predicted to be authentic (confidence: {proba[0][1]:.2%})')
        else:
            st.error(f'The bank note is predicted to be counterfeit (confidence: {proba[0][0]:.2%})')
        
        # Display prediction probability
        st.write(f'Probability of being authentic: {proba[0][1]:.2%}')
        st.write(f'Probability of being counterfeit: {proba[0][0]:.2%}')
        
        # Display model's decision function if available
        if hasattr(model, 'decision_function'):
            decision = model.decision_function(scaled_features)
            st.write(f"Model's decision function value: {decision[0]:.4f}")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

st.write("""
### Feature Information:
- Variance: Variance of the image pixel values (continuous)
- Skewness: Skewness of the image pixel values (continuous)
- Kurtosis: Kurtosis of the image pixel values (continuous)
- Entropy: Entropy of the image (continuous)

Note: Features are calculated directly from the grayscale image pixel values to closely match the original dataset.
""")

st.write("""
### Troubleshooting:
If the model is not correctly recognizing counterfeit notes:
1. Ensure the uploaded image is clear, well-lit, and shows the entire banknote.
2. Check that the calculated feature values (before scaling) are within expected ranges.
3. If issues persist, the model may need to be retrained with a more diverse dataset that includes a wide range of counterfeit notes.
""")