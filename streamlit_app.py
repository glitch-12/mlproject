import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('my_model.h5')

# Title of the app
st.title('Handwriting Character Recognition')

# User can upload an image
file = st.file_uploader("Upload an image", type="jpg")

if file is not None:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image / 255.0

    # Predict the image
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)

    # Display the prediction
    st.write('Predicted class:', predicted_class[0])
