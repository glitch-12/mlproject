import streamlit as st
from PIL import Image
import numpy as np
import cv2

# Load the saved model
# model = tf.keras.models.load_model('my_model.h5')
model = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000_fp16.caffemodel')

# Title of the app
st.title('हिंदी हैंडव्रिटिंग डेटेक्शन')

# User can upload an image
file = st.file_uploader("अपलोड करें एक छवि", type="jpg")

if file is not None:
    # Display the uploaded image
    image = Image.open(file)
    st.image(image, caption='अपलोड की छवि।', use_column_width=True)

    # Preprocess the image
    image = image.resize((300, 300))
    image = np.array(image)

    # Prepare the image for the model
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Perform the handwriting detection
    model.setInput(blob)
    output_layers_names = model.getUnconnectedOutLayersNames()
    layer_outputs = model.forward(output_layers_names)

    # Get the detected handwriting
    detected_handwriting = layer_outputs[0]
    detected_handwriting = np.squeeze(detected_handwriting)

    (h, w) = detected_handwriting.shape[:2]
    center = (w // 2, h // 2)
    radius = int(3 * (h / 4))
    mask = np.zeros((h, w), dtype="uint8")
    cv2.circle(mask, center, radius, 255, -1)
    detected_handwriting = cv2.bitwise_and(detected_handwriting, detected_handwriting, mask=mask)

    (c, _) = cv2.mean(detected_handwriting)
    detected_handwriting = (detected_handwriting > (c - 50)).astype("uint8")

    # Display the detection
    st.write('检测到的手写：', detected_handwriting)
