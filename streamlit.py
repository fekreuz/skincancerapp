import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import folium
import geocoder
import pydeck as pdk
import pandas as pd
import requests  

# load the pre-trained model
model = load_model('checkpoints/best_model_freeze150_friday.h5')

# labels for the classes
class_labels = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

# mapping of class labels to full categories
class_categories = {
    'AKIEC': "Actinic keratoses and intraepithelial carcinoma / Bowen's disease",
    'BCC': "Basal cell carcinoma",
    'BKL': "Benign keratosis-like lesions / Solar lentigines / Seborrheic keratoses and Lichen-planus like keratoses",
    'DF': "Dermatofibroma",
    'MEL': "Melanoma",
    'NV': "Melanocytic nevi",
    'VASC': "Vascular lesions (Angiomas, Angiokeratomas, Pyogenic Granulomas and Hemorrhage)",
}

# function to preprocess the uploaded image
def preprocess_image(image):
    # resize the image to 224x224
    image = image.resize((224, 224))
    # convert the image to a numpy array
    image_array = np.array(image)
    # expand dimensions to match the model input shape (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# function to convert prediction to percentages and find the highest prediction
def process_prediction(prediction):
    percentages = (prediction * 100).flatten()
    rounded_percentages = {class_categories[class_labels[i]]: round(percentages[i], 2) for i in range(len(class_labels))}
    sorted_percentages = dict(sorted(rounded_percentages.items(), key=lambda x: x[1], reverse=True))
    max_category = max(rounded_percentages, key=rounded_percentages.get)
    return sorted_percentages, max_category

# function to get dermatologist information based on GPS coordinates
def get_dermatologist_info(latitude, longitude):
    # Replace this with a real API endpoint that provides dermatologist information
    # Example: api_url = f"https://example-dermatologists-api.com/?lat={latitude}&lng={longitude}"
    # Make a request to the API and retrieve dermatologist information
    # response = requests.get(api_url)
    # if response.status_code == 200:
    #     return response.json()
    # else:
    #     return None

    # Placeholder for the dermatologist information
    return {
        "Name": "Dr. John Doe",
        "Address": "123 Dermatology Street, City, Country",
        "Phone": "+1 (555) 123-4567",
        "Website": "https://www.exampledermatology.com",
    }

# streamlit app
def main():
    st.title("Skin Cancer Detection App 🔬🧫")
    st.write("This app uses a pre-trained deep learning model to predict whether an uploaded skin image "
             "shows signs of skin cancer. The predictions are for demonstration purposes only and should not "
             "be considered as actual medical diagnoses.")
    
    # add acknowledgements
    st.subheader("Acknowledgements:")
    st.write("1. Tschandl, P., Rinner, C., Apalla, Z. et al. Human-computer collaboration for skin cancer recognition. "
             "Nat Med (2020). [Link](https://doi.org/10.1038/s41591-020-0942-0)")
    st.write("2. Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source "
             "dermatoscopic images of common pigmented skin lesions. Sci Data 5, 180161 (2018). "
             "[Link](https://doi.org/10.1038/sdata.2018.161)")
    
    # display the Streamlit model icon
    st.image("streamlit_model_icon.png", use_column_width=True)

    # upload an image file
    uploaded_image = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # preprocess the image
        st.write("Processing the image...")
        progress_bar = st.progress(0)
        for i in range(100):
            processed_image = preprocess_image(Image.open(uploaded_image))
            progress_bar.progress(i + 1)

        # make predictions using the model
        st.write("Making predictions...")
        prediction = model.predict(processed_image)

        # convert prediction to percentages and find the highest prediction
        percentages, max_category = process_prediction(prediction)

        # show the predictions with the highest category in green
        st.subheader("Predictions (in %):")
        prediction_df = pd.DataFrame(percentages.items(), columns=["Category", "Percentage"])
        prediction_df["Percentage"] = prediction_df["Percentage"].apply(lambda x: f"{x:.2f}%")
        st.dataframe(prediction_df.style.apply(lambda x: ["background: white" if x.name != max_category else "background: green" for i in x], axis=1), height=300)

        # get user's location using geocoder by IP 
        location = geocoder.ip('me')
        user_latitude, user_longitude = location.latlng

        # show the map with the user's location
        st.subheader("Nearest Dermatologist Location:")
        map_data = {
            "latitude": [user_latitude],
            "longitude": [user_longitude],
        }
        map_df = pd.DataFrame(map_data)
        st.pydeck_chart(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=user_latitude,
                longitude=user_longitude,
                zoom=15,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_df,
                    get_position="[longitude, latitude]",
                    get_color="[0, 255, 0, 100]",
                    get_radius=100,
                ),
            ],
        ))

        # get dermatologist information based on GPS coordinates
        dermatologist_info = get_dermatologist_info(user_latitude, user_longitude)
        if dermatologist_info:
            st.subheader("Dermatologist Information:")
            for key, value in dermatologist_info.items():
                st.write(f"**{key}:** {value}")

if __name__ == '__main__':
    main()

