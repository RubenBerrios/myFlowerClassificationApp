import os
import keras
from keras.models import load_model
import streamlit as st
import tensorflow as tf
import numpy as np

allFlowerNames = ['daisy','dandelion','roses','sunflowers','tulips']

#load the model created using the jupyter notebook
model = load_model('flower_classify_model.keras')

#create tabs for main page and visualization
tab1, tab2 = st.tabs(["Prediction", "Visualization"])

#tab for uploading images to predict
with tab1:
    st.title('Plant Classification model')
    st.text('List of plants the model was trained to detect:')
    s = ''
    for i in allFlowerNames:
        s += "- " + i + "\n"
    st.markdown(s)
    

    #model prediction method
    def classify_images(image_path):
        input_image = tf.keras.utils.load_img(
            image_path,
            color_mode='rgb', 
            target_size=(180,180),
            interpolation='nearest',
            keep_aspect_ratio=False)
        input_image_array = tf.keras.utils.img_to_array(input_image)
        input_image_exp_dim = tf.expand_dims(input_image_array,0)
        predictions = model.predict(input_image_exp_dim)
        result = tf.nn.softmax(predictions[0])
        outcome = 'This Plant type is: ' + allFlowerNames[np.argmax(result)]
        return outcome

    #upload file
    uploaded_file = st.file_uploader('Choose a file to upload', type=["jpeg"])
    if uploaded_file is not None:
        #display the image on web app
        st.image(uploaded_file, width=240)
        st.info(classify_images(uploaded_file))
        

#tab for displaying visualizations
with tab2:
    st.title("Visualizations")
    st.image('Visualization1.jpeg', width=500)
    st.markdown("""
                This visualization shows the number of images each plant has in the dataset. This chart helped us identify any plant class imbalances in our dataset. 
                We can see that the dandelion had the highest amount of images in the dataset with 898 images. We can also see that daisy has the minimum amount of 
                images in the dataset with 633 images. Training a machine learning model on an imbalanced dataset may result in outcomes that are biased toward certain plants.
                """)
    st.image('Visualization2.jpeg', width=500)
    st.markdown("""
                The distribution of image sizes in the dataset allowed us to gain insight into our dataset. We can see that a vast majority of the images 
                in the dataset had a file size less than 50,000 bytes(0.05MB)
                """)
    st.image('Visualization3.jpeg', width=500)
    st.markdown("""
                For each image in the dataset, we extracted the independent average RGB value over all the pixels, grouped them together per Flower, 
                and were able to calculate the average RGB value per Flower. This 3D graph shows that Tulips and Roses are really close in color, 
                possibly making it harder for the Machine Learning model to distinguish between one another. Whereas Daisy, Dandelion, and Sunflowers 
                are more spread apart from one another.
                """)
    col1,col2 = st.columns(2)
    with col1:
        st.image('TrainingAndValidationAccuracy.jpeg', width=400)
    
    with col2:
        st.image('TrainingAndValidationLoss.jpeg',width=400)
    st.markdown("""
                The dataset that was collected was split into an 80/20 split; 80% of the dataset was used for training the model, and 20% was used for testing. 
                The results of using this validation method showed the training accuracy of the model was  86%, validation accuracy was 57%. The training loss was 39% and the validation loss was over 140%. 
                





""")