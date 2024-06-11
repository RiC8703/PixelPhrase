import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

vgg16_base_model=VGG16()
vgg16_model=Model(inputs=vgg16_base_model.inputs, outputs=vgg16_base_model.layers[-2].output)

model=tf.keras.models.load_model('working/PixelPhrase.keras')

with open('working/tokenizer.pkl', 'rb') as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

st.set_page_config(page_title="PixelPhrase")

st.title("PixelPhrase-Image Caption Generator")
st.markdown(
    "This model uses a pretrained model(VGG16) for extracting features from the images and a trained LSTM model is used to predict the next words of the caption."
    )

uploaded_image=st.file_uploader("Choose an image:", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    st.subheader("Uploaded Image")
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    st.subheader("Generated Caption:")
    with st.spinner("Generating caption..."):
        image=load_img(uploaded_image, target_size=(224, 224))
        image=img_to_array(image)
        image=image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image=preprocess_input(image)

        feature=vgg16_model.predict(image, verbose=0)

        max_length=35

        def idx_to_word(integer, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == integer:
                    return word
            return None
        
        def predict_caption(model, image, tokenizer, max_length):
            in_text = 'startseq'
            for i in range(max_length):
                sequence=tokenizer.texts_to_sequences([in_text])[0]
                sequence=pad_sequences([sequence], max_length)
                yhat=model.predict([image, sequence], verbose=0)
                yhat=np.argmax(yhat)
                word=idx_to_word(yhat, tokenizer)
                if word is None:
                    break
                in_text+=' ' + word
                if word=='endseq':
                    break
            return in_text
        
        generated_caption=predict_caption(model, feature, tokenizer, max_length)
        generated_caption=generated_caption.replace("startseq","").replace("endseq","")

    st.markdown(
        f'<div style="border-left: 6px solid #ccc; padding: 5px 20px; margin-top: 20px;">'
        f'<p style="font-style: italic;">“{generated_caption}”</p>'
        f'</div>',
        unsafe_allow_html=True
    )


