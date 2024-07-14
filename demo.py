import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.h5")  # Memuat model yang telah dilatih sebelumnya
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))  # Memuat gambar uji dengan ukuran target 64x64 piksel
    input_arr = tf.keras.preprocessing.image.img_to_array(image)  # Mengubah gambar menjadi array numpy
    input_arr = np.array([input_arr])  # Mengubah gambar tunggal menjadi batch array
    predictions = model.predict(input_arr)  # Melakukan prediksi menggunakan model
    return np.argmax(predictions)  # Mengembalikan indeks kelas dengan probabilitas tertinggi

st.header("Model Prediksi")  # Judul halaman prediksi
test_image = st.file_uploader("Pilih gambar:")  # Pemilih untuk mengunggah gambar uji

if st.button("Tampilkan gambar"):
    st.image(test_image, width=400, use_column_width=True)  # Tombol untuk menampilkan gambar yang diunggah

# Predict button
if st.button("Predik"):
    if test_image is not None:
        result_index = model_prediction(test_image)
        # Membaca label dari file labels.txt
        with open("labels.txt") as f:
            content = f.readlines()
        labels = [i.strip() for i in content]
        st.success("Ini adalah {}".format(labels[result_index]))  # Menampilkan hasil prediksi
