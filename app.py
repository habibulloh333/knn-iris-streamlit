
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier

# =====================================
# 1. KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="Aplikasi Prediksi Iris",
    page_icon="🌸",
    layout="centered"
)

# Lokasi folder project
BASE_DIR = Path(__file__).resolve().parent

# Nama file dataset dan model
DATA_FILE = BASE_DIR / "data_training.csv"
MODEL_FILE = BASE_DIR / "model_iris_knn.pkl"

# Fitur yang digunakan
FITUR = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

# =====================================
# 2. MELATIH DAN MENYIMPAN MODEL KNN
# =====================================

@st.cache_resource
def load_model():

    # Membaca dataset training
    data = pd.read_csv(DATA_FILE)

    # Memastikan semua fitur tersedia
    for kolom in FITUR:
        if kolom not in data.columns:
            raise ValueError(
                f"Kolom {kolom} tidak ditemukan dalam dataset."
            )

    # Mencari kolom kelas
    kolom_kelas = [
        kolom for kolom in data.columns
        if kolom not in FITUR + ["Id", "Status"]
        and not kolom.startswith("Unnamed:")
    ]

    if len(kolom_kelas) != 1:
        raise ValueError(
            "Kolom kelas tidak dapat ditentukan. "
            f"Kolom yang ditemukan: {kolom_kelas}"
        )

    target = kolom_kelas[0]

    # Memisahkan data fitur dan kelas
    X_train = data[FITUR]
    y_train = data[target]

    # Memastikan jumlah data mencukupi
    if len(X_train) < 3:
        raise ValueError("Data training minimal harus memiliki 3 baris.")

    # Membuat model KNN dengan K=3
    model = KNeighborsClassifier(n_neighbors=3)

    # Melatih model
    model.fit(X_train, y_train)

    # Menyimpan model dalam format Pickle
    joblib.dump(model, MODEL_FILE)

    return model


# =====================================
# 3. MEMUAT MODEL
# =====================================

try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "File data_training.csv tidak ditemukan. "
        "Pastikan file berada satu folder dengan app.py."
    )
    st.stop()

except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat model: {e}")
    st.stop()


# =====================================
# 4. TAMPILAN APLIKASI
# =====================================

st.title("Aplikasi Prediksi Iris - Model KNN")

st.write(
    "Masukkan nilai fitur untuk memprediksi jenis bunga Iris."
)

# =====================================
# 5. INPUT DATA PENGGUNA
# =====================================

sepal_length = st.number_input(
    "Sepal Length",
    min_value=0.0,
    max_value=10.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width",
    min_value=0.0,
    max_value=10.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length",
    min_value=0.0,
    max_value=10.0,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width",
    min_value=0.0,
    max_value=10.0,
    value=0.2,
    step=0.1
)


# =====================================
# 6. PROSES PREDIKSI
# =====================================

if st.button("Prediksi"):

    # Menggabungkan input menjadi satu data
    data_baru = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=FITUR
    )

    # Melakukan prediksi menggunakan model KNN
    prediction = model.predict(data_baru)[0]

    # Mengubah hasil prediksi menjadi nama kelas
    label = str(prediction).lower()

    if "setosa" in label or label == "1":
        result = "Setosa"

    elif "versicolor" in label or label == "2":
        result = "Versicolor"

    elif "virginica" in label or label == "3":
        result = "Virginica"

    else:
        result = str(prediction)

    # Menampilkan hasil prediksi
    st.success(f"Hasil Prediksi: {result}")
