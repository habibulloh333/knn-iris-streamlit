
import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================

st.set_page_config(
    page_title="Klasifikasi Bunga Iris",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Klasifikasi Bunga Iris")
st.write(
    "Aplikasi klasifikasi bunga Iris menggunakan "
    "algoritma k-Nearest Neighbors (k-NN) "
    "dengan metode Euclidean Distance."
)

st.divider()


# ==========================================
# 2. MEMBACA DATASET
# ==========================================

@st.cache_data
def load_data():
    training = pd.read_csv("data_training.csv")
    testing = pd.read_csv("data_testing.csv")

    return training, testing


try:
    training_df, testing_df = load_data()

except FileNotFoundError:
    st.error(
        "File dataset tidak ditemukan. "
        "Pastikan data_training.csv dan "
        "data_testing.csv berada dalam folder app.py."
    )
    st.stop()


# ==========================================
# 3. MENENTUKAN ATRIBUT
# ==========================================

fitur = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

# Mengambil atribut data Training
X_train = training_df[fitur].to_numpy(dtype=float)

# Mengambil kelas data Training
kelas = training_df["Species"].to_numpy()

# Mengambil atribut data Testing
X_test = testing_df[fitur].to_numpy(dtype=float)

# Mengambil kelas sebenarnya
kelas_asli = testing_df["Species"].to_numpy()

# Mengambil ID data Testing
id_testing = testing_df["Id"].to_numpy()

# Menentukan nilai K
k = 3


# ==========================================
# 4. FUNGSI EUCLIDEAN DISTANCE
# ==========================================

def euclidean_distance(xi, yi):
    return np.sqrt(np.sum((xi - yi) ** 2))


# ==========================================
# 5. FUNGSI ALGORITMA K-NN
# ==========================================

def k_nearest_neighbors(X_train, kelas, x_test, k):

    distances = []

    for i in range(len(X_train)):

        distance = euclidean_distance(
            x_test,
            X_train[i]
        )

        distances.append(
            (X_train[i], distance, kelas[i])
        )

    # Mengurutkan berdasarkan jarak terkecil
    distances.sort(key=lambda x: x[1])

    # Mengambil K tetangga terdekat
    neighbors = distances[:k]

    # Menghitung suara setiap kelas
    class_votes = {}

    for neighbor in neighbors:

        label = neighbor[2]

        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1

    # Menentukan kelas dengan suara terbanyak
    prediction = max(
        class_votes,
        key=class_votes.get
    )

    return prediction


# ==========================================
# 6. MELAKUKAN PREDIKSI DATA TESTING
# ==========================================

hasil_prediksi = []

for i in range(len(X_test)):

    klasifikasi = k_nearest_neighbors(
        X_train,
        kelas,
        X_test[i],
        k
    )

    hasil_prediksi.append(klasifikasi)


# ==========================================
# 7. MENGHITUNG AKURASI
# ==========================================

jumlah_benar = np.sum(
    np.array(hasil_prediksi) == kelas_asli
)

akurasi = (
    jumlah_benar / len(kelas_asli)
) * 100


# ==========================================
# 8. MENAMPILKAN INFORMASI MODEL
# ==========================================

st.subheader("Informasi Model")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Data Training",
    len(X_train)
)

col2.metric(
    "Data Testing",
    len(X_test)
)

col3.metric(
    "Nilai K",
    k
)


# ==========================================
# 9. MENAMPILKAN HASIL EVALUASI
# ==========================================

st.subheader("Evaluasi Model")

col1, col2 = st.columns(2)

col1.metric(
    "Akurasi Model",
    f"{akurasi:.2f}%"
)

col2.metric(
    "Prediksi Benar",
    f"{jumlah_benar}/{len(kelas_asli)}"
)


# ==========================================
# 10. TABEL HASIL PREDIKSI DATA TESTING
# ==========================================

st.subheader("Hasil Klasifikasi Data Testing")

hasil_testing = pd.DataFrame({
    "ID Testing": id_testing,
    "Kelas Aktual": kelas_asli,
    "Hasil Prediksi": hasil_prediksi
})

hasil_testing["Keterangan"] = np.where(
    hasil_testing["Kelas Aktual"] ==
    hasil_testing["Hasil Prediksi"],
    "Benar",
    "Salah"
)

st.dataframe(
    hasil_testing,
    use_container_width=True,
    hide_index=True
)


# ==========================================
# 11. MENAMPILKAN TETANGGA TERDEKAT
# ==========================================

st.subheader("3 Tetangga Terdekat Data Testing")

for i in range(len(X_test)):

    distances = []

    for j in range(len(X_train)):

        distance = euclidean_distance(
            X_test[i],
            X_train[j]
        )

        distances.append(
            (distance, kelas[j])
        )

    distances.sort(key=lambda x: x[0])

    neighbor_data = []

    for urutan, neighbor in enumerate(
        distances[:k],
        start=1
    ):

        neighbor_data.append({
            "Urutan": urutan,
            "Species": neighbor[1],
            "Jarak": round(neighbor[0], 6)
        })

    with st.expander(
        f"ID Testing: {id_testing[i]}"
    ):

        st.dataframe(
            pd.DataFrame(neighbor_data),
            use_container_width=True,
            hide_index=True
        )


# ==========================================
# 12. FORM PREDIKSI DATA BARU
# ==========================================

st.divider()

st.subheader("Prediksi Jenis Bunga Iris")

st.write(
    "Masukkan empat nilai atribut bunga "
    "untuk mengetahui hasil klasifikasinya."
)

with st.form("form_prediksi"):

    col1, col2 = st.columns(2)

    with col1:

        sepal_length = st.number_input(
            "Sepal Length (cm)",
            min_value=0.1,
            value=5.1,
            step=0.1
        )

        sepal_width = st.number_input(
            "Sepal Width (cm)",
            min_value=0.1,
            value=3.5,
            step=0.1
        )

    with col2:

        petal_length = st.number_input(
            "Petal Length (cm)",
            min_value=0.1,
            value=1.4,
            step=0.1
        )

        petal_width = st.number_input(
            "Petal Width (cm)",
            min_value=0.1,
            value=0.2,
            step=0.1
        )

    submit = st.form_submit_button(
        "Prediksi Sekarang",
        use_container_width=True
    )


# ==========================================
# 13. PROSES PREDIKSI DATA BARU
# ==========================================

if submit:

    x_test = np.array([
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ])

    prediction = k_nearest_neighbors(
        X_train,
        kelas,
        x_test,
        k
    )

    st.success(
        f"Hasil Prediksi: {prediction}"
    )

    # Menghitung jarak seluruh data training
    distances = []

    for i in range(len(X_train)):

        distance = euclidean_distance(
            x_test,
            X_train[i]
        )

        distances.append(
            (distance, kelas[i])
        )

    # Mengurutkan jarak
    distances.sort(key=lambda x: x[0])

    # Mengambil 3 tetangga terdekat
    neighbors = distances[:k]

    neighbor_df = pd.DataFrame(
        neighbors,
        columns=["Jarak", "Species"]
    )

    neighbor_df.index = range(1, len(neighbor_df) + 1)
    neighbor_df.index.name = "Urutan"

    st.write("### 3 Tetangga Terdekat")

    st.dataframe(
        neighbor_df,
        use_container_width=True
    )

    st.info(
        "Hasil prediksi ditentukan berdasarkan "
        "kelas dengan jumlah suara terbanyak "
        "dari 3 tetangga terdekat."
    )


# ==========================================
# 14. INFORMASI PENUTUP
# ==========================================

st.divider()

st.caption(
    "Aplikasi Klasifikasi Bunga Iris | "
    "Metode k-Nearest Neighbors (k-NN) | "
    "Euclidean Distance | K = 3"
)
