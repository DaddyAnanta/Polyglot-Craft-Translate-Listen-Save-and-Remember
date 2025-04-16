# 📊 Aplikasi Terjemahan Teks & Pembelajaran Kosakata Berbasis AI

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.44.0-brightgreen)](https://streamlit.io/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8.1-yellow)](https://www.nltk.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-lightgrey)](https://pandas.pydata.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.19.0-orange)](https://plotly.com/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE) 
[![Last Commit](https://img.shields.io/github/last-commit/[YourUsername]/[YourRepoName])](https://github.com/[YourUsername]/[YourRepoName]/commits/main) 
Aplikasi ini memanfaatkan kekuatan AI untuk menerjemahkan teks dan membantu Anda memperkaya kosakata melalui fitur review, visualisasi dengan gambar, dan pelacakan statistik pembelajaran. Dibangun dengan Streamlit untuk antarmuka yang interaktif dan mudah digunakan.

## ✨ Fitur Utama

* **Terjemahan Teks Cerdas:** Terjemahkan teks antar bahasa menggunakan model AI (misalnya, melalui Deepseek R3 via OpenRouter).
* **Riwayat & Tinjauan:** Simpan hasil terjemahan Anda untuk ditinjau kembali kapan saja, memperkuat proses mengingat kosakata.
* **Edit & Hapus:** Kelola riwayat terjemahan Anda dengan mudah, hapus entri yang salah atau tidak relevan.
* **Visualisasi AI:** Hasilkan gambar secara otomatis berdasarkan teks terjemahan menggunakan model AI seperti `FLUX.1 [schnell]` untuk memberikan konteks visual pada kosakata baru.
* **Upload Gambar Manual:** Kaitkan gambar Anda sendiri dengan terjemahan tertentu untuk personalisasi pengalaman belajar.
* **Statistik Pembelajaran:** Pantau kemajuan belajar Anda melalui visualisasi data yang informatif.

## 🚀 Demo & Tampilan Aplikasi

Berikut adalah beberapa cuplikan tampilan aplikasi:

**1. Proses Terjemahan & Hasil**
Masukkan teks, pilih bahasa, dan dapatkan hasil terjemahan. Anda dapat menghasilkan ulang jika perlu.
![Proses Terjemahan](https://github.com/user-attachments/assets/1a77b791-ac5b-4574-8c89-3b481db0bb6c)
![Hasil Terjemahan](https://github.com/user-attachments/assets/b3cedee5-9915-4d98-8fc0-d89ac185d41c)

**2. Tinjau & Kelola Riwayat**
Lihat kembali terjemahan sebelumnya dan hapus jika ada kesalahan.
![Tinjau Riwayat](https://github.com/user-attachments/assets/b37385eb-968d-4db0-91b3-33f335d1785b)
![Kelola Riwayat](https://github.com/user-attachments/assets/d911be0b-745a-4854-a44b-1dbe774ee8db)

**3. Hasilkan Gambar dengan AI (FLUX.1 [schnell])**
Buat visualisasi untuk teks terjemahan Anda secara otomatis.
![Input Gambar AI](https://github.com/user-attachments/assets/0e61c307-2a42-4587-b371-a85a986fd1a3)
![Hasil Gambar AI](https://github.com/user-attachments/assets/808277da-aa71-4b7f-92da-27d3253fc320)

**4. Upload Gambar Manual**
Tambahkan gambar pribadi Anda ke entri terjemahan.
![Upload Gambar](https://github.com/user-attachments/assets/1008e52a-13ab-4474-b5d9-969f035341ed)

**5. Statistik Pembelajaran**
Visualisasikan kemajuan dan data pembelajaran Anda.
![Statistik 1](https://github.com/user-attachments/assets/0ac486be-b822-439e-b512-1459941be00d)
![Statistik 2](https://github.com/user-attachments/assets/0c63594c-aa33-4ecf-a093-83bbbda519b3)

## 🛠️ Teknologi yang Digunakan

* **Bahasa:** Python 3.10+
* **Framework Web:** Streamlit
* **Analisis Teks:** NLTK (opsional, tergantung penggunaan spesifik)
* **Manipulasi Data:** Pandas
* **Visualisasi Data:** Plotly
* **Model AI (Contoh):** DeepSeek R3 (via OpenRouter API), FLUX.1 (via API)

## 💻 Cara Menjalankan di Komputer Lokal

Ikuti langkah-langkah berikut untuk menjalankan aplikasi ini di mesin Anda:

1.  **Prasyarat:**
    * Pastikan Anda telah menginstal Python (versi 3.10 atau lebih baru).
    * (Opsional tapi disarankan) Gunakan *virtual environment* untuk mengisolasi dependensi:
        ```bash
        python -m venv venv
        # Aktivasi di Linux/macOS
        source venv/bin/activate
        # Aktivasi di Windows (Command Prompt/PowerShell)
        venv\Scripts\activate
        ```

2.  **Clone Repositori:**
    ```bash
    git clone https://github.com/DaddyAnanta/Polyglot-Craft-Translate-Listen-Save-and-Remember.git
    cd Polyglot-Craft-Translate-Listen-Save-and-Remember
    ```

3.  **Install Dependensi:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Konfigurasi API Key:**
    * Aplikasi ini memerlukan API Key dari OpenRouter untuk mengakses model AI seperti DeepSeek R3. Dapatkan API Key Anda di [https://openrouter.ai/](https://openrouter.ai/).
    * Buat folder bernama `.streamlit` di direktori root proyek Anda (jika belum ada).
    * Di dalam folder `.streamlit`, buat file bernama `secrets.toml`.
    * Tambahkan API Key Anda ke dalam file `secrets.toml` seperti berikut:
        ```toml
        # .streamlit/secrets.toml

        OPENROUTER_API_KEY = "sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" 
        # Ganti dengan API Key OpenRouter Anda yang sebenarnya
        
        # Jika menggunakan API lain (misalnya untuk FLUX), tambahkan di sini
        # OTHER_API_KEY = "xxxxyyyyzzzz" 
        ```
    * **Penting:** Jangan pernah membagikan file `secrets.toml` atau API key Anda secara publik. Pastikan `.streamlit/secrets.toml` ada di file `.gitignore` Anda.

5.  **Jalankan Aplikasi Streamlit:**
    ```bash
    streamlit run app.py 
    ```
    *(Ganti `app.py` dengan nama file utama aplikasi Streamlit Anda jika berbeda)*

6.  Buka browser Anda dan akses alamat lokal yang ditampilkan oleh Streamlit (biasanya `http://localhost:8501`).

## 🚀 Selamat Mencoba!

Jelajahi fitur terjemahan, manfaatkan riwayat untuk belajar, dan lihat bagaimana visualisasi dapat membantu Anda memahami kosakata baru.

## 🤝 Berkontribusi

Kontribusi sangat diterima! Jika Anda ingin berkontribusi:

1.  Fork repositori ini.
2.  Buat branch fitur baru (`git checkout -b fitur/NamaFitur`).
3.  Commit perubahan Anda (`git commit -am 'Menambahkan fitur X'`).
4.  Push ke branch (`git push origin fitur/NamaFitur`).
5.  Buat Pull Request baru.

## 📄 Lisensi

Proyek ini dilisensikan di bawah Lisensi MIT - lihat file [LICENSE](LICENSE) untuk detailnya.
