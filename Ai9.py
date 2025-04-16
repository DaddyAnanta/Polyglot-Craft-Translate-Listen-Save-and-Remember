# --- AWAL BAGIAN IMPORT ---
import streamlit as st
from gtts import gTTS
from tempfile import NamedTemporaryFile

#Image
import shutil
from gradio_client import Client
import time
from datetime import date

import pandas as pd
import numpy as np


import re
import html
import os
# from streamlit_extras.metric_cards import style_metric_cards # Hapus jika tidak dipakai
import google.generativeai as genai
from langchain.prompts import PromptTemplate # Masih dipakai di Halaman 2
import json # Untuk memparsing output AI jika formatnya JSON
import json # Untuk memparsing output AI jika formatnya JSON
import base64 # Untuk encoding/decoding Base64
from io import BytesIO
from PIL import Image
import requests


### START Statistic

from streamlit_extras.metric_cards import style_metric_cards


import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as pc
import matplotlib.pyplot as plt

### Wordcloud   
from wordcloud import WordCloud 
import re # Untuk ekspresi regule   r (opsional, bisa juga pakai string)
import string # Untuk daftar tand   a baca
import nltk # Library utama NLP 
    
# nltk.data.path.append('/home/master/nltk_data') 
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True) 
    
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords    
from nltk.stem import PorterStemmer 
from collections import Counter     
import warnings 
warnings.filterwarnings("ignore")   




### END Statistic





# --- AKHIR BAGIAN IMPORT ---


# --- Konfigurasi Dasar ---
st.set_page_config(layout="wide", page_title="Kamus Bahasa")
DATA_FILE = "kamus_data.csv" # Nama file untuk menyimpan data kamus

# --- Styling (CSS Anda dipertahankan) ---
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: white;
    background-size: cover;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("https://wallpapers-clan.com/wp-content/uploads/2025/02/straw-hat-pirates-flag-night-ocean-desktop-wallpaper-preview.jpg");
    background-position: center;
    background-color: rgba(45, 51, 107, 0.8); /* Pertahankan transparansi */
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0.3);
}
[data-testid="stToolbar"] {
    right: 2rem;
}

/* == Styling Sidebar == */
/* Warna teks umum di sidebar */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] small,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] button {
    color: white;
}
[data-testid="stSidebar"] button {
    border-color: white;
}

/* Styling KHUSUS untuk Selectbox di Sidebar */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div[data-baseweb="select"] > div:first-child {
    background-color: black !important;
    color: white !important;
    border-radius: 0.25rem;
}

/* Panah dropdown selectbox */
[data-testid="stSidebar"] [data-testid="stSelectbox"] svg {
    fill: white !important;
}

/* Styling dropdown menu */
div[data-baseweb="popover"][aria-label="Dropdown menu"] ul[role="listbox"] {
    background-color: black;
    border: 1px solid #555;
}
div[data-baseweb="popover"][aria-label="Dropdown menu"] ul[role="listbox"] li {
    color: white;
}
div[data-baseweb="popover"][aria-label="Dropdown menu"] ul[role="listbox"] li:hover,
div[data-baseweb="popover"][aria-label="Dropdown menu"] ul[role="listbox"] li[aria-selected="true"] {
    background-color: #333;
}


</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Fungsi Helper (Fungsi Gemini & Data Tetap Sama) ---

@st.cache_resource # Cache resource agar model tidak di-load ulang setiap interaksi
def load_gemini_model(api_key):
    """Memuat dan menginisialisasi model Gemini Pro."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-pro") # atau gemini-pro jika 1.5 belum stabil
        # Untuk chat model, start_chat() biasanya dipanggil saat pertama kali send_message
        # Mengembalikan model langsung mungkin lebih fleksibel
        # chat = model.start_chat(history=[])
        # return chat
        st.session_state.gemini_model_instance = model # Simpan instance model ke session state
        st.session_state.gemini_chat_instance = model.start_chat(history=[]) # Simpan instance chat terpisah
        return st.session_state.gemini_chat_instance # Kembalikan chat instance untuk kompatibilitas Halaman 2
    except Exception as e:
        st.error(f"Gagal memuat model Gemini: {e}")
        return None
    
# Deepseek
def call_openrouter_api(api_key, prompt, model_id="deepseek/deepseek-chat"): # Gunakan ID model OpenRouter
    """Memanggil OpenRouter Chat Completions API."""
    api_url = "https://openrouter.ai/api/v1/chat/completions" # URL OpenRouter
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}", # Gunakan Key OpenRouter
        # Header tambahan yang direkomendasikan OpenRouter (opsional tapi bagus)
        "HTTP-Referer": "kamus-bahasa-app", # Ganti dengan nama app Anda
        "X-Title": "Kamus Bahasa Devi", # Ganti dengan judul app Anda
    }
    payload = {
        "model": model_id, # Model ID sesuai format OpenRouter
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        response_json = response.json()

        if response_json.get("choices") and len(response_json["choices"]) > 0:
            message = response_json["choices"][0].get("message")
            if message and message.get("content"):
                return message["content"].strip()
            else:
                st.error(f"Struktur respons OpenRouter tidak sesuai (missing message/content): {response_json}")
                return None
        else:
            # Cek jika ada error spesifik dari OpenRouter
            if response_json.get("error"):
                 st.error(f"Error dari OpenRouter API: {response_json['error']}")
            else:
                 st.error(f"Struktur respons OpenRouter tidak sesuai (missing choices): {response_json}")
            return None

    except requests.exceptions.HTTPError as http_err:
        # Tangani error HTTP secara spesifik, termasuk 402
        if response.status_code == 402:
             st.error(f"Error 402: Payment Required. Saldo OpenRouter Anda tidak mencukupi atau perlu setup billing. Silakan cek akun OpenRouter Anda.")
        else:
             st.error(f"HTTP Error saat memanggil OpenRouter API: {http_err} - Response: {response.text}")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error koneksi saat memanggil OpenRouter API: {e}")
        return None
    except Exception as e:
        st.error(f"Error tidak terduga saat memproses respons OpenRouter: {e}")
        return None


@st.cache_data # Cache data agar file tidak dibaca ulang terus menerus jika tidak berubah
def load_data(filepath):
    """Memuat data kamus dari file CSV, menambahkan kolom jika perlu."""
    # Tambahkan 'date' ke daftar kolom yang diharapkan
    expected_columns = ["language", "word", "translate", "example_sentence",
                       "translate_example_sentence", "speak", "speak_word",
                        "image_base64", "date"] # Tambahkan 'date'

    if os.path.exists(filepath):
        try:
            df = pd.read_csv(filepath)
            # Cek dan tambahkan kolom yang hilang
            for col in expected_columns:
                if col not in df.columns:
                    st.info(f"Menambahkan kolom '{col}' ke DataFrame.")
                    df[col] = pd.NA # Inisialisasi kolom baru dengan NA
            # Pastikan tipe data kolom baru sesuai
            if 'image_base64' in df.columns:
                 df['image_base64'] = df['image_base64'].astype(object)
            if 'date' in df.columns:
                 df['date'] = df['date'].astype(object) # Simpan tanggal sebagai string/object
            return df[expected_columns]
        except pd.errors.EmptyDataError:
             st.warning(f"File data '{filepath}' kosong. Membuat DataFrame baru.")
             return pd.DataFrame(columns=expected_columns)
        except Exception as e:
            st.error(f"Gagal memuat data dari {filepath}: {e}")
            return pd.DataFrame(columns=expected_columns)
    else:
        st.info(f"File data '{filepath}' tidak ditemukan. Membuat DataFrame baru.")
        return pd.DataFrame(columns=expected_columns)

def save_data(df, filepath):
    """Menyimpan DataFrame ke file CSV."""
    try:
        # Pastikan kolom yang diharapkan ada dan tipenya sesuai
        expected_columns = ["language", "word", "translate", "example_sentence",
                           "translate_example_sentence", "speak", "speak_word",
                           "image_filename", "image_base64", "date"]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = pd.NA

        # Pastikan tipe object untuk kolom yang bisa NA atau string
        if 'image_base64' in df.columns: df['image_base64'] = df['image_base64'].astype(object)
        if 'date' in df.columns: df['date'] = df['date'].astype(object)
        if 'image_filename' in df.columns: df['image_filename'] = df['image_filename'].astype(object)

        # Simpan ke CSV, representasikan nilai NA (termasuk pd.NA) sebagai string kosong
        df.to_csv(filepath, index=False, na_rep='')
        st.cache_data.clear()
        # st.write("DEBUG: Data berhasil disimpan ke CSV.") # Debug opsional
    except Exception as e:
        st.error(f"Gagal menyimpan data ke {filepath}: {e}")
        st.exception(e) # Tampilkan traceback saat save gagal
def parse_ai_response(response_text):
    """Mencoba memparsing respons AI (diasumsikan JSON)."""
    try:
        # Hapus ```json dan ``` jika ada
        cleaned_text = response_text.strip().removeprefix("```json").removesuffix("```").strip()
        data = json.loads(cleaned_text)
        # Validasi dasar: pastikan kunci yang diharapkan ada
        # Sesuaikan dengan kunci yang ada di PromptTemplate Halaman 2
        required_keys = ["translate", "translate_example_sentence", "speak", "speak_word"]
        if all(key in data for key in required_keys):
            # Bersihkan nilai string dari spasi ekstra
            for key in required_keys:
                 if isinstance(data[key], str):
                     data[key] = data[key].strip()
            return data
        else:
            st.error(f"Output JSON dari AI tidak lengkap. Kunci yang hilang: {[k for k in required_keys if k not in data]}")
            return None
    except json.JSONDecodeError as e:
        st.error(f"Gagal memparsing respons AI sebagai JSON: {e}\nRespons Mentah:\n{response_text}")
        return None
    except Exception as e:
        st.error(f"Error saat memproses respons AI: {e}")
        return None

# --- Memuat API Key Gemini (Hanya untuk Halaman 1 & 2) ---
# Ini tetap diperlukan karena Halaman 2 menggunakan Gemini
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    if not gemini_api_key or gemini_api_key == "MASUKKAN_API_KEY_ANDA_DI_SINI":
        st.error("API Key Gemini belum diatur di secrets.toml. Silakan atur terlebih dahulu.")
        chat_model = None # Set model ke None jika key tidak valid
        st.stop() # Hentikan jika key tidak ada
    else:
        # Muat model Gemini dan simpan chat instance ke session state
        chat_model = load_gemini_model(gemini_api_key)
        if chat_model is None:
             st.stop() # Hentikan jika model gagal load

except FileNotFoundError:
    st.error("File .streamlit/secrets.toml tidak ditemukan. Buat file tersebut dan masukkan API Key Gemini.")
    chat_model = None
    st.stop()
except KeyError:
    st.error("Key 'GEMINI_API_KEY' tidak ditemukan dalam file secrets.toml.")
    chat_model = None
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat mengakses API Key Gemini: {e}")
    chat_model = None
    st.stop()


# --- Memuat Data Kamus ---
df = load_data(DATA_FILE)

# --- Pilihan Halaman (Sidebar) ---
st.sidebar.title("Menu Navigasi")
page_options = ["Lihat Kamus", "Tambah/Hapus Kata", "Generate Gambar (AI)", "Upload Gambar Manual", "Statistic"]
selected_page = st.sidebar.radio("Pilih Halaman:", page_options)

# ===========================================
# --- Halaman 1: Lihat Kamus (Tidak Diubah)---
# ===========================================
if selected_page == "Lihat Kamus":
    st.markdown(f"<h1 style='text-align: center; color: black;'>Kamus Bahasa🐉️</h1>", unsafe_allow_html=True)

    if df.empty:
        st.warning("Kamus masih kosong. Silakan tambahkan kata baru melalui menu 'Tambah/Hapus Kata'.")
    else:
        # --- Sidebar Filter (Hanya jika ada data) ---
        st.sidebar.title("Filter Bahasa")
        select_all_option = "Tampilkan Semua"
        # Ambil bahasa unik dari DataFrame yang sudah dimuat
        available_languages = df["language"].unique()
        language_options = np.append(available_languages, select_all_option).tolist()

        default_index = 0 # Default ke opsi pertama jika 'English' tidak ada atau df kosong
        if 'English' in language_options:
            try:
                default_index = language_options.index('English')
            except ValueError:
                pass # 'English' tidak ada, gunakan default 0

        division_name = st.sidebar.selectbox(
            "Bahasa",
            options=language_options,
            index=default_index,
            key="view_language_select" # Key unik untuk view
        )

        # Filter berdasarkan bahasa
        filtered_df = df.copy()
        if division_name != select_all_option:
            filtered_df = filtered_df[filtered_df["language"] == division_name]

        if filtered_df.empty and division_name != select_all_option:
            st.warning(f"Tidak ada kata untuk bahasa '{division_name}'.")
        elif filtered_df.empty and division_name == select_all_option:
             st.warning("Kamus masih kosong.") # Pesan jika kamus benar-benar kosong
        elif filtered_df.empty:
            st.warning(f"Tidak ada data untuk bahasa '{division_name}'.") # Pesan jika filter tidak menghasilkan apa pun
        else:
            # Dapatkan daftar kata unik yang sudah difilter dan diurutkan
            word_list = sorted(filtered_df['word'].unique())
            selectvalue = st.selectbox(
                "Pilih Kata (atau Tampilkan Semua hasil filter secara berurutan)",
                options=word_list + [select_all_option],
                key="view_word_select" # Key unik untuk view
            )

            # Tentukan DataFrame yang akan digunakan untuk navigasi
            if selectvalue != select_all_option:
                final_df = filtered_df[filtered_df['word'] == selectvalue].reset_index(drop=True)
            else:
                final_df = filtered_df.sort_values(by=['language', 'word']).reset_index(drop=True) # Urutkan jika tampil semua


            # --- State Management untuk Navigasi ---
            # Gunakan key unik untuk state navigasi halaman ini
            nav_state_key_prefix = f"view_nav_{division_name}_{selectvalue}_"
            current_index_key = nav_state_key_prefix + 'current_index'
            total_items_key = nav_state_key_prefix + 'total_items'

            if current_index_key not in st.session_state:
                st.session_state[current_index_key] = 0
            if total_items_key not in st.session_state:
                st.session_state[total_items_key] = 0

            current_total_items = len(final_df)
            # Reset index jika jumlah item berubah atau index tidak valid
            if st.session_state.get(total_items_key, 0) != current_total_items or st.session_state.get(current_index_key, 0) >= current_total_items:
                st.session_state[current_index_key] = 0
            st.session_state[total_items_key] = current_total_items


            # --- Tampilkan satu item berdasarkan current_index ---
            if not final_df.empty and st.session_state[total_items_key] > 0 :
                try:
                    current_idx = st.session_state[current_index_key]
                    row = final_df.iloc[current_idx]

                    # --- Kolom Kiri (Info Kata & Gambar) ---
                    col_left, col_right = st.columns([2,3]) # Lebarkan kolom kiri sedikit

                    with col_left:
                        st.markdown(f"<div style='font-size: 30px; font-weight: bold; text-align: center; color:black;'>{html.escape(str(row.get('word', '')))}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 16px; font-weight: bold; text-align: left; color:black; margin-top:10px; margin-bottom:10px;'>Terjemahan:<br>{html.escape(str(row.get('translate', '')))} ( {html.escape(str(row.get('speak_word', '')))} ) </div>", unsafe_allow_html=True)

                         # Tampilkan Gambar jika ada
                        image_b64 = row.get('image_base64')
                        if isinstance(image_b64, str) and image_b64: # Cek apakah string dan tidak kosong
                            try:
                                img_bytes_display = base64.b64decode(image_b64)
                                st.image(img_bytes_display, caption=f"Gambar untuk '{row.get('word', '')}'", use_column_width=True)
                            except Exception as img_e:
                                st.warning(f"Gagal menampilkan gambar tersimpan: {img_e}")
                        else:
                            st.caption("_(Tidak ada gambar tersimpan untuk kata ini)_") # Placeholder jika tidak ada gambar


                    # --- Kolom Kanan (Contoh Kalimat & Navigasi) ---
                    with col_right:
                        # Ambil data kalimat dengan fallback string kosong jika NaN/None
                        original_sentence = str(row.get('example_sentence', ''))
                        original_translated_sentence = str(row.get('translate_example_sentence', ''))
                        word_to_highlight_original = str(row.get('word', ''))
                        word_to_highlight_translated = str(row.get('translate', ''))

                        # --- Highlighting (Kode Anda sudah bagus) ---
                        highlighted_sentence = html.escape(original_sentence) # Default aman
                        if word_to_highlight_original and original_sentence:
                            try:
                                safe_word_html_orig = html.escape(word_to_highlight_original)
                                replacement_html_orig = f"<b><u>{safe_word_html_orig}</u></b>"
                                safe_word_regex_orig = re.escape(word_to_highlight_original)
                                pattern_orig = r'\b' + safe_word_regex_orig + r'\b'
                                highlighted_sentence = re.sub(pattern_orig, replacement_html_orig, original_sentence, flags=re.IGNORECASE)
                            except re.error as e:
                                st.warning(f"Regex error (original): {e}")

                        highlighted_translated_sentence = html.escape(original_translated_sentence) # Default aman
                        if word_to_highlight_translated and original_translated_sentence:
                            try:
                                safe_word_html_trans = html.escape(word_to_highlight_translated)
                                replacement_html_trans = f"<b><u>{safe_word_html_trans}</u></b>"
                                safe_word_regex_trans = re.escape(word_to_highlight_translated)
                                pattern_trans = r'\b' + safe_word_regex_trans + r'\b'
                                highlighted_translated_sentence = re.sub(pattern_trans, replacement_html_trans, original_translated_sentence, flags=re.IGNORECASE)
                            except re.error as e:
                                st.warning(f"Regex error (translated): {e}")

                        # Tampilkan Kalimat
                        st.markdown("**Contoh Kalimat:**")
                        st.markdown(f"<div style='font-size: 16px;text-align: left; background-color:rgba(0, 0, 255, 0.6); color:white; padding: 10px; border-radius: 5px; min-height: 60px;'>{highlighted_sentence}</div>", unsafe_allow_html=True)
                        speak_text = str(row.get('speak', ''))
                        if speak_text:
                            st.markdown(f"<div style='font-size: 14px;text-align: left; color:grey; margin-top: 5px; margin-bottom: 15px;'><i>Pengucapan: {html.escape(speak_text)}</i></div>", unsafe_allow_html=True)
                        else:
                             st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True) # Beri jarak bawah


                        st.markdown("**Terjemahan Kalimat:**")
                        st.markdown(f"<div style='font-size: 16px; text-align: left;color:black; background-color:rgba(255, 255, 0, 0.7); padding: 10px; border-radius: 5px; min-height: 60px;margin-bottom:30px;'>{highlighted_translated_sentence}</div>", unsafe_allow_html=True)

                        

                        # --- Tombol Navigasi & Suara ---
                        col_nav_prev, col_nav_info, col_nav_sound, col_nav_next = st.columns([1, 2, 3, 1])

                        with col_nav_prev:
                            disable_prev = current_idx == 0
                            if st.button("⬅️ Prev", disabled=disable_prev, use_container_width=True, key=f"prev_btn_{nav_state_key_prefix}"):
                                st.session_state[current_index_key] -= 1
                                st.rerun()

                        with col_nav_info:
                            st.markdown(f"<div style='text-align: center; margin-top: 10px;'>Item {current_idx + 1} dari {st.session_state[total_items_key]}</div>", unsafe_allow_html=True)

                        with col_nav_sound:
                            if original_sentence:
                                sound_button_key = f"play_sound_{nav_state_key_prefix}{current_idx}"
                                if st.button(f"🔊 Putar Suara: {html.escape(word_to_highlight_original)}", key=sound_button_key, use_container_width=True):
                                    lang_code = 'en' if row.get('language', '').lower() == 'english' else 'ja' if row.get('language', '').lower() == 'japanese' else 'id'
                                    try:
                                        tts = gTTS(text=original_sentence, lang=lang_code, slow=False)
                                        with NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
                                            tts.save(tmpfile.name)
                                            st.audio(tmpfile.name, format="audio/mp3")
                                        # os.remove(tmpfile.name) # Pertimbangkan jika file tmp menumpuk
                                    except Exception as e:
                                        st.error(f"Gagal memutar suara: {e}")

                        with col_nav_next:
                            disable_next = current_idx >= st.session_state[total_items_key] - 1
                            if st.button("Next ➡️", disabled=disable_next, use_container_width=True, key=f"next_btn_{nav_state_key_prefix}"):
                                st.session_state[current_index_key] += 1
                                st.rerun()

                except IndexError:
                    st.warning("Indeks data tidak valid. Mereset...")
                    st.session_state[current_index_key] = 0
                    st.rerun()
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat menampilkan data: {e}")
                    import traceback
                    st.error(traceback.format_exc()) # Tampilkan traceback untuk debug
            elif st.session_state[total_items_key] == 0 and (division_name != select_all_option or selectvalue != select_all_option):
                st.info("Filter yang dipilih tidak menghasilkan data.")

# ===========================================
# --- Halaman 2: Tambah/Hapus Kata ---
# ===========================================
elif selected_page == "Tambah/Hapus Kata":
    st.markdown(f"<h1 style='text-align: center; color: black;'>Tambah Kata Baru (DeepSeek AI) 🧑‍🎤</h1>", unsafe_allow_html=True) # Judul diubah

    # --- Inisialisasi session state untuk preview ---
    if 'ai_generated_data' not in st.session_state:
        st.session_state.ai_generated_data = None
    if 'original_input_data' not in st.session_state:
        st.session_state.original_input_data = None

    # --- Ambil API Key DeepSeek ---
    try:
        # Ganti ke OPENROUTER_API_KEY
        openrouter_api_key = st.secrets["OPENROUTER_API_KEY"]
        if not openrouter_api_key or "KEY_OPENROUTER_ANDA" in openrouter_api_key:
             st.error("API Key OpenRouter belum diatur dengan benar di secrets.toml.")
             st.stop()
    except FileNotFoundError:
        st.error("File .streamlit/secrets.toml tidak ditemukan.")
        st.stop()
    except KeyError:
        st.error("Key 'OPENROUTER_API_KEY' tidak ditemukan dalam file secrets.toml.")
        st.stop()
    except Exception as e:
        st.error(f"Gagal membaca API Key OpenRouter: {e}")
        st.stop()

    # --- Form Tambah Kata ---
    with st.form("add_word_form"):
        st.subheader("Masukkan Detail Kata Baru")
        selected_language = st.selectbox(
            "Pilih Bahasa",
            ("English", "Japanese"),
            key="add_language"
        )
        new_word = st.text_input("Kata Baru", key="add_word")
        new_example_sentence = st.text_area("Contoh Kalimat", key="add_example", height=100)

        submit_button = st.form_submit_button("✨ Generate Terjemahan & Pengucapan (DeepSeek AI)")

        if submit_button:
            st.session_state.ai_generated_data = None
            st.session_state.original_input_data = None

            if not new_word or not new_example_sentence:
                st.warning("Harap isi semua field (Kata Baru dan Contoh Kalimat).")
            else:
                # --- Memanggil AI DEEPSEEK ---
                with st.spinner("Meminta bantuan DeepSeek AI untuk menerjemahkan..."):
                    # Prompt template tetap sama, meminta JSON
                    prompt_template = PromptTemplate(
                            template="""Anda adalah asisten penerjemah multibahasa.
                            Tugas Anda adalah menerima input berupa bahasa target, sebuah kata/frasa, dan contoh kalimat dalam bahasa tersebut.
                            Anda HARUS menghasilkan output HANYA berupa JSON yang valid tanpa penjelasan atau teks tambahan di luar JSON.
                            JSON harus berisi kunci berikut:
                            1. "translate": Terjemahan kata/frasa '{word}' ke Bahasa Indonesia. Sesuaikan terjemahan kata agar konsisten dengan konteks kalimat.
                            2. "translate_example_sentence": Terjemahan lengkap kalimat '{sentence}' ke Bahasa Indonesia, serta sesuaikan terjemahan dari 'word'.
                            3. "speak": Cara pengucapan kata/frasa '{sentence}' dalam bahasa {language} menggunakan Simple Phonetic Respelling yang intuitif (contoh: 'cat' menjadi 'kat', 'phone' menjadi 'fohn', 'roar' menjadi 'ror', 'through' menjadi 'throo'). Jangan gunakan International Phonetic Alphabet.
                            4. "speak_word": Cara pengucapan kata/frasa '{word}' dalam bahasa {language} menggunakan Simple Phonetic Respelling yang intuitif (contoh: 'cat' menjadi 'kat', 'phone' menjadi 'fohn', 'roar' menjadi 'ror', 'through' menjadi 'throo'). Jangan gunakan International Phonetic Alphabet.

                            Input:
                            Bahasa: {language}
                            Kata/Frasa: {word}
                            Contoh Kalimat: {sentence}

                            Output JSON:
                            """,
                            input_variables=["language", "word", "sentence"]
                        )

                    prompt = prompt_template.format(
                        language=selected_language,
                        word=new_word,
                        sentence=new_example_sentence
                    )

                    # --- Panggil fungsi API DeepSeek ---
                    response_text = call_openrouter_api(openrouter_api_key, prompt, model_id="deepseek/deepseek-chat") # Pastikan model_id benar

                    if response_text:
                        # Parsing respons JSON (tetap sama)
                        parsed_data = parse_ai_response(response_text)
                        if parsed_data:
                            # --- SIMPAN KE SESSION STATE UNTUK PREVIEW ---
                            # Pastikan ini adalah dictionary literal yang benar
                            st.session_state.original_input_data = {
                                "language": selected_language,
                                "word": new_word.strip(),
                                "example_sentence": new_example_sentence.strip()
                            }
                            # ---------------------------------------------
                            st.session_state.ai_generated_data = parsed_data
                            st.info("Hasil AI berhasil digenerate. Silakan review di bawah.")
    # --- Tampilkan Preview dan Tombol Add (Logika ini tetap sama) ---
    if st.session_state.get('ai_generated_data') and st.session_state.get('original_input_data'):
        # ... (Kode untuk menampilkan markdown preview tidak berubah) ...
        # ... (Kode untuk tombol Add dan logikanya tidak berubah) ...
        st.markdown("---")
        st.subheader("👀 Preview Hasil AI (DeepSeek)") # Update subheader preview

        original_input = st.session_state.original_input_data
        ai_data = st.session_state.ai_generated_data

        today_date = date.today()
        formatted_date = today_date.strftime("%Y-%m-%d") # Format YYYY-MM-DD (ISO standard)


        # Tampilkan data input dan hasil AI
        st.markdown(f"**Bahasa:** {original_input['language']}")
        st.markdown(f"**Kata Asli:** {original_input['word']}")
        st.markdown(f"**Contoh Kalimat Asli:** {original_input['example_sentence']}")
        st.markdown("---")
        st.markdown(f"**Terjemahan Kata (AI):** {ai_data['translate']}")
        st.markdown(f"**Terjemahan Kalimat (AI):** {ai_data['translate_example_sentence']}")
        # Pastikan kunci 'speak_word' dan 'speak' ada di ai_data (sudah divalidasi di parse_ai_response)
        st.markdown(f"**Pengucapan Kata (AI):** `{ai_data.get('speak_word', 'N/A')}`")
        st.markdown(f"**Pengucapan Kalimat (AI):** `{ai_data.get('speak', 'N/A')}`")
        st.markdown(f"**Date:** {formatted_date}")
        st.markdown("---")

        # --- Tombol Add ---
        if st.button("➕ Tambahkan ke Kamus", key="confirm_add_btn"):
            with st.spinner("Menyimpan data ke kamus..."):
                # --- DAPATKAN TANGGAL HARI INI ---
                today_date = date.today()
                formatted_date = today_date.strftime("%Y-%m-%d")
                # --- DEBUG 1: Cek tanggal yang diformat ---
                st.write(f"DEBUG: Tanggal Diformat = {formatted_date} (Tipe: {type(formatted_date)})")

                # Buat data baru dari session state dan tambahkan tanggal
                new_data_row = {
                        "language": original_input["language"],
                        "word": original_input["word"],
                        "translate": ai_data["translate"],
                        "example_sentence": original_input["example_sentence"],
                        "translate_example_sentence": ai_data["translate_example_sentence"],
                        "speak": ai_data["speak"],
                        "speak_word": ai_data["speak_word"],
                        "image_filename": pd.NA,
                        "image_base64": pd.NA,
                        "date": formatted_date  # Tanggal string ditambahkan di sini
                    }
                # --- DEBUG 2: Cek dictionary baris baru ---
                st.write(f"DEBUG: new_data_row = {new_data_row}")

                try:
                    new_df_row = pd.DataFrame([new_data_row])
                    # --- DEBUG 3: Cek DataFrame baris baru & tipe datanya ---
                    st.write("DEBUG: DataFrame Baris Baru (new_df_row):")
                    st.dataframe(new_df_row)
                    st.write(f"DEBUG: Tipe data kolom 'date' di new_df_row: {new_df_row['date'].dtype}")


                    # Reload data terbaru sebelum menambahkan (penting!)
                    current_df = load_data(DATA_FILE)
                    # --- DEBUG 4: Cek DataFrame saat ini (sebelum concat) ---
                    st.write("DEBUG: DataFrame Saat Ini (current_df) - 5 baris terakhir:")
                    st.dataframe(current_df.tail())
                    st.write(f"DEBUG: Tipe data kolom 'date' di current_df: {current_df['date'].dtype if 'date' in current_df.columns else 'Kolom tidak ada'}")


                    # Tambahkan data baru dan simpan
                    updated_df = pd.concat([current_df, new_df_row], ignore_index=True)
                    # --- DEBUG 5: Cek DataFrame setelah concat & tipe datanya ---
                    st.write("DEBUG: DataFrame Setelah Concat (updated_df) - 5 baris terakhir:")
                    st.dataframe(updated_df.tail())
                    st.write(f"DEBUG: Tipe data kolom 'date' di updated_df: {updated_df['date'].dtype}")
                    st.write(f"DEBUG: Nilai kolom 'date' di updated_df (list): {updated_df['date'].to_list()}")


                    # --- DEBUG 6: Cek DataFrame TEPAT SEBELUM disimpan ---
                    st.write("DEBUG: DataFrame TEPAT SEBELUM Disimpan:")
                    st.dataframe(updated_df)


                    save_data(updated_df, DATA_FILE) # Panggil fungsi simpan

                    st.success(f"Kata '{original_input['word']}' (ditambahkan pada {formatted_date}) berhasil ditambahkan ke kamus!")

                    st.session_state.ai_generated_data = None
                    st.session_state.original_input_data = None
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal menyimpan data baru ke kamus: {e}")
                    # Tampilkan traceback jika perlu untuk debug lebih lanjut
                    st.exception(e)

    # --- Bagian Hapus Kata (Tetap sama seperti sebelumnya) ---
    st.markdown("---")
    st.subheader("Hapus Kata dari Kamus")
    # Reload df terbaru untuk bagian hapus
    df_latest = load_data(DATA_FILE)
    if df_latest.empty:
        st.info("Kamus masih kosong, tidak ada kata untuk dihapus.")
    else:
         words_to_delete = df_latest['word'].unique().tolist() # Ambil unique
         # Pastikan list tidak kosong sebelum membuat selectbox
         if words_to_delete:
             selected_word_to_delete = st.selectbox("Pilih kata yang ingin dihapus:", [""] + sorted(words_to_delete), key="delete_word_select")
             if selected_word_to_delete:
                 # Tampilkan detail kata yang akan dihapus untuk konfirmasi (opsional)
                 st.caption(f"Detail kata '{selected_word_to_delete}':")
                 st.dataframe(df_latest[df_latest['word'] == selected_word_to_delete], use_container_width=True)
                 if st.button(f"🗑️ Konfirmasi Hapus Kata '{selected_word_to_delete}'", key="delete_btn", type="primary"):
                     indices_to_drop = df_latest[df_latest['word'] == selected_word_to_delete].index
                     if not indices_to_drop.empty:
                         df_after_delete = df_latest.drop(indices_to_drop).reset_index(drop=True)
                         save_data(df_after_delete, DATA_FILE)
                         st.success(f"Kata '{selected_word_to_delete}' berhasil dihapus.")
                         # Reset pilihan delete box dan rerun
                         st.session_state.delete_word_select = "" # Kosongkan pilihan selectbox
                         # Hapus state preview jika ada
                         st.session_state.ai_generated_data = None
                         st.session_state.original_input_data = None
                         st.rerun()
                     else:
                         st.warning(f"Kata '{selected_word_to_delete}' tidak ditemukan saat mencoba menghapus.")
         else:
              st.info("Tidak ada kata unik yang bisa dihapus.")

# ===========================================================
# --- Halaman 3: Generate Gambar (AI) - MENGGUNAKAN HUGGING FACE ---
# ===========================================================
elif selected_page == "Generate Gambar (AI)":
    st.markdown(f"<h1 style='text-align: center; color: black;'>Generate Gambar (FLUX.1 [schnell]) 🖼️</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: black; font-style: italic;'>Fitur ini menggunakan Hugging Face (gradio_client).</p>", unsafe_allow_html=True)

    # --- Inisialisasi session state untuk preview gambar HF ---
    if 'hf_generated_image_path' not in st.session_state:
        st.session_state.hf_generated_image_path = None # Menyimpan PATH gambar hasil generate
    if 'hf_image_prompt_info' not in st.session_state:
        st.session_state.hf_image_prompt_info = None # Menyimpan word & prompt

    # Muat data terbaru
    df_auto = load_data(DATA_FILE)
    df_auto = df_auto[df_auto["image_base64"].isna()]

    if df_auto.empty:
        st.warning("Kamus masih kosong. Tambahkan kata terlebih dahulu.")
    else:
        st.subheader("Pilih Kata untuk Generate Gambar")
        # Hanya perlu selectbox kata
        word_list_auto = sorted(df_auto['word'].unique().tolist())
        selected_word_auto = st.selectbox(
            "Pilih kata dari kamus:",
            options=[""] + word_list_auto, # Opsi kosong di awal
            key="auto_img_word_select"
        )

        if selected_word_auto:
            # Cari data baris pertama yang cocok
            # Gunakan .loc untuk menghindari SettingWithCopyWarning potensial
            selected_row_data_auto_list = df_auto.loc[df_auto['word'] == selected_word_auto]
            if not selected_row_data_auto_list.empty:
                 selected_row_data_auto = selected_row_data_auto_list.iloc[0]
                 prompt_sentence_auto = str(selected_row_data_auto.get('example_sentence', ''))

                 if not prompt_sentence_auto:
                     st.warning(f"Kata '{selected_word_auto}' tidak memiliki contoh kalimat valid untuk dijadikan prompt.")
                 else:
                     st.markdown("**Contoh Kalimat (Prompt Dasar):**")
                     st.info(prompt_sentence_auto)

                     # Tombol Generate
                     if st.button(f"🎨 Generate Gambar untuk '{selected_word_auto}'", key="generate_hf_img_btn"):
                         # Reset state lama sebelum generate baru
                         st.session_state.hf_generated_image_path = None
                         st.session_state.hf_image_prompt_info = None

                         with st.spinner(f"Menghubungi Hugging Face & Menggambar... (Model: FLUX.1-schnell)"):
                             try:
                                 # --- Persiapan Prompt (Gunakan PromptTemplate) ---
                                 image_prompt_template_hf = PromptTemplate(
                                     template="""
                                     Task: Generate an image based on the provided sentence.
                                     Output Requirement: ONLY the image file.
                                     Image Style Guidance: High quality, detailed, clear composition, photorealistic style, cinematic lighting.
                                     Sentence for Image Generation:
                                     "{sentence}"
                                     """,
                                     input_variables=["sentence"]
                                 )
                                 hf_image_prompt = image_prompt_template_hf.format(sentence=prompt_sentence_auto)

                                 # --- Panggil HF Client ---
                                 client = Client("black-forest-labs/FLUX.1-schnell")
                                 progress_bar = st.progress(0) # Simulasi progress bar
                                 for percent_complete in range(0, 100, 20):
                                     time.sleep(0.5)
                                     progress_bar.progress(percent_complete)

                                 result = client.predict(
                                     prompt=hf_image_prompt,
                                     seed=0,
                                     randomize_seed=True,
                                     width=1024,
                                     height=1024,
                                     num_inference_steps=4,
                                     api_name="/infer"
                                 )
                                 progress_bar.progress(100)

                                 # --- Proses Hasil ---
                                 if isinstance(result, (list, tuple)) and len(result) > 0 and isinstance(result[0], str):
                                     generated_image_path = result[0]
                                     if os.path.exists(generated_image_path):
                                         # --- SIMPAN PATH KE SESSION STATE UNTUK PREVIEW ---
                                         st.session_state.hf_generated_image_path = generated_image_path
                                         st.session_state.hf_image_prompt_info = {
                                             "word": selected_word_auto,
                                             "prompt_sentence": prompt_sentence_auto
                                         }
                                         st.info("Gambar berhasil digenerate. Silakan review di bawah.")
                                     else:
                                         st.error(f"AI menghasilkan path '{generated_image_path}', tapi file tidak ditemukan.")
                                 else:
                                     st.error("Format hasil dari Hugging Face tidak sesuai ekspektasi.")
                                     st.write("Hasil mentah:", result)

                             except Exception as e:
                                 st.error(f"Gagal generate gambar via Hugging Face: {e}")
                                 st.warning("Pastikan library 'gradio_client' terinstal dan koneksi internet stabil.")
            else:
                 st.error(f"Data untuk kata '{selected_word_auto}' tidak ditemukan.")


        # --- Tampilkan Preview Gambar dan Tombol Add (DI LUAR Tombol Generate) ---
        # Cek apakah ada path gambar di session state
        if st.session_state.get('hf_generated_image_path') and st.session_state.get('hf_image_prompt_info'):
            st.markdown("---")
            st.subheader("🖼️ Preview Gambar Hasil AI (Hugging Face)")

            img_info_hf = st.session_state.hf_image_prompt_info
            img_path_hf = st.session_state.hf_generated_image_path

            st.markdown(f"**Untuk Kata:** {img_info_hf['word']}")
            st.markdown(f"**Dari Prompt:** {img_info_hf['prompt_sentence']}")

            # Tampilkan gambar dari path
            if os.path.exists(img_path_hf):
                st.image(img_path_hf, caption="Preview Gambar", use_column_width=True)
                st.markdown("---")

                # --- Tombol Add Gambar Base64 ---
                if st.button("💾 Tambah Gambar Base64 ke Kamus", key="confirm_add_hf_img_btn"):
                    try:
                        # 1. Baca image bytes dari file path
                        with open(img_path_hf, "rb") as f:
                            image_bytes_hf = f.read()

                        # 2. Encode ke Base64
                        base64_encoded_string_hf = base64.b64encode(image_bytes_hf).decode('utf-8')

                        # 3. Update DataFrame (Reload dulu data terbaru)
                        current_df_hf = load_data(DATA_FILE)
                        word_to_update_hf = img_info_hf['word']

                        target_indices_hf = current_df_hf[current_df_hf['word'] == word_to_update_hf].index
                        if not target_indices_hf.empty:
                            target_index_hf = target_indices_hf[0] # Update kejadian pertama
                            # Update kolom image_base64
                            current_df_hf.loc[target_index_hf, 'image_base64'] = base64_encoded_string_hf
                            # Hapus filename lama jika ada (karena Base64 lebih prioritas)

                            # 4. Simpan DataFrame
                            save_data(current_df_hf, DATA_FILE)
                            st.success(f"Gambar Base64 dari AI berhasil disimpan untuk kata '{word_to_update_hf}'!")

                            # 5. Kosongkan state preview
                            st.session_state.hf_generated_image_path = None
                            st.session_state.hf_image_prompt_info = None
                            # Tidak perlu hapus file manual, biarkan OS/Gradio yang urus /tmp
                            st.rerun()
                        else:
                             st.error(f"Kata '{word_to_update_hf}' tidak ditemukan di kamus saat mencoba menyimpan gambar.")

                    except FileNotFoundError:
                        st.error(f"File gambar '{img_path_hf}' tidak ditemukan saat mencoba menambahkannya. Silakan generate ulang.")
                        # Reset state jika file hilang
                        st.session_state.hf_generated_image_path = None
                        st.session_state.hf_image_prompt_info = None
                        st.rerun()
                    except Exception as e_add:
                        st.error(f"Gagal membaca atau menyimpan gambar Base64: {e_add}")
            else:
                # Jika file path di state tidak valid/hilang
                st.error("File gambar untuk preview tidak ditemukan. Silakan coba generate ulang.")
                # Reset state
                st.session_state.hf_generated_image_path = None
                st.session_state.hf_image_prompt_info = None
# ===========================================
# --- Halaman 4: Upload Gambar Manual ---
# ===========================================
elif selected_page == "Upload Gambar Manual":
    st.markdown(f"<h1 style='text-align: center; color: black;'>Upload Gambar Manual  manualmente </h1>", unsafe_allow_html=True)

    df_manual = load_data(DATA_FILE) # Muat data terbaru
    df_manual = df_manual[df_manual["image_base64"].isna()]

    if df_manual.empty:
        st.warning("Kamus masih kosong. Tambahkan kata terlebih dahulu sebelum mengupload gambar.")
    else:
        st.subheader("Pilih Kata dan Upload Gambar")
        select_all_option = "Tampilkan Semua"
        # Ambil bahasa unik dari DataFrame yang sudah dimuat
        available_languages = df_manual["language"].unique()
        language_options = np.append(available_languages, select_all_option).tolist()
        default_index = 0 # Default ke opsi pertama jika 'English' tidak ada atau df kosong
        if 'English' in language_options:
            try:
                default_index = language_options.index('English')
            except ValueError:
                pass # 'English' tidak ada, gunakan default 0

        division_name = st.sidebar.selectbox(
            "Bahasa",
            options=language_options,
            index=default_index,
            key="view_language_select" # Key unik untuk view
        )

        # Filter berdasarkan bahasa
        filtered_df = df_manual.copy()
        if division_name != select_all_option:
            filtered_df = filtered_df[filtered_df["language"] == division_name]
        
        word_list_manual = sorted(filtered_df['word'].unique().tolist())
        selected_sentence_manual = ""
        selected_word_manual = st.selectbox(
            "Pilih kata dari kamus:",
            options=[""] + word_list_manual, # Opsi kosong di awal
            key="manual_img_word_select"
        )
        selected_sentence_manual = df_manual[df_manual["word"] == selected_word_manual]
        selected_sentence = (
    selected_sentence_manual[selected_sentence_manual.columns[3]].values[0]
    if len(selected_sentence_manual) > 0 else ""
)

        if selected_word_manual:
            st.info(f"Anda akan menambahkan gambar untuk kata: *{selected_word_manual}* dengan kalimat **{selected_sentence}**")
            

            uploaded_file = st.file_uploader(
                "Pilih file gambar (PNG, JPG, JPEG):",
                type=["png", "jpg", "jpeg"],
                key="manual_file_uploader"
            )

            if uploaded_file is not None:
                # Tampilkan preview
                try:
                    # Baca bytes untuk preview dan validasi
                    image_bytes = uploaded_file.getvalue()
                    st.image(image_bytes, caption="Preview Gambar yang Diupload", width=300) # Sesuaikan lebar preview

                    # Validasi sederhana (opsional tapi bagus)
                    try:
                        img = Image.open(BytesIO(image_bytes))
                        img.verify() # Coba load header gambar
                        st.caption(f"Tipe file terdeteksi: {img.format}, Ukuran: {len(image_bytes)/1024:.1f} KB")
                        is_image_valid = True
                    except Exception as e_img:
                        st.error(f"File yang diupload tampaknya bukan gambar yang valid: {e_img}")
                        is_image_valid = False

                    if is_image_valid:
                        # Tombol untuk menyimpan
                        if st.button(f"💾 Simpan Gambar untuk '{selected_word_manual}'", key="save_manual_img_btn"):
                            with st.spinner("Memproses dan menyimpan gambar..."):
                                try:
                                    # 1. Encode ke Base64 (string)
                                    base64_encoded_string = base64.b64encode(image_bytes).decode('utf-8')

                                    # 2. Reload data terbaru sebelum update
                                    current_df_manual = load_data(DATA_FILE)

                                    # 3. Cari index baris (ambil yang pertama jika duplikat)
                                    target_indices = current_df_manual[current_df_manual['word'] == selected_word_manual].index
                                    if not target_indices.empty:
                                        target_index = target_indices[0]

                                        # 4. Update kolom image_base64
                                        current_df_manual.loc[target_index, 'image_base64'] = base64_encoded_string
                                        # Opsional: Hapus referensi file lama jika ada?

                                        # 5. Simpan DataFrame
                                        save_data(current_df_manual, DATA_FILE)
                                        st.success(f"Gambar Base64 berhasil disimpan untuk kata '{selected_word_manual}'!")

                                        # 6. Rerun untuk membersihkan state uploader
                                        st.rerun()
                                    else:
                                         st.error(f"Kata '{selected_word_manual}' tidak ditemukan di kamus saat mencoba menyimpan gambar.")

                                except Exception as e_save:
                                    st.error(f"Gagal menyimpan data Base64: {e_save}")
                       

                except Exception as e_read:
                    st.error(f"Gagal membaca file yang diupload: {e_read}")

# ===========================================
# --- Halaman 5: Statistic --- (Asumsi ini halaman ke-5 atau sesuai urutan Anda)
# ===========================================
elif selected_page == "Statistic":
    st.markdown(f"<h1 style='text-align: center; color: black;'>Statistik Pembelajaran 📊</h1>", unsafe_allow_html=True) # Ganti warna ke putih agar kontras?

    # Muat data terbaru untuk statistik
    df_stats = load_data(DATA_FILE)

    if df_stats.empty:
        st.warning("Data kamus masih kosong, tidak ada statistik untuk ditampilkan.")
    else:
        # --- BARIS 1: Scorecards ---
        st.markdown("### Ringkasan Umum") # Subjudul untuk baris
        # Definisikan Kolom
        col1, col2, col3 = st.columns(3)

        with col1:
            # Metrik 1: Total Entri
            total_entries = len(df_stats)
            st.metric(label="Jumlah Total Entri", value=total_entries)

        with col2:
            # Metrik 2: Jumlah Bahasa Unik
            # Anda mungkin ingin menghapus NaN dulu jika kolom bahasa bisa kosong
            # num_languages = df_stats['language'].dropna().nunique()
            num_languages = df_stats['language'].nunique() # Asumsi tidak ada NaN atau NaN dihitung unik
            st.metric(label="Jumlah Bahasa", value=num_languages)

        with col3:
            # Metrik 3: Entri Baru Hari Ini
            new_today_count = "N/A" # Default jika error
            try:
                if 'date' in df_stats.columns:
                    # Konversi ke datetime, paksa error jadi NaT, lalu hapus NaT
                    valid_dates = pd.to_datetime(df_stats['date'], errors='coerce').dropna()
                    # Ambil HANYA bagian tanggal dari datetime hari ini
                    today_date = date.today()
                    # Bandingkan bagian tanggal dari kolom dengan tanggal hari ini
                    new_today_count = len(valid_dates[valid_dates.dt.date == today_date])
                else:
                    new_today_count = 0 # Jika kolom date tidak ada

            except Exception as e:
                st.error(f"Error menghitung entri hari ini: {e}", icon="⚠️") # Tampilkan error di app
                new_today_count = "Error"

            st.metric(label="Entri Baru Hari Ini", value=new_today_count)

        # --- Terapkan Styling ke Metric Cards ---
        # Panggil fungsi *setelah* semua st.metric dibuat di atas
        # Anda bisa memanggilnya tanpa argumen untuk gaya default, atau kustomisasi:
        style_metric_cards(
            border_left_color="#FF4B4B", # Contoh warna merah untuk garis kiri
            border_size_px = 2,          # Ukuran garis batas dalam pixel
            border_radius_px = 5,       # Radius lengkungan sudut
            box_shadow = True            # Aktifkan bayangan kotak
        )



        st.markdown("<hr>", unsafe_allow_html=True) # Garis pemisah
        

        # --- BARIS 2: Placeholder untuk Plot ---
        plot_col1, plot_col2 = st.columns(2)

        ################################## PLOT 1

        with plot_col1:
            st.markdown("##### Plot 1: Distribusi Bahasa")
            # Di sini nanti Anda akan menambahkan kode plot, contoh:
            # language_counts = df_stats['language'].value_counts()
            # st.bar_chart(language_counts)
            df_plot1 = df_stats.copy()
            df_plot1 = df_plot1.groupby("language")["language"].count().reset_index(name="count")

            # --- (Penting) Mengurutkan data sebelum plotting ---
            # Dengan go, seringkali lebih mudah mengurutkan data di DataFrame terlebih dahulu.
            # Urutkan dari terkecil ke terbesar agar bar teratas adalah yang terbesar di plot horizontal
            df_plot1 = df_plot1.sort_values(by="count", ascending=True)

            # 1. Membuat objek 'trace' Bar Chart
            # Ini mendefinisikan *bagaimana* data Anda akan digambar.
            trace1 = go.Bar(
                x=df_plot1['count'],        
                y=df_plot1['language'],     
                orientation='h',            
                name='Jumlah Bahasa',       
                text=df_plot1['count'],     
                textposition='inside',
                textfont=dict(size=16),     
                marker=dict(colorscale= "viridis", color=df_plot1['count']))

            # 2. Membuat objek 'layout'
            # Ini mendefinisikan tampilan keseluruhan chart (judul, label sumbu, dll.)
            layout = go.Layout(
                title=None
                ,
                xaxis=dict(
                    title=None,
                    tickfont=dict(size=15),
                    showline=True,         # <<< AKTIFKAN garis sumbu Y
                    linewidth=2.5,           # <<< Atur ketebalan garis sumbu Y
                    linecolor='black',
                ),
                yaxis=dict(
                    title= None,
                    tickfont=dict(size=15),
                    showline=True,         # <<< AKTIFKAN garis sumbu Y
                    linewidth=2.5,           # <<< Atur ketebalan garis sumbu Y
                    linecolor='black',

                ),
                bargap=0.5,
                plot_bgcolor="white",
                 width=800,  
                height=400, 
            )
            fig = go.Figure(data=[trace1], layout=layout)
            st.plotly_chart(fig, use_container_width=True)


        ################################## PLOT 2
        with plot_col2:
            st.markdown("##### Tren Penambahan Kata")
            # *** Penting: Pastikan data terurut berdasarkan tanggal ***

            df_line = df_stats.copy()
            df_line['date'] = pd.to_datetime(df_line['date'], errors='coerce')
            df_line = df_line.sort_values(by='date')
            df_aggregated = df_line.groupby('date').size().reset_index(name='count')


            # 1. Membuat Trace Scatter (Sama seperti sebelumnya)
            trace1 = go.Scatter(
                x=df_aggregated['date'],
                y=df_aggregated['count'],
                mode='lines+markers',
                name='Jumlah Data',
                line=dict(color='#39EC6F', width=2),
                marker=dict(color='#1B56FD', size=10),
                hovertemplate= # <<< TEMPLATE UNTUK KONTEN HOVER DIATUR DI SINI
                    # Menampilkan Tanggal dan Jumlah dengan label yang jelas:
                    "<b>Tanggal:</b> %{x|%d %b %Y}<br>" +  # Baris 1: Tanggal (format dd Mon YYYY)
                    "<b>Jumlah:</b> %{y}<extra></extra>"
            )

            # 2. Membuat Layout dengan Format Tanggal pada Sumbu X
            layout = go.Layout(
                
                xaxis=dict(
                    title='Tanggal',
                    tickformat='%d %b', 
                    showline=True, linewidth=2.5, linecolor='black',
                    rangeslider=dict(
                        visible=False
                    ),
                    showgrid=False,
                    tickfont=dict(size=15),
                    
                ),
                yaxis=dict(
                    title=None,
                    showline=True, linewidth=2.5, linecolor='black',
                    tickfont=dict(size=15),
                    showgrid=False
                ),
                plot_bgcolor='white',
                hovermode='x unified',
                width=800,  
                height=400, 

            )

            # 3. Membuat Figure (Sama seperti sebelumnya)
            fig = go.Figure(data=[trace1], layout=layout)

            # 4. Menampilkan Plot Interaktif
            st.plotly_chart(fig, use_container_width=True)

        
        plot_col1, plot_col2 = st.columns(2)
        ################################## PLOT 3
        with plot_col1:
            st.markdown("##### Wordcloud Kata Yang Sering Muncul")

            # -----------------------------------------------------------

            # 1. Salin DataFrame
            df_word_cloud = df_stats.copy()

            # 2. Pastikan kolom 'example_sentence' ada dan persiapan awal
            if 'example_sentence' not in df_word_cloud.columns:
                # Gunakan st.error untuk menampilkan pesan error di Streamlit
                st.error("Error: Kolom 'example_sentence' tidak ditemukan dalam DataFrame.")
            else:
                df_word_cloud = df_word_cloud.dropna(subset=['example_sentence'])
                df_word_cloud['example_sentence'] = df_word_cloud['example_sentence'].astype(str)

                # --- Inisialisasi Alat Pembersihan Teks ---
                stop_words = set(stopwords.words('english')) # Stopwords B. Inggris
                stemmer = PorterStemmer()
                punctuation_table = str.maketrans('', '', string.punctuation)
                # -----------------------------------------

                # --- Fungsi untuk Membersihkan Satu Teks ---
                def preprocess_text(text):
                    text = text.lower()
                    text = text.translate(punctuation_table)
                    tokens = word_tokenize(text)
                    processed_tokens = []
                    for word in tokens:
                        if word.isalpha() and word not in stop_words and len(word) > 1:
                            processed_tokens.append(word)
                    stemmed_tokens = [stemmer.stem(word) for word in processed_tokens]
                    return stemmed_tokens
                # -----------------------------------------

                # --- Proses Teks ---
                all_processed_tokens = []

                total_rows = len(df_word_cloud['example_sentence'])
                for i, text_entry in enumerate(df_word_cloud['example_sentence']):
                    # status_text.text(f"Memproses teks {i+1}/{total_rows}")
                    processed_tokens = preprocess_text(text_entry)
                    all_processed_tokens.extend(processed_tokens)

                # 3. Hitung Frekuensi Kata
                word_frequencies = Counter(all_processed_tokens)

                # Cek apakah ada kata setelah pembersihan
                if not word_frequencies:
                    # Gunakan st.warning untuk pesan peringatan
                    st.warning("Tidak ada kata yang tersisa setelah proses pembersihan teks.")
                else:
                    # 4. Buat objek WordCloud
                    wordcloud_generator = WordCloud(
                        width=800, # Sedikit sesuaikan ukuran jika perlu
                        height=400,
                        background_color='white',
                        colormap='plasma',
                        max_words=150,
                        contour_width=1,
                        contour_color='grey',
                        collocations=False
                    )
                    try:
                        # 5. Generate WordCloud image
                        wordcloud_image = wordcloud_generator.generate_from_frequencies(word_frequencies)

                        # --- 6. Tampilkan Word Cloud menggunakan Matplotlib & Streamlit ---

                        #    a. Buat figure dan axes Matplotlib secara eksplisit
                        fig, ax = plt.subplots(figsize=(10, 5)) # figsize opsional

                        #    b. Tampilkan gambar word cloud pada axes ('ax')
                        ax.imshow(wordcloud_image, interpolation='bilinear')

                        #    c. Matikan sumbu pada axes ('ax')
                        ax.axis('off')

                        #    d. Tampilkan figure ('fig') di Streamlit
                        st.pyplot(fig) # <<< PERBAIKAN UTAMA DI SINI

                    except Exception as e:
                        # Gunakan st.error untuk menampilkan error di Streamlit
                        st.error(f"Terjadi error saat membuat atau menampilkan word cloud: {e}")


        ################################## PLOT 4
        with plot_col2:
            st.markdown("##### Persentase Keterangan Gambar")
            # 1. Data Preparation (dengan perbaikan variabel)
            df_temp = df_stats.copy() # Gunakan nama sementara
            # Buat kategori baru, tangani NaN dan string kosong
            df_temp["kategori_gambar"] = np.where(
                df_temp["image_base64"].isna() | (df_temp["image_base64"] == ''),
                "Tanpa Gambar",
                "Dengan Gambar"
            )

            # Grouping (hasil disimpan di df_grouped)
            df_grouped = df_temp.groupby(["kategori_gambar","language"])["language"].count().reset_index(name="count")

            # Agregasi Level 1 (dari df_grouped)
            df_level1 = df_grouped.groupby('kategori_gambar')['count'].sum().reset_index()

            # --- Persiapan Data untuk go.Sunburst (ids, labels, parents, values) ---
            ids = []
            labels = []
            parents = []
            values = []
            level1_labels_ordered = df_level1['kategori_gambar'].tolist() # Simpan urutan L1

            # Tambahkan data Level 1
            for label, count in zip(level1_labels_ordered, df_level1['count'].tolist()):
                ids.append(label)
                labels.append(label)
                parents.append("")
                values.append(count)

            # Tambahkan data Level 2 (iterasi pada df_grouped)
            for index, row in df_grouped.iterrows():
                level2_id = f"{row['kategori_gambar']} - {row['language']}"
                ids.append(level2_id)
                labels.append(row['language'])
                parents.append(row['kategori_gambar'])
                values.append(row['count'])
            # ----------------------------------------

            # --- Persiapan Warna (L1 specific, L2 unique end Viridis) ---
            viridis_colors = px.colors.sequential.Viridis
            segment_colors = []

            # Warna Level 1 (Specific: Tanpa Gambar=lightgrey, Lainnya=Start Viridis)
            level1_colors_final = []
            viridis_start_index = 0
            for label in level1_labels_ordered: # Gunakan urutan label L1
                if label == "Tanpa Gambar":
                    level1_colors_final.append('lightgrey')
                else:
                    if viridis_start_index < len(viridis_colors):
                        level1_colors_final.append(viridis_colors[viridis_start_index])
                        viridis_start_index += 1
                    else:
                        level1_colors_final.append('darkgrey')
            segment_colors.extend(level1_colors_final)

            # Warna Level 2 (Unique per language from end Viridis) - Gunakan df_grouped
            unique_languages = df_grouped['language'].unique().tolist() # <<< Dari df_grouped
            num_unique_languages = len(unique_languages)
            num_end_colors_needed = num_unique_languages
            # ... (logika ambil end_viridis_pool seperti sebelumnya) ...
            if num_end_colors_needed <= len(viridis_colors):
                end_viridis_pool = viridis_colors[-num_end_colors_needed:]
            else:
                # print(f"Peringatan:...") # Optional
                multiplier = (num_end_colors_needed // len(viridis_colors)) + 1
                extended_viridis = viridis_colors * multiplier
                end_viridis_pool = extended_viridis[-num_end_colors_needed:]

            language_to_end_color_map = {lang: color for lang, color in zip(unique_languages, end_viridis_pool)}
            level2_languages_ordered = df_grouped['language'].tolist() # <<< Dari df_grouped
            level2_colors = [language_to_end_color_map[lang] for lang in level2_languages_ordered]
            segment_colors.extend(level2_colors)
            # ----------------------------------------------------------

            # --- Persiapan Text Template ---
            text_templates = []

            # Template untuk Level 1: Label (bold) + newline + Persentase thd Total (1 desimal)
            level1_template = "<b>%{label}</b><br>%{percentRoot:.1%}"
            text_templates.extend([level1_template] * len(df_level1)) # Ulangi template L1 sebanyak segmen L1

            # Template untuk Level 2: Hanya Label Bahasa
            level2_template = "%{label}"
            text_templates.extend([level2_template] * len(df_grouped)) # Ulangi template L2 sebanyak segmen L2
            # -----------------------------

            # 3. Buat Trace go.Sunburst dengan texttemplate
            trace = go.Sunburst(
                ids=ids,
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                # hoverinfo tetap bisa digunakan atau diganti hovertemplate kustom
                # hoverinfo='label+percent parent+value',
                hovertemplate="<b>%{label}</b><br>Jumlah: %{value}<br>%{percentRoot:.1%} dari Total<extra></extra>", # Hover kustom
                marker=dict(
                    colors=segment_colors
                ),
                texttemplate=text_templates,  # <<< Gunakan list template teks
                # textinfo='none', # <<< Biasanya tidak perlu jika texttemplate dipakai
                insidetextorientation='radial', # Atur orientasi teks agar mudah dibaca
                textfont_size=11,          # Atur ukuran font teks pada segmen
                # outside_textfont_size # jika teks meluber keluar
            )

            # 4. Buat Layout
            layout = go.Layout(
                title=None, # Beri judul
                margin=dict(t=50, l=25, r=25, b=25),
                width=800,  
                height=400, 
            )

            # 5. Buat Figure dan Tampilkan
            fig = go.Figure(data=[trace], layout=layout)
            st.plotly_chart(fig, use_container_width=True)




# --- Footer atau bagian akhir lainnya ---
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi Kamus Bahasa v1.1")
