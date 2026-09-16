import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# ==========================================
# 1. KONFIGURASI HALAMAN & CUSTOM CSS
# ==========================================
st.set_page_config(
    page_title="Sistem Analisis Risiko Penerbangan",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeksi CSS Kustom agar styling metric-box & title berfungsi
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        color: #4B5563;
        margin-bottom: 25px;
    }
    .metric-box {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
    .metric-box h4 {
        margin: 0;
        font-size: 14px;
        color: #6B7280;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL MACHINE LEARNING & ENCODER (.PKL)
# ==========================================
@st.cache_resource
def load_all_artifacts():
    model_xgboost = None
    encoder = None
    
    try:
        with open("model_xgboost.pkl", "rb") as f:
            model_xgboost = pickle.load(f)
    except Exception as e:
        st.error(f"Error XGBoost: {e}")

    try:
        with open("encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
    except Exception as e:
        st.error(f"Error Encoder: {e}")
        
    return model_xgboost, encoder

# ==========================================
# 3. SIDEBAR NAVIGASI
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/f/f1/Gunadarma_University_Logo.png", width=100)
st.sidebar.title("Navigasi Sistem")

# Navigasi dipangkas menjadi 2 menu utama
menu = st.sidebar.radio(
    "Pilih Halaman:",
    ["Dashboard & Informasi Model", "Prediksi Tingkat Keparahan"],
    key="main_navigation_menu"
)

# ==========================================
# MENU 1: DASHBOARD & INFORMASI MODEL
# ==========================================
if menu == "Dashboard & Informasi Model":
    st.markdown('<div class="main-title">✈️ Sistem Analisis & Prediksi Risiko Penerbangan</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Universitas Gunadarma - Fakultas Teknologi Industri</div>', unsafe_allow_html=True)
    
    # Metric Cards Ringkasan
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-box"><h4>Total Insiden Historis</h4><p style="font-size: 24px; font-weight: bold; color: #1E3A8A; margin:0;">150,000+ Records</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-box"><h4>Fitur Prediktor Utama</h4><p style="font-size: 24px; font-weight: bold; color: #10B981; margin:0;">5 Dimensi Kritis</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-box"><h4>Algoritma Klasifikasi</h4><p style="font-size: 24px; font-weight: bold; color: #F59E0B; margin:0;">XGBoost</p></div>', unsafe_allow_html=True)

    st.write("")
    st.info("""
        **Deskripsi Sistem:**
        Sistem ini dibangun untuk memodelkan risiko dan memprediksi tingkat keparahan (*Severity*) insiden penerbangan berdasarkan data historis menggunakan algoritma *Ensemble Machine Learning* (XGBoost).
    """)
    
    st.write("---")
    st.subheader("📊 Visualisasi & Eksplorasi Data Interaktif")

    # Visualisasi Dua Kolom Interaktif (Plotly)
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("##### 🍩 Distribusi Tingkat Keparahan (Overall Severity)")
        severity_data = pd.DataFrame({
            'Tingkat Keparahan': ['Non-Fatal', 'Fatal', 'Incident'],
            'Jumlah Insiden': [65000, 12000, 3000]
        })
        fig_donut = px.pie(
            severity_data, 
            values='Jumlah Insiden', 
            names='Tingkat Keparahan',
            hole=0.5,
            color='Tingkat Keparahan',
            color_discrete_map={'Non-Fatal': '#10B981', 'Fatal': '#EF4444', 'Incident': '#3B82F6'}
        )
        fig_donut.update_layout(margin=dict(t=20, b=20, l=10, r=10), height=300)
        st.plotly_chart(fig_donut, use_container_width=True)

    with chart_col2:
        st.markdown("##### ⚡ Kinerja Model XGBoost")
        model_metrics = pd.DataFrame({
            'Metrik': ['Akurasi XGBoost'],
            'Nilai (%)': [84.55]
        })
        fig_bar = px.bar(
            model_metrics, 
            x='Metrik', 
            y='Nilai (%)', 
            text='Nilai (%)',
            color='Metrik',
            color_discrete_sequence=['#1E3A8A']
        )
        fig_bar.update_yaxes(range=[0, 100])
        fig_bar.update_layout(margin=dict(t=20, b=20, l=10, r=10), height=300, showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Interactive Filter & Drill-Down Section
    st.write("---")
    st.subheader("🔍 Profil Risiko Berdasarkan Fase Penerbangan")
    
    selected_phase = st.selectbox(
        "Pilih Fase Penerbangan untuk Melakukan Filtering Data:",
        ["TAKEOFF", "LANDING", "APPROACH", "CRUISE", "CLIMB", "MANEUVERING"]
    )

    phase_data_map = {
        "TAKEOFF": {"Non-Fatal": 12400, "Fatal": 2100, "Incident": 450},
        "LANDING": {"Non-Fatal": 18200, "Fatal": 950, "Incident": 800},
        "APPROACH": {"Non-Fatal": 6100, "Fatal": 2300, "Incident": 200},
        "CRUISE": {"Non-Fatal": 8300, "Fatal": 3100, "Incident": 300},
        "CLIMB": {"Non-Fatal": 4200, "Fatal": 1100, "Incident": 150},
        "MANEUVERING": {"Non-Fatal": 2800, "Fatal": 1900, "Incident": 90}
    }

    current_phase_data = phase_data_map.get(selected_phase, {"Non-Fatal": 5000, "Fatal": 1000, "Incident": 200})
    df_phase = pd.DataFrame({
        'Kategori': list(current_phase_data.keys()),
        'Jumlah Kasus': list(current_phase_data.values())
    })

    fig_phase = px.bar(
        df_phase, 
        x='Jumlah Kasus', 
        y='Kategori', 
        orientation='h',
        color='Kategori',
        text='Jumlah Kasus',
        color_discrete_map={'Non-Fatal': '#10B981', 'Fatal': '#EF4444', 'Incident': '#3B82F6'}
    )
    fig_phase.update_layout(height=280, margin=dict(t=10, b=10, l=10, r=10), showlegend=False)
    
    col_p1, col_p2 = st.columns([2, 1])
    with col_p1:
        st.plotly_chart(fig_phase, use_container_width=True)
    with col_p2:
        st.write("")
        st.markdown(f"**📌 Highlight Fase {selected_phase}:**")
        total_cases = sum(current_phase_data.values())
        fatal_rate = (current_phase_data['Fatal'] / total_cases) * 100
        
        st.metric("Total Insiden Tercatat", f"{total_cases:,}")
        st.metric("Rasio Fatalitas (Fatality Rate)", f"{fatal_rate:.1f}%")
        
        if fatal_rate > 20:
            st.error("⚠️ Fase ini tergolong berisiko tinggi (*High Risk Phase*).")
        else:
            st.success("✅ Fase ini memiliki *survival rate* relatif tinggi.")

    # INTEGRASI INFORMASI MODEL & DATASET (Dahulunya Halaman 3)
    st.write("---")
    with st.expander("📚 **Detail Metodologi Riset & Atribut Dataset (Bab 3)**", expanded=False):
        col_m1, col_m2 = st.columns(2)
        
        with col_m1:
            st.markdown("""
            ##### ⚙️ Spesifikasi Riset & Model
            * **Metodologi Penelitian:** CRISP-DM (*Cross-Industry Standard Process for Data Mining*)
            * **Algoritma Utama:** XGBoost (*Extreme Gradient Boosting*)
            * **Evaluasi Kinerja:** Akurasi Model **84,55%**
            """)
            
        with col_m2:
            st.markdown("""
            ##### 📋 5 Atribut Prediktor Krusial (Bab 3.3)
            1. `Weather Condition` (Kondisi Cuaca)
            2. `Broad Phase of Flight` (Fase Penerbangan)
            3. `Aircraft Damage` (Tingkat Kerusakan Pesawat)
            4. `Number of Engines` (Jumlah Mesin)
            5. `Engine Type` (Tipe Mesin Pesawat)
            """)

# ==========================================
# MENU 2: PREDIKSI TINGKAT KEPARAHAN
# ==========================================
elif menu == "Prediksi Tingkat Keparahan":
    st.subheader("🔮 Form Prediksi Risiko (XGBoost)")
    st.write("Silakan masukkan parameter kondisi penerbangan di bawah ini untuk menguji prediksi model:")

    col1, col2 = st.columns(2)
    
    with col1:
        weather_display = st.selectbox(
            "1. Kondisi Cuaca (Weather Condition):",
            [
                "VMC - Visual Meteorological Conditions (Cuaca Cerah / Visual)", 
                "IMC - Instrument Meteorological Conditions (Cuaca Buruk / Instrumen)", 
                "UNK - Unknown (Tidak Diketahui)"
            ]
        )
        weather = weather_display[:3]
        
        phase = st.selectbox(
            "2. Fase Penerbangan (Broad Phase of Flight):",
            ["TAKEOFF", "CLIMB", "CRUISE", "DESCENT", "APPROACH", "LANDING", "MANEUVERING", "TAXI", "GO-ROUND", "STANDING", "UNKNOWN"]
        )
        damage = st.selectbox(
            "3. Kerusakan Pesawat (Aircraft Damage):",
            ["Substantial", "Destroyed", "Minor", "None", "Unknown"]
        )
        
    with col2:
        num_engines = st.slider(
            "4. Jumlah Mesin (Number of Engines):",
            min_value=1, max_value=4, value=1, step=1
        )
        engine_type = st.selectbox(
            "5. Tipe Mesin (Engine Type):",
            ["Reciprocating", "Turbo Prop", "Turbo Jet", "Turbo Fan", "Turbo Shaft", "Unknown"]
        )

    st.write("---")
    
    if st.button("🚀 Hitung Estimasi Risiko / Prediksi Keparahan", use_container_width=True):
        
        # DataFrame Input disesuaikan NAMA KOLOM SAMA PERSIS DENGAN NOTEBOOK
        raw_input = pd.DataFrame([{
            "Weather Condition": weather,
            "Broad Phase of Flight": phase,
            "Aircraft Damage": damage,
            "Number of Engines": int(num_engines),
            "Engine Type": engine_type
        }])
        
        st.write("**Data Input Pengguna:**")
        st.markdown(raw_input.to_html(index=False, classes='table'), unsafe_allow_html=True)
        st.write("")
        
        model_xgboost, encoder = load_all_artifacts() 
        
        # PENGECEKAN KETERSEDIAAN MODEL & ENCODER
        if model_xgboost is None or encoder is None:
            st.error("⚠️ File `model_xgboost.pkl` atau `encoder.pkl` tidak ditemukan! Pastikan file berada di direktori aplikasi.")
        else:
            try:
                # Transformasi input menggunakan OneHotEncoder otomatis
                X_input = encoder.transform(raw_input)
                
                # Melakukan Prediksi
                prediction = model_xgboost.predict(X_input)
                target_labels = {0: "Incident", 1: "Non-Fatal", 2: "Fatal"}
                result = target_labels.get(prediction[0], "Unknown")
                
                # Hitung Probabilitas
                prob_text = ""
                if hasattr(model_xgboost, 'predict_proba'):
                    probabilities = model_xgboost.predict_proba(X_input)
                    max_prob = np.max(probabilities[0]) * 100
                    prob_text = f"(Probabilitas: {max_prob:.2f}%)"
                
                # TAMPILKAN HASIL PREDIKSI
                if result == "Fatal":
                    st.error(f"### HASIL PREDIKSI (XGBoost): **{result}** {prob_text}")
                elif result == "Non-Fatal":
                    st.success(f"### HASIL PREDIKSI (XGBoost): **{result}** {prob_text}")
                else:
                    st.info(f"### HASIL PREDIKSI (XGBoost): **{result}** {prob_text}")

            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat prediksi: {str(e)}")

        # EXPLAINABLE AI SECTION
        with st.expander("🔍 **Lihat Penjelasan & Analisis Faktor Risiko (Dinamis)**", expanded=True):
            st.markdown("### 💡 Interpretasi Faktor Input terhadap Prediksi:")
            
            weather_dict = {
                "VMC": "☀️ **Cuaca Cerah (VMC):** Menurunkan risiko fatalitas secara signifikan. Visibilitas penerbangan visual yang jernih memberikan ruang bagi pilot untuk bermanuver dan melakukan *forced landing* secara terkontrol.",
                "IMC": "🌧️ **Cuaca Buruk/Instrumen (IMC):** Meningkatkan risiko kecelakaan fatal. Visibilitas terbatas memaksa navigasi bergantung penuh pada instrumen, meningkatkan potensi disorientasi spasial.",
                "UNK": "❓ **Cuaca Tidak Diketahui (UNK):** Faktor lingkungan tidak dapat dikuantifikasi secara pasti dalam inferensi ini."
            }
            phase_dict = {
                "TAKEOFF": "🛫 **Fase Lepas Landas (TAKEOFF):** Risiko insiden tinggi karena daya mesin maksimal, namun kedekatan dengan area pendaratan darurat bandara dapat membantu mitigasi korban jiwa.",
                "CLIMB": "🧗 **Fase Menanjak (CLIMB):** Pesawat berada dalam transisi ke Ketinggian Jelajah; gangguan tenaga mesin pada fase ini menuntut penanganan darurat yang cepat.",
                "CRUISE": "✈️ **Fase Jelajah (CRUISE):** Fase relatif paling stabil, namun insiden pada ketinggian tinggi umumnya memiliki dampak kerusakan berat jika terjadi kegagalan sistem utama.",
                "DESCENT": "📉 **Fase Menurun (DESCENT):** Persiapan memasuki area pendekatan bandara, risiko dipengaruhi oleh ketepatan navigasi dan kondisi cuaca setempat.",
                "APPROACH": "🛬 **Fase Pendekatan (APPROACH):** Salah satu fase paling kritis (*critical phase*); resiko benturan dengan daratan (*CFIT*) meningkat jika visibilitas buruk.",
                "LANDING": "🛬 **Fase Pendaratan (LANDING):** Memiliki frekuensi insiden tinggi (seperti *runway excursion*), namun kecepatan pesawat yang relatif rendah cenderung menghasilkan tingkat kelangsungan hidup (*survival rate*) lebih tinggi.",
                "MANEUVERING": "🔄 **Fase Manuver (MANEUVERING):** Penerbangan di ketinggian rendah dengan sudut belok tajam meningkatkan risiko *stall* atau kehilangan kendali.",
                "TAXI": "🚜 **Fase Taxi (TAXI):** Pergerakan pelan di darat; risiko fatalitas jiwa sangat rendah, dominan hanya kerusakan struktur minor pada pesawat.",
                "GO-ROUND": "🔄 **Fase Batal Mendarat (GO-ROUND):** Pembatalan pendaratan menuntut akselerasi mendadak pada ketinggian rendah, memerlukan kewaspadaan tinggi.",
                "STANDING": "🅿️ **Fase Parkir/Berhenti (STANDING):** Pesawat berada di posisi diam; potensi fatalitas korban jiwa hampir tidak ada.",
                "UNKNOWN": "❓ **Fase Tidak Diketahui (UNKNOWN):** Informasi fase operasional tidak tercatat pada dataset historis."
            }
            damage_dict = {
                "Substantial": "🔧 **Kerusakan Substantial:** Struktur utama pesawat mengalami kerusakan fisik berlebih, namun integritas kabin/kokpit umumnya masih mampu melindungi penumpang dari benturan fatal.",
                "Destroyed": "💥 **Pesawat Hancur (Destroyed):** Energi benturan sangat besar hingga menghancurkan struktur utama pesawat. Ini merupakan faktor pendorong paling kuat terhadap hasil keparahan **Fatal**.",
                "Minor": "🛠️ **Kerusakan Minor:** Kerusakan fisik ringan pada kompartemen pesawat; risiko keselamatan jiwa penumpang sangat rendah.",
                "None": "✅ **Tidak Ada Kerusakan (None):** Pesawat dalam kondisi utuh, indikator keparahan cenderung **Non-Fatal / Incident**.",
                "Unknown": "❓ **Tingkat Kerusakan Tidak Diketahui:** Dampak structural tidak dapat diproyeksikan."
            }
            engine_dict = {
                "Reciprocating": "🛩️ **Mesin Piston (Reciprocating):** Umum digunakan pada pesawat penerbangan umum (*general aviation*) berkuran kecil. Kecepatan jelajah dan kecepatan benturan yang lebih rendah menekan risiko keparahan fatal.",
                "Turbo Prop": "🌀 **Mesin Turboprop:** Digunakan pada pesawat regional/baling-baling; memiliki tingkat keandalan mekanis menengah-tinggi.",
                "Turbo Jet": "🚀 **Mesin Turbojet:** Mesin jet kecepatan tinggi; kapasitas massa dan energi kinetik saat insiden tergolong tinggi.",
                "Turbo Fan": "✈️ **Mesin Turbofan:** Standar komersial modern dengan tingkat keandalan keselamatan yang sangat tinggi (*high reliability*).",
                "Turbo Shaft": "🚁 **Mesin Turboshaft:** Umum digunakan pada helikopter; karakteristik risiko terkait erat dengan manuver rotasi penerbangan.",
                "Unknown": "❓ **Tingkat Mesin Tidak Diketahui:** Karakteristik propulsi tidak dapat diidentifikasi secara pasti."
            }
            
            st.markdown(f"- {weather_dict.get(weather, '')}")
            st.markdown(f"- {phase_dict.get(phase, '')}")
            st.markdown(f"- {damage_dict.get(damage, '')}")
            st.markdown(f"- {engine_dict.get(engine_type, '')}")
            
            if num_engines == 1:
                st.markdown("- 1️⃣ **Jumlah Mesin (1 Unit):** Pesawat mesin tunggal tidak memiliki redundansi tenaga. Jika terjadi kegagalan mesin, pesawat harus segera melakukan pendaratan darurat (*gliding/forced landing*).")
            elif num_engines == 2:
                st.markdown("- 2️⃣ **Jumlah Mesin (2 Unit):** Pesawat memiliki redundansi daya dasar (*one-engine inoperative capability*), memungkinkan penerbangan berlanjut terbatas jika satu mesin mati.")
            else:
                st.markdown(f"- 🔢 **Jumlah Mesin ({num_engines} Unit):** Tingkat redundansi sistem propulsi sangat tinggi, meminimalisir risiko kehilangan daya total di udara.")