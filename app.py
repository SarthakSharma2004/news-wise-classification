import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title='NewsWise',
    layout='wide',
    page_icon='📰'
)

# Background and Sidebar Colors
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d1117;
    }
    [data-testid="stSidebar"] {
        background-color: #161b22;
    }
    /* Remove top padding */
    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Sidebar Navigation ------------------
with st.sidebar:
    page = option_menu(
        None,
        ["Home", "About"],
        icons=["house", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"background-color": "#161b22"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#0d1117"},
        }
    )

# ------------------ Load your saved pipeline ------------------

@st.cache_resource
def load_model():
    model = joblib.load("/Users/sarthaksharna/NewsWise/model/tfidf_svc_bbc_classifier.pkl")
    return model


model = load_model()

# ------------------ Home Page ------------------

if page == "Home":
    # Title
    st.markdown(
        """
        <h1 style='text-align: center; color: #9a4dff; margin-bottom: 10px;'>
            NewsWise 📰
        </h1>
        """,
        unsafe_allow_html=True
    )

    # Spacer
    st.markdown("---")

    # User Input Area with neon border
    st.markdown(
        """
        <style>
        textarea {
            border: 2px solid #9a4dff !important;
            border-radius: 10px !important;
            box-shadow: 0 0 10px #9a4dff;
            background-color: #161b22 !important;
            color: white !important;
            font-size: 16px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
    """
    <h3 style='text-align: center; color: white;'>
        ✍️ Feed your news snippet here and let NewsSense flex its AI muscles!
    </h3>
    """,
    unsafe_allow_html=True
    )   

    st.markdown("<br><br><br><br>", unsafe_allow_html=True)

    user_input = st.text_area(
        "",
        placeholder="Paste or type your news snippet here...",
        height=200
    )

    # Neon Button Styling
    st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        background-color: #0d1117;
        color: #9a4dff;
        border: 2px solid #9a4dff;
        border-radius: 10px;
        padding: 0.5em 1em;
        font-size: 18px;
        transition: all 0.3s ease;
        width: 50%;
        max-width: 250px;
        margin: auto;
        display: block;
    }
    div.stButton > button:first-child:hover {
        background-color: #9a4dff;
        color: white;
        border: 2px solid #ffffff;
        box-shadow: 0 0 20px #9a4dff;
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True
)

    # Predict Button
    if st.button("🚀 Classify News", use_container_width=True):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text before clicking Classify.")
        else:
            predicted_class = model.predict([user_input])[0]
            category_map = {
                0: "Business",
                1: "Entertainment",
                2: "Politics",
                3: "Sport",
                4: "Tech"
            }
            predicted_category = category_map.get(predicted_class, "Unknown")

            st.markdown("---")
            st.markdown(
                f"""
                <h2 style='text-align: center; color: #4c5ce1; font-size: 36px;'>
                    ✅ Predicted Category: {predicted_category}
                </h2>
                """,
                unsafe_allow_html=True
            )

# -------------------- ABOUT PAGE --------------------------------------------------

if page == "About":
    st.markdown(
        """
        <h1 style="color: #f56315; font-size: 40px; text-align: left;">
            About NewsWise
        </h1>
        <br><br><br><br><br>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div style="
            background-color: #161b22;
            padding: 30px;
            border-radius: 10px;
            border-left: 5px solid #e78020;
            color: #c9d1d9;
            font-size: 18px;
            line-height: 1.7;
            max-width: 100%;
            text-align: left;
        ">

        <h3 style="color: #e78020;">📌 Problem Statement</h3>
        The digital world generates an overwhelming volume of news articles daily, making it challenging for platforms and readers to efficiently organize and consume content by relevant topics.

        <br><br><br>

        <h3 style="color: #e78020;">🎯 Objective</h3>
        To build a system capable of efficiently categorizing news articles into clear, actionable topics like business, technology, sports, entertainment, and politics, enabling streamlined content delivery and personalized reading experiences.

        <br><br><br>

        <h3 style="color: #e78020;">🛠️ Solution</h3>
        Developed a machine learning system to classify news articles using different text representation and word embedding techniques , including:
        <ul>
            <li>TF-IDF Vectorization</li>
            <li>Bag of Words (BoW)</li>
            <li>Custom-trained Word2Vec embeddings</li>
        </ul>
        After rigorous experimentation and hyperparameter tuning across models, a TF-IDF vectorizer combined with a Support Vector Machine (SVM) classifier achieved the best performance.

        <br><br><br>

        <h3 style="color: #e78020;">⚡ Results</h3>
        The optimized model achieved an impressive accuracy of <b style="color: #58a6ff;">98.3%</b> in categorizing news articles, demonstrating robust generalization and practical readiness for real-time deployment.

        <br><br><br>

        <h3 style="color: #e78020;">🚀 About NewsSense</h3>
        <b>NewsSense</b> is a lightweight, interactive web application that allows users to paste or type news snippets and instantly receive topic categorizations using our fine-tuned ML pipeline.

        </div>
        """,
        unsafe_allow_html=True
    )


# ------------------------------ END OF CODE -----------------------------------------