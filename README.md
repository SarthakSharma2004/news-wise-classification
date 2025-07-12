# 📰 NewsWise

## ✨ News Categorization 

### Link to App :

https://news-wise.streamlit.app/

---

## 📌 **What is NewsWise?**

NewsWise is a sleek, interactive **Streamlit app** that **automatically categorizes news snippets** into clear, actionable topics like:

- 🏦 **Business**
- 🎬 **Entertainment**
- 🗳️ **Politics**
- ⚽ **Sport**
- 💻 **Tech**

It leverages **machine learning models (TF-IDF + SVM) trained with 98.3% accuracy** to deliver instant predictions in a **modern, dark-themed interface.**

---

## 🎯 **Objective**

The modern digital world generates **millions of news articles daily**, making it challenging to consume and organize relevant content.

**NewsWise aims to:**

✅ Enable **clean, actionable categorization** of news articles.  
✅ Support **streamlined content delivery** for readers and platforms.  
✅ Serve as a hands-on ML deployment project showcasing text classification.

---

## 🛠️ **Tech Stack**

- **Frontend:** Streamlit (interactive app, neon-themed UI)

- **Frameworks and Libraries:** Python, Scikit-learn, gensim, nltk

- **ML Techniques Used:** 
   - TF-IDF Vectorization
   - Bag of Words (BoW)
   - Custom-trained Word2Vec 
   - Classification Models

- **Deployment:** Easily deployable on Streamlit Community Cloud or Render.

---

## 🚀 **Features**

✨ **Paste or type any news snippet to get instant classification.**  
✨ **Dark, modern UI with neon highlights for an aesthetic experience.**  
✨ **Interactive sidebar for seamless navigation.**  
✨ **Lightweight, fast, and beginner-friendly codebase.**

---


### **How this helps:**

✅ **Clean, modern structure:** Good spacing and sections for clarity.  
✅ **Uses bold headings for easy scanning.**  
✅ **Uses emojis for visual appeal.**  
✅ **Looks excellent on GitHub on both desktop and mobile.**  
✅ **Ready to copy-paste into your repo directly.**


---


## 🗂️ **Project Structure**

news-wise-classification/

│
├── app.py # Main Streamlit app

├── model/

│ └── tfidf_svc_bbc_classifier.pkl # Saved ML model

├── requirements.txt # Dependencies

└── README.md # Project documentation


---

## 🛠️ **Setup Instructions**

1️⃣ Clone the repository:
```bash
git clone https://github.com/SarthakSharma2004/news-wise-classification.git
cd news-wise-classification


pip install -r requirements.txt


streamlit run app.py


# ---------------------------------------



