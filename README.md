# 🫀 MediSense — Heart Health Risk & Wellness Guide  

MediSense is a **machine learning–powered web application** that predicts **heart health risk** and provides **personalized wellness guidance**.  
Built on the **BRFSS 2022 dataset**, it combines data science with an interactive UI to deliver actionable health insights in a user-friendly way.  

⚠️ **Disclaimer**: This tool is for **educational purposes only**. It is **not medical advice**.  

---

## 🌟 Features at a Glance

✨ **Risk Prediction**  
- Enter age, BMI, sleep hours, smoking/alcohol status, diabetes history, and more.  
- Get a **personalized probability (%)** of heart disease risk.  

📊 **Interactive Visualizations**  
- 📈 **Risk Factor Analysis** → see how lifestyle changes affect your risk.  
- 🗺️ **US Risk Map** → compare your predicted risk with state & national averages.  
- 🧾 **Personal Report** → BMI gauge, health profile, and action plan.  

🏋️ **Wellness Guidance**  
- 🏃‍♂️ Exercise suggestions with embedded YouTube videos.  
- 🥗 Tier-based dietary advice + illustrated **food gallery**.  
- 💡 Daily health tips that update every day.  

📥 **Downloadable Report**  
- Export a **Markdown summary** with your results & recommendations.  

---

## 🧠 Machine Learning Workflow

🧹 **Data Preprocessing**  
- Selected key health features (BMI, sleep, diabetes, activity, etc.)  
- Handled missing values (imputation)  
- One-Hot Encoding for categorical variables  
- Standardization of numeric features  

🤖 **Modeling**  
- Compared **Logistic Regression** and **Random Forest**  
- Evaluated with Accuracy, F1-score, ROC-AUC  
- Selected the best model based on **highest ROC-AUC**:contentReference[oaicite:0]{index=0}  

💾 **Artifacts & Deployment**  
- Trained model saved with **Joblib**  
- Frontend built with **Streamlit**  
- Interactive charts with **Plotly**  

---

## 🛠️ Tech Stack

- 🐍 Python (Pandas, NumPy, Scikit-learn, Joblib)  
- 📊 Plotly, Seaborn, Matplotlib  
- 🌐 Streamlit, Streamlit-Lottie  
- 🗄️ BRFSS 2022 Dataset (CDC)  
- ☁️ Ready for deployment via **Streamlit Cloud / Docker**  

---

## 📷 Screenshots (Examples)

- 🏠 **Home Page — Risk Prediction**  
- 📈 **BMI Gauge & Explanation**  
- 🗺️ **US Risk Map vs Personal Risk**  
- 🥗 **Diet & Exercise Recommendations**  

*(Screenshots can be added from `/assets/` directory)*  

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+  
- Virtual environment recommended  

### Installation
```bash
git clone https://github.com/<your-username>/MediSense.git
cd MediSense
pip install -r requirements.txt
