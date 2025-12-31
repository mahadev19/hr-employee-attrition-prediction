# 🏢 HR Analytics: Employee Attrition Prediction

**An end-to-end machine learning project to predict employee attrition (turnover) and provide actionable HR insights.**

Built using Python, Scikit-learn, Pandas, Matplotlib/Seaborn, and deployed as an interactive web app with **Streamlit**.

<img width="1365" height="722" alt="image" src="https://github.com/user-attachments/assets/310c16ce-f34a-4ad6-b04d-07abd31ae5e6" />

## 🚀 Live Demo
[**Try the App Here!**](https://hr-employee-attrition-prediction-s.streamlit.app/)  

## 📊 Project Overview
Employee attrition is a major cost for companies. This project analyzes the famous **IBM HR Analytics Employee Attrition & Performance** dataset (1,470 employees) to:
- Identify key factors driving employees to leave
- Build a predictive model to flag at-risk employees
- Provide data-driven recommendations to improve retention

### Key Insights Discovered
- Employees working **overtime** are significantly more likely to leave
- **Long commute distance** (>15 miles) strongly correlates with attrition
- Low **job satisfaction**, **salary hikes**, and **total working years** are top predictors
- Younger employees and those in Sales/Lab roles show higher turnover

## 🛠️ Tech Stack
- **Python**
- **Pandas** & **NumPy** – Data processing
- **Matplotlib** & **Seaborn** – Exploratory Data Analysis
- **Scikit-learn** – Random Forest Classifier (86–88% accuracy)
- **Joblib** – Model persistence
- **Streamlit** – Interactive web dashboard

## 📈 Model Performance
| Metric          | Value     |
|-----------------|-----------|
| Accuracy        | ~84–      |
| Precision       | ~0.46     |
| Recall          | ~0.13     |
| ROC AUC         | ~0.79     |

Top predictors (Feature Importance from Random Forest):
1. OverTime
2. Monthly Income
3. Total Working Years
4. Age
5. Distance From Home

## 🎯 Features of the Web App
- Interactive **EDA** with visualizations (attrition by job role, age, etc.)
- **Predict attrition risk** for any employee using sliders
- Real-time probability output
- Actionable **HR recommendations**


1. Clone the repository:
   ```bash
   git clone https://github.com/mahadev19/hr-employee-attrition-prediction.git
   cd hr-employee-attrition-prediction
   
🌍 Deployment
Deployed on Streamlit Community Cloud (free)
Connected directly to this GitHub repository for automatic updates.
💡 Business Impact
This tool enables HR teams to:

Proactively identify employees at risk of leaving
Target retention strategies (e.g., remote work for long commuters, better hikes)
Reduce turnover costs significantly

