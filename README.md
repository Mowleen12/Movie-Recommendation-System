# 🎬 Movie Recommendation System  

### *By Mowleen Armstrong*  

## 📌 Project Overview  
The **Movie Recommendation System** is a machine learning-based application that suggests movies based on user preferences. Built with **Streamlit**, it allows users to choose a genre and set a minimum critic score for personalized recommendations. The system uses a **Random Forest Classifier** for accurate predictions.  

## 🛠 Features  
✅ **User-Created Dataset**: Built using Microsoft Excel.  
✅ **Genre-Based Recommendations**: Users select a genre, and the system suggests relevant movies.  
✅ **Critic Score Filtering**: Users can set a minimum critic score to refine recommendations.  
✅ **Machine Learning Model**: Uses a **Random Forest Classifier** and **OneHotEncoder**.  
✅ **Dark Mode UI**: A sleek, modern dark theme for a better user experience.  

## 🔧 Tech Stack  
- **Programming Language**: Python  
- **Libraries Used**:  
  - `pandas`, `numpy` for data processing  
  - `scikit-learn` for machine learning  
  - `streamlit` for UI  

## 📂 Dataset Information  
The dataset used is **Movies Dataset.csv**, which includes:  
- `Name`: Movie title  
- `Genre`: Movie genre  
- `Critic_score`: Movie rating  
- `G_C`: Additional category data  

## 🚀 How to Run  
1. Install dependencies:  
   ```bash
   pip install pandas numpy streamlit scikit-learn
2. To run the application:
   streamlit run movie_recommendation_system.py
