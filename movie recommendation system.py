import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
file_path = "Movies Dataset.csv"  # Ensure the file is in the correct location
df = pd.read_csv(file_path)

# Trim spaces from column names
df = df.rename(columns=lambda x: x.strip())

# Convert 'Critic_score' into binary categories (high vs. low)
df_cleaned = df.dropna(subset=["Critic_score"])  # Remove missing target values
median_score = df_cleaned["Critic_score"].median()
df_cleaned["Critic_category"] = np.where(df_cleaned["Critic_score"] >= median_score, 1, 0)  # 1 = High, 0 = Low

# Select categorical columns
categorical_cols = ["Genre", "G_C"]

# Apply one-hot encoding
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded_cols = encoder.fit_transform(df_cleaned[categorical_cols])
encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out())

# Drop categorical columns and target column from the original dataset
numeric_df = df_cleaned.drop(columns=categorical_cols + ["Critic_score", "Name", "Critic_category"], errors="ignore")

# Concatenate encoded and numeric features
X_encoded = pd.concat([numeric_df.reset_index(drop=True), encoded_df], axis=1)
X_encoded = X_encoded.fillna(X_encoded.mean())  # Fill missing numeric values with column mean
y = df_cleaned["Critic_category"]

# Train a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)
rf_classifier.fit(X_encoded, y)

# UI Layout
st.set_page_config(page_title="Movie Recommendation System", layout="wide")

# Centered Title
st.markdown("<h1 style='text-align: center;'>Movie Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; font-size: 14px; color: gray;'>by Mowleen Armstrong</h5>", unsafe_allow_html=True)

# Apply Dark Theme and Fix Slider Issue
st.markdown(
    """
    <style>
    body, .stApp { background-color: #0e1117; color: white; }
    .stButton>button { background-color: #222; color: white; border-radius: 8px; }
    .stSelectbox div, .stMarkdown, .stTextInput label { color: white !important; }

    /* Fix slider bar color */
    div[data-testid="stSlider"] > div {
        filter: invert(1) !important;
    }

    /* Fix slider text (label) */
    div[data-testid="stSlider"] p {
        color: white !important;
    }

    /* Hide the number under the slider */
    div[data-testid="stSlider"] div[role="presentation"] {
        display: none !important;
    }

    /* Fix genre dropdown text */
    .css-1wa3eu0-placeholder, .css-qrbaxs, .css-1cpxqw2 {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# User Inputs
genre_choice = st.selectbox("Select a genre:", df_cleaned["Genre"].unique())
rating_choice = st.slider("Select minimum critic score:", int(df_cleaned["Critic_score"].min()), int(df_cleaned["Critic_score"].max()), int(median_score))

# Recommendation Function
def recommend_movie(genre, rating):
    user_input = pd.DataFrame(0, index=[0], columns=X_encoded.columns)
    genre_col = f"Genre_{genre}"
    if genre_col in user_input.columns:
        user_input[genre_col] = 1

    user_input.fillna(X_encoded.mean(), inplace=True)
    prediction = rf_classifier.predict(user_input)
    recommended_movies = df_cleaned[df_cleaned["Critic_category"] == prediction[0]]

    if recommended_movies.empty:
        return "No suitable recommendations found."

    return recommended_movies[["Name", "Genre", "Critic_score"]].sample(min(5, len(recommended_movies)))

# Recommend Button
if st.button("Recommend Movie"):
    recommendations = recommend_movie(genre_choice, rating_choice)
    st.write("### Recommended Movies:")
    st.write(recommendations)

# Footer with Copyright Notice
st.markdown(
    """
    <hr>
    <p style="text-align: center;">© 2025 Movie Recommendation System</p>
    """,
    unsafe_allow_html=True,
)
