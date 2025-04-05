# 🎬 Movie Recommendation System (Content-Based Filtering)

This project is a content-based movie recommendation system that suggests movies based on content similarity. It uses Sentence-BERT (SBERT) embeddings to capture semantic meaning from textual data like movie titles, overviews, genres, and credits. The recommendation engine is built using KNN (K-Nearest Neighbors) with cosine similarity as the distance metric.

---

## ❓ Problem Statement

With thousands of movies available online, users often find it difficult to decide what to watch next. This project aims to help users discover similar movies based on a movie they like — without relying on user ratings or behavior.

---

## 🎯 Objectives

- Build a scalable content-based movie recommendation engine.
- Use text embeddings to capture deep semantic similarities between movies.
- Combine textual and numerical features for improved recommendations.
- Deploy the model as an interactive web app.

---

## 🛠 Technology Stack

- **Programming Language**: Python  
- **Data Processing**: PySpark, Pandas  
- **ML Model**: Sentence-BERT (all-MiniLM-L6-v2), KNN  
- **Deployment**: Pickle for model persistence  
- **Dashboard**: Tableau  
- **UI**: Streamlit  

---

## 📁 Dataset

The dataset contains over **700,000 movie entries** with the following key columns:

- `title`, `overview`, `genres`, `original_language`, `credits`  
- `runtime`, `budget`, `revenue`, `vote_average`, `vote_count`, `release_date`  
- `recommendations`, `status`, `keywords`

---

## 🔄 Data Preprocessing

- Removed duplicate records.
- Used TMDB API to fill null values in columns like `genres`, `overview`, and `credits`.
- Replaced missing textual data with default values.
- Kept only movies with `"Released"` status.
- Dropped irrelevant or sparse columns with over 70% null values (`keywords`, `tagline`, `poster_path`, `backdrop_path`, `revenue`, `budget`).
- Combined important textual features (`title`, `genres`, `original_language`, `overview`, `credits`) into a single column `combined_text`.

---

## 🧪 Feature Engineering: `weighted_rating`

After preprocessing, we engineered additional features to enhance recommendation accuracy, including `weighted_rating`, `release_year`, and normalized `runtime`.

---

## 🧠 Model Building

- Text Features: `title`, `genres`, `original_language`, `overview`, `credits`
- Numerical Features: `runtime`, `weighted_rating`, `release_year`
- Applied **Sentence-BERT (SBERT)** to convert `combined_text` into 384-dimensional vectors.
- Normalized numerical features using **MinMaxScaler**.
- Concatenated SBERT embeddings with scaled numerical features to build high-dimensional vectors.
- Used **Cosine Similarity** to measure closeness between movies.
- Applied **KNN** to retrieve the Top-N most similar movies based on cosine scores.
- SBERT Model: `all-MiniLM-L6-v2` (lightweight and efficient)

---

## 🚀 Model Deployment

Stored the model and assets using **Pickle**:
- `recommendation_model.pkl` — Trained KNN model
- `movie_embeddings.pkl` — SBERT + numeric feature vectors
- `movies_df.pkl` — Preprocessed movie dataset

---

## 🌐 Front-End Website (Streamlit UI)

- Built using **Streamlit**
- Users enter a movie name → receive **Top 10** similar movies
- Dynamically fetches **movie posters** from TMDB API
- Provides a fast and interactive experience using the pre-trained model

---

## 📊 Model Evaluation

Due to lack of user rating data, evaluation was done using:
- **Precision@10**: ~70%
- **Recall@10**: ~70%

---

## 🔮 Future Enhancements

- Implement a **hybrid model** combining content and collaborative filtering
- Use **user profiles** for more personalized recommendations
- Improve performance using **Approximate Nearest Neighbors (ANN)**
- Enhance UI with movie **trailers**, **ratings**, and more **meta-info**

---

## 🙌 Acknowledgments

- [Millions of Movies Dataset on Kaggle](https://www.kaggle.com/datasets/akshaypawar7/millions-of-movies)  
- [The Movie Database (TMDB)](https://www.themoviedb.org/) for providing the API used to fetch movie posters and metadata

