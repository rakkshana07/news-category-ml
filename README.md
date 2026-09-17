# news-category-ml
News Article Category Classification using Machine Learning and Streamlit
# 📰 News Article Category Classification using Machine Learning

A machine learning project that automatically classifies news articles into five categories using **Natural Language Processing (NLP)** and **Machine Learning**. The trained model is deployed as an interactive **Streamlit web application**.

## 🎯 Project Overview

News websites publish a large number of articles every day. Manually categorizing these articles can be time-consuming.

This project uses machine learning to automatically classify a news article into one of five categories:

* 🏅 Sports
* 🏛️ Politics
* 💼 Business
* 💻 Technology
* 🎬 Entertainment

The application accepts a **news title and article text** and predicts the appropriate category.

## 🚀 Features

* News article text classification
* Text preprocessing and cleaning
* TF-IDF feature extraction
* Comparison of three machine learning algorithms
* Accuracy, Precision, Recall and F1-score evaluation
* Confusion matrix visualization
* Automatic selection of the best-performing model
* Interactive Streamlit web interface
* Prediction confidence display
* Cloud deployment using Streamlit Community Cloud

## 🛠️ Technologies Used

| Technology          | Purpose                        |
| ------------------- | ------------------------------ |
| Python              | Programming language           |
| Pandas              | Data processing                |
| NumPy               | Numerical operations           |
| Scikit-learn        | Machine learning and NLP       |
| TF-IDF              | Text feature extraction        |
| Logistic Regression | Classification                 |
| Decision Tree       | Classification                 |
| Random Forest       | Classification                 |
| Streamlit           | Web application and deployment |
| Git & GitHub        | Version control                |

## 📂 Dataset

The project uses a news article dataset containing **1,490 articles** across five categories.

The original dataset contains the following columns:

* `ArticleId`
* `Text`
* `Category`

### Categories

| Category      | Number of Articles |
| ------------- | -----------------: |
| Sport         |                346 |
| Business      |                336 |
| Politics      |                274 |
| Entertainment |                273 |
| Technology    |                261 |
| **Total**     |          **1,490** |

During preprocessing, duplicate article texts were removed before model training.

## 🔄 Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Missing Value Handling
   ↓
Duplicate Removal
   ↓
Text Preprocessing
   ↓
Train-Test Split
   ↓
TF-IDF Feature Extraction
   ↓
Machine Learning Models
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Model Saving
   ↓
Streamlit Web Application
```

## 🧹 Data Preprocessing

The following preprocessing steps are applied:

1. Remove missing text and category values
2. Remove duplicate articles
3. Convert text to lowercase
4. Remove URLs
5. Remove HTML content
6. Remove unnecessary special characters
7. Normalize whitespace
8. Remove English stop words using TF-IDF

## 🔢 Feature Extraction

**TF-IDF (Term Frequency-Inverse Document Frequency)** is used to convert the news article text into numerical features that machine learning algorithms can understand.

The vectorizer uses:

```text
max_features = 5000
stop_words = "english"
```

## 🤖 Machine Learning Models

Three classification algorithms were trained and evaluated:

### 1. Logistic Regression

Used as a text classification model and achieved the highest performance among the tested models.

### 2. Decision Tree

A tree-based classification algorithm used for comparison.

### 3. Random Forest

An ensemble learning algorithm consisting of multiple decision trees.

## 📊 Model Performance

The models were evaluated using Accuracy, Precision, Recall and F1-score.

| Model                   |   Accuracy |  Precision |     Recall |   F1-score |
| ----------------------- | ---------: | ---------: | ---------: | ---------: |
| **Logistic Regression** | **95.83%** | **95.97%** | **95.83%** | **95.79%** |
| Random Forest           |     94.79% |     95.03% |     94.79% |     94.69% |
| Decision Tree           |     75.00% |     75.00% |     75.00% |     74.77% |

Based on the evaluation results, **Logistic Regression was selected as the final model**.

## 📈 Evaluation Metrics

The project uses:

* **Accuracy** – Measures the overall percentage of correct predictions.
* **Precision** – Measures how many predicted instances of a category are actually correct.
* **Recall** – Measures how many actual instances of a category are correctly identified.
* **F1-score** – Provides a combined measure of precision and recall.
* **Confusion Matrix** – Shows correct and incorrect predictions for each category.

## 🌐 Streamlit Application

The trained model and TF-IDF vectorizer are saved using Python pickle files:

```text
model.pkl
vectorizer.pkl
```

The Streamlit application loads these files and performs predictions on new articles.

### Input

The user enters:

* News Title
* News Article

### Output

The application displays:

* Predicted category
* Prediction confidence

## ▶️ Run the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/rakkshana07/news-category-ml.git
```

### 2. Navigate to the project

```bash
cd news-category-ml
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📁 Project Structure

```text
news-category-ml/
│
├── data/
│   └── News.csv
│
├── app.py
├── train_model.py
│
├── model.pkl
├── vectorizer.pkl
│
├── model_comparison.csv
│
├── logistic_regression_confusion_matrix.png
├── decision_tree_confusion_matrix.png
├── random_forest_confusion_matrix.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

## ☁️ Deployment

The Streamlit application is deployed using **Streamlit Community Cloud**.

### Live Application

**News Category Classification App**

https://news-category-ml.streamlit.app/

## 🔮 Future Enhancements

* Add more news categories
* Use larger and more diverse datasets
* Experiment with advanced NLP techniques
* Add Naive Bayes, SVM and ensemble models
* Improve prediction confidence visualization
* Add multilingual news classification
* Integrate real-time news APIs
* Explore deep learning and transformer-based models

## 👩‍💻 Author

**Rakkshana.K**

B.Tech – Artificial Intelligence and Data Science

GitHub: `rakkshana07`

---

⭐ If you find this project useful, consider giving the repository a star!
