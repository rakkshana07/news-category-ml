
import streamlit as st
import pickle
import re


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="News Category Predictor",
    page_icon="📰",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL AND VECTORIZER
# --------------------------------------------------

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)

    with open("vectorizer.pkl", "rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


model, vectorizer = load_model()


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep only letters, numbers and spaces
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# --------------------------------------------------
# CATEGORY MAPPING
# --------------------------------------------------

category_names = {
    "sport": "SPORTS",
    "politics": "POLITICS",
    "business": "BUSINESS",
    "tech": "TECHNOLOGY",
    "entertainment": "ENTERTAINMENT"
}


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("📰 NEWS CATEGORY PREDICTOR")

st.write(
    "Enter a news title and article text to predict its category "
    "using a Machine Learning model."
)

st.divider()


# --------------------------------------------------
# USER INPUT
# --------------------------------------------------

news_title = st.text_input(
    "📝 News Title",
    placeholder="Enter the news headline..."
)

news_article = st.text_area(
    "📄 News Article",
    placeholder="Enter the complete news article here...",
    height=250
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🔮 Predict Category", use_container_width=True):

    if news_title.strip() == "" and news_article.strip() == "":
        st.warning("⚠️ Please enter a news title or article.")

    else:
        # Combine title and article
        combined_text = news_title + " " + news_article

        # Clean text
        cleaned_text = clean_text(combined_text)

        # Convert text into TF-IDF features
        text_vector = vectorizer.transform([cleaned_text])

        # Predict category
        prediction = model.predict(text_vector)[0]

        # Convert category to display name
        display_category = category_names.get(
            prediction,
            prediction.upper()
        )

        st.success("Prediction completed successfully!")

        st.markdown("### 🎯 Predicted Category")

        st.subheader(f"📰 {display_category}")

        # --------------------------------------------------
        # PREDICTION CONFIDENCE
        # --------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(text_vector)[0]

            confidence = max(probabilities) * 100

            st.write(
                f"**Prediction Confidence:** {confidence:.2f}%"
            )

            st.progress(int(confidence))


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("ℹ️ About the Project")

    st.write(
        "This application uses Machine Learning to classify "
        "news articles into five categories."
    )

    st.write("**Categories:**")

    st.write("• Sports")
    st.write("• Politics")
    st.write("• Business")
    st.write("• Technology")
    st.write("• Entertainment")

    st.divider()

    st.write("**Machine Learning Models Tested:**")

    st.write("• Logistic Regression")
    st.write("• Decision Tree")
    st.write("• Random Forest")

    st.divider()

    st.write(
        "🏆 Best Model: Logistic Regression"
    )

    st.write(
        "F1-Score: 95.79%"
    )
