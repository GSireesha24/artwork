import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Artwork Platform",
    page_icon="🎨",
    layout="wide"
)

# ---------------------------------------------------
# GENERATE SAMPLE DATA (Replace with real CSV later)
# ---------------------------------------------------
np.random.seed(42)

data = pd.DataFrame({
    "artist_name": np.random.choice(["Aarav", "Diya", "Rohan", "Meera"], 400),
    "artist_experience": np.random.randint(1, 20, 400),
    "art_style": np.random.choice(["Abstract", "Modern", "Classic"], 400),
    "medium": np.random.choice(["Oil", "Acrylic", "Digital"], 400),
    "views": np.random.randint(500, 15000, 400),
    "likes": np.random.randint(50, 3000, 400),
    "comments": np.random.randint(10, 800, 400),
    "shares": np.random.randint(5, 400, 400),
    "sold_flag": np.random.choice([0, 1], 400)
})

# Create engagement score (target)
data["engagement_score"] = (
    0.4 * data["likes"] +
    0.3 * data["comments"] +
    0.2 * data["shares"] +
    0.1 * data["views"] / 100
)

# ---------------------------------------------------
# ML PIPELINE
# ---------------------------------------------------
X = data[[
    "views", "likes", "comments", "shares",
    "artist_experience", "art_style", "medium"
]]

y = data["engagement_score"]

categorical_features = ["art_style", "medium"]
numeric_features = ["views", "likes", "comments", "shares", "artist_experience"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X, y)

# ---------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------
st.sidebar.title("🎨 Artwork ")
page = st.sidebar.radio(
    "Select Dashboard",
    [
        "Artwork & Artist Profile Overview",
        "Artwork Engagement Performance Overview",
        "ML Prediction"
    ]
)

# ===================================================
# 1️⃣ ARTWORK & ARTIST PROFILE OVERVIEW
# ===================================================
if page == "Artwork & Artist Profile Overview":

    st.title("🎨 Artwork & Artist Profile Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Artworks", len(data))
    col2.metric("Total Artists", data["artist_name"].nunique())
    col3.metric("Avg Experience (Years)", round(data["artist_experience"].mean(), 1))
    col4.metric("Sold Rate (%)", round(data["sold_flag"].mean() * 100, 1))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Art Style Distribution")
        fig1 = px.pie(data, names="art_style", hole=0.5)
        fig1.update_layout(template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Medium Usage")

        medium_counts = data["medium"].value_counts().reset_index()
        medium_counts.columns = ["medium", "count"]

        fig2 = px.bar(
            medium_counts,
            x="medium",
            y="count",
            color="medium"
        )
        fig2.update_layout(template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    st.subheader("Experience vs Engagement")
    fig3 = px.scatter(
        data,
        x="artist_experience",
        y="engagement_score",
        color="art_style",
        size="views"
    )
    fig3.update_layout(template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

# ===================================================
# 2️⃣ ENGAGEMENT PERFORMANCE OVERVIEW
# ===================================================
if page == "Artwork Engagement Performance Overview":

    st.title("🔥 Artwork Engagement Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg Views", round(data["views"].mean(), 0))
    col2.metric("Avg Likes", round(data["likes"].mean(), 0))
    col3.metric("Avg Comments", round(data["comments"].mean(), 0))
    col4.metric("Avg Engagement Score", round(data["engagement_score"].mean(), 1))

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Engagement Score Distribution")
        fig4 = px.histogram(data, x="engagement_score", nbins=30)
        fig4.update_layout(template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)

    with col2:
        st.subheader("Views vs Likes")
        fig5 = px.scatter(
            data,
            x="views",
            y="likes",
            color="engagement_score",
            size="comments"
        )
        fig5.update_layout(template="plotly_dark")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    st.subheader("Top Performing Art Styles")
    top_styles = data.groupby("art_style")["engagement_score"].mean().reset_index()

    fig6 = px.bar(
        top_styles,
        x="art_style",
        y="engagement_score",
        color="art_style"
    )
    fig6.update_layout(template="plotly_dark")
    st.plotly_chart(fig6, use_container_width=True)

# ===================================================
# 3️⃣ ML PREDICTION
# ===================================================
if page == "ML Prediction":

    st.title("🤖 Predict Artwork Popularity")

    col1, col2 = st.columns(2)

    with col1:
        views = st.number_input("Views", min_value=0)
        likes = st.number_input("Likes", min_value=0)
        comments = st.number_input("Comments", min_value=0)
        shares = st.number_input("Shares", min_value=0)

    with col2:
        experience = st.number_input("Artist Experience (Years)", min_value=0)
        art_style = st.selectbox("Art Style", data["art_style"].unique())
        medium = st.selectbox("Medium", data["medium"].unique())

    if st.button("🚀 Predict Popularity"):

        input_df = pd.DataFrame({
            "views": [views],
            "likes": [likes],
            "comments": [comments],
            "shares": [shares],
            "artist_experience": [experience],
            "art_style": [art_style],
            "medium": [medium]
        })

        prediction = model.predict(input_df)[0]

        st.success(f"🎯 Predicted Engagement Score: {round(prediction, 2)}")

        fig7 = px.pie(
            values=[prediction, max(0, 10000 - prediction)],
            names=["Predicted Score", ""],
            hole=0.7
        )
        fig7.update_layout(showlegend=False, template="plotly_dark")
        st.plotly_chart(fig7, use_container_width=True)