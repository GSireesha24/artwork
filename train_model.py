import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression

# Simple training data
data = pd.DataFrame({
    "views": [100, 200, 300, 400],
    "likes": [10, 20, 30, 40],
    "comments": [5, 10, 15, 20],
    "engagement_score": [50, 60, 70, 80]
})

X = data[["views", "likes", "comments"]]
y = data["engagement_score"]

model = LinearRegression()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))

print("Model saved successfully!")