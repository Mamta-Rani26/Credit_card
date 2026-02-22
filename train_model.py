import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

print("Loading dataset...")

# Load dataset
df = pd.read_csv("creditcard.csv")

print("Dataset loaded successfully")

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Saving model...")

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")