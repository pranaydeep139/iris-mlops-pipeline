import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import json

print("Loading data/iris.csv...")
df = pd.read_csv('data/iris.csv')
df.columns = [col.replace('(cm)', '').strip().replace(' ', '_').lower() for col in df.columns]
print("Data loaded and columns standardized.")

X = df.iloc[:, :-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split into training and testing sets.")

print("Training RandomForestClassifier model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model training complete.")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

os.makedirs('artifacts', exist_ok=True)
model_path = 'artifacts/model.pkl'
joblib.dump(model, model_path)
print(f"Model saved to: {model_path}")

metrics = {"accuracy": accuracy}
metrics_path = 'artifacts/metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=4)
print(f"Metrics saved to: {metrics_path}")
