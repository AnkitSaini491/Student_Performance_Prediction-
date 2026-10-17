
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Sample student dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8, 2, 5, 9, 3],
    "attendance": [50, 55, 60, 65, 70, 75, 80, 90, 58, 72, 95, 62],
    "previous_marks": [35, 40, 45, 50, 55, 60, 65, 75, 42, 58, 85, 48],
    "final_marks": [30, 38, 45, 48, 58, 65, 70, 85, 40, 60, 92, 46]
}

df = pd.DataFrame(data)

X = df[["study_hours", "attendance", "previous_marks"]]
y = df["final_marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

joblib.dump(model, "student_model.pkl")

print("Model trained successfully!")
print("Mean Absolute Error:", round(mae, 2))
