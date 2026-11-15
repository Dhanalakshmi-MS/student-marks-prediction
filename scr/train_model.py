
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
data = pd.read_csv("data/student_scores.csv")

# 2. Separate input and output
X = data[["Hours"]]
y = data["Marks"]

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict marks
predictions = model.predict(X_test)

# 6. Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# 7. Predict marks for a new student
hours = float(input("Enter study hours: "))
predicted_marks = model.predict(pd.DataFrame({"Hours": [hours]}))

print("Predicted Marks:", round(predicted_marks[0], 2))

# 8. Visualize the results
plt.scatter(X, y, label="Actual Marks")
plt.plot(X, model.predict(X), color="red", label="Prediction Line")
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Student Marks Prediction")
plt.legend()
plt.show()
