import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


# Email dataset
data = {
    "email": [
        "Your meeting is scheduled for tomorrow",
        "Your Amazon order has been shipped",
        "Your college assignment is due tomorrow",
        "Your bank statement is ready",
        "Thank you for your purchase",
        "Your electricity bill is available",

        "URGENT click here to verify your account",
        "You won a prize click here to claim",
        "Your account is suspended verify your password",
        "Click this link to win a free gift",
        "Urgent security alert login immediately",
        "Your bank account will be closed verify now"
    ],

    "label": [
        "Safe",
        "Safe",
        "Safe",
        "Safe",
        "Safe",
        "Safe",

        "Phishing",
        "Phishing",
        "Phishing",
        "Phishing",
        "Phishing",
        "Phishing"
    ]
}


# Create dataframe
df = pd.DataFrame(data)


# Separate emails and labels
X = df["email"]
y = df["label"]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# Convert text into numbers
vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# Create and train the model
model = LogisticRegression()

model.fit(X_train, y_train)


# Make predictions
predictions = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)

print("Phishing Email Detection Model")
print("--------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")


# Confusion matrix
cm = confusion_matrix(
    y_test,
    predictions,
    labels=["Safe", "Phishing"]
)

print("\nConfusion Matrix:")
print(cm)


# Display confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Safe", "Phishing"]
)

display.plot()
plt.title("Phishing Email Detection")
plt.show()


# Test a new email
email = input("\nEnter an email to check: ")

email_vector = vectorizer.transform([email])

result = model.predict(email_vector)[0]

print("\nResult:", result)

if result == "Phishing":
    print("Warning: This email may be a phishing email!")
else:
    print("This email appears to be safe.")