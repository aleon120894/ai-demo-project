import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Читання CSV з правильним енкодінгом
df = pd.read_csv(
    "data/sms_spam/spam.csv",
    encoding="latin-1"
)

# У деяких версіях датасету перші 3 стовпці зайві
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# 🛠 Видаляємо порожні значення і гарантуємо, що все у str
df["message"] = df["message"].astype(str)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Навчання моделі
clf = MultinomialNB()
clf.fit(X_train_vec, y_train)

# Тестування
y_pred = clf.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
