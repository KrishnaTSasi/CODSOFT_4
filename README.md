# CODSOFT_4

TASK 4 - Spam SMS Detection

 📌 Objective
Build an AI model to classify SMS messages as **spam** or **ham** using TF-IDF vectorization and ML classifiers.

 🧠 Models Used
- Naive Bayes: **Accuracy 96.68%**
- Logistic Regression: **Accuracy 95.25%**
- Support Vector Machine (SVM): **Accuracy 97.67%** ✅ Best

 📊 Visualization
Included class distribution histogram to highlight dataset imbalance.

 📁 Files
- `spam_sms_detection.ipynb`: Full project notebook
- `spam.csv`: SMS Spam Collection dataset (from UCI)
- `requirements.txt`: Python dependencies
- `assets/`: Optional images used in the notebook

 📦 Installation
```bash
pip install -r requirements.txt

**Code**

Here are the **complete steps** for your **TASK 4: Spam SMS Detection** project — perfect for your report, README, or LinkedIn post.

---

## ✅ TASK 4: Spam SMS Detection

**Objective**: Build a machine learning model that classifies SMS messages as **spam** or **ham (legit)** using TF-IDF and classifiers like Naive Bayes, Logistic Regression, and SVM.

---

### 🔧 Steps / Procedure:

#### 1. **Import Required Libraries**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
```

---

#### 2. **Load and Clean the Dataset**

* Load `spam.csv` from UCI.
* Remove unnecessary columns like `Unnamed: 2`, `Unnamed: 3`, `Unnamed: 4`.
* Rename columns for clarity.

```python
df = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
```

---

#### 3. **Visualize Class Distribution**

Plot histogram (bar plot) to see the number of spam vs. ham messages.

```python
sns.countplot(x='label', data=df, palette='pastel')
plt.xticks([0, 1], ['Ham', 'Spam'])
plt.title('Distribution of Ham and Spam Messages')
plt.xlabel('Message Type')
plt.ylabel('Count')
plt.show()
```

---

#### 4. **Split the Data**

Split into training and testing sets (80/20 split).

```python
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
```

---

#### 5. **Text Vectorization using TF-IDF**

Convert text into numerical format using TF-IDF.

```python
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

---

#### 6. **Train and Evaluate Models**

##### ➤ Naive Bayes

```python
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
print("Naive Bayes Accuracy:", accuracy_score(y_test, nb.predict(X_test_tfidf)))
```

##### ➤ Logistic Regression

```python
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr.predict(X_test_tfidf)))
```

##### ➤ Support Vector Machine

```python
svm = SVC()
svm.fit(X_train_tfidf, y_train)
print("SVM Accuracy:", accuracy_score(y_test, svm.predict(X_test_tfidf)))
```

---

### ✅ Final Accuracies:

| Model               | Accuracy          |
| ------------------- | ----------------- |
| Naive Bayes         | 96.68%            |
| Logistic Regression | 95.25%            |
| SVM                 | **97.67%** ✅ Best |



