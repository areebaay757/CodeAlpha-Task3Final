from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import nltk
import numpy as np

# Load data
df = pd.read_csv('cleaned_ielts_writing_dataset.csv')
print("✅ Dataset loaded successfully!")

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['clean_essay'])

# Target variable
y = df['Overall']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Prediction
y_pred = best_model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared: {r2}")

# Example essay for scoring
example_essay = ["In today's world, technology plays a vital role in almost every aspect of our lives, and education is no exception. With the advancement of the internet, online learning platforms have become widely available, allowing students to access a wealth of information at their fingertips. One of the major benefits of technology in education is that it provides opportunities for personalized learning. Students can learn at their own pace, review materials, and even ask questions through various online forums. This flexibility has proven to be especially beneficial for students who may not be able to attend traditional classrooms due to geographical, financial, or personal constraints. However, while technology has revolutionized education, it also brings some challenges. A major concern is the over-reliance on digital devices, which can lead to distractions and decreased face-to-face interactions between students and teachers. Furthermore, not all students have access to the same level of technology, which can create an educational divide, especially in underdeveloped regions. Despite these drawbacks, technology is undoubtedly a powerful tool that can enhance the learning experience. It is up to educators and policymakers to ensure that the use of technology is balanced and that its benefits are maximized while minimizing its potential disadvantages."]

# Transform the example essay and predict score
example_essay_transformed = vectorizer.transform(example_essay)
predicted_score = best_model.predict(example_essay_transformed)

print(f"Predicted IELTS Score for Example Essay: {predicted_score[0]:.2f}")
