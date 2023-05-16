import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def duration_to_seconds(duration_str):
    duration = pd.to_timedelta(duration_str)
    return duration.total_seconds()

def preprocess_data(X):
    X = X.copy()
    
    # Convert date columns to numerical features
    X['deadline_weekday'] = X['deadline'].dt.strftime('%w').astype(int)
    X['deadline_month'] = X['deadline'].dt.month
    X['deadline_day'] = X['deadline'].dt.day
    X['deadline_year'] = X['deadline'].dt.year
    X['deadline_hour'] = X['deadline'].dt.hour

    # Convert boolean columns to int
    bool_columns = ['staff_pick', 'spotlight']
    X[bool_columns] = X[bool_columns].astype(int)

    # Get dummy variables for categorical columns
    categorical_columns = ['state_changed_at_weekday', 'created_at_weekday', 'launched_at_weekday']
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Convert datetime columns to duration in seconds
    datetime_columns = ['create_to_launch', 'launch_to_deadline', 'launch_to_state_change']
    for col in datetime_columns:
        X[f"{col}_seconds"] = X[col].apply(duration_to_seconds)

    # Drop the original datetime columns
    X = X.drop(datetime_columns + ['deadline'], axis=1)

    return X

# Load the dataset
data = pd.read_csv("kickstarter_data_full.csv", low_memory=False)

columns_to_drop = ['id', 'photo', 'name', 'blurb', 'slug', 'disable_communication', 'currency_symbol',
                   'currency_trailing_code', 'state_changed_at', 'created_at', 'launched_at', 'creator',
                   'location', 'profile', 'urls', 'source_url', 'friends', 'is_starred', 'is_backing', 'permissions']

data_cleaned = data.drop(columns=columns_to_drop)

date_columns = ['deadline']

#Convert deadline column from object data type to datetime64
for col in date_columns:
    data_cleaned[col] = pd.to_datetime(data_cleaned[col])

# Filling missing values
data_cleaned['category'].fillna('Unknown', inplace=True)
mean_columns = ['name_len', 'name_len_clean', 'blurb_len', 'blurb_len_clean']

#Filling missing spaces with mean value
for col in mean_columns:
    data_cleaned[col].fillna(data_cleaned[col].mean(), inplace=True)

# One Hot Encoding
data_cleaned = pd.get_dummies(data_cleaned, columns=['country', 'category', 'currency'], drop_first=True)

# Now target variable is 'SuccessfulBool' instead of 'state'
# Now target variable is 'SuccessfulBool' instead of 'state'
X = data_cleaned.drop(['SuccessfulBool', 'state'], axis=1)
y = data_cleaned['SuccessfulBool']


le = LabelEncoder()
y = le.fit_transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess the training and test sets
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# SMOTE for handling class imbalance
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Training XGBClassifier
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_classifier.fit(X_train_resampled, y_train_resampled)

# Predict on the training and test sets
y_pred_train = xgb_classifier.predict(X_train_resampled)
y_pred_test = xgb_classifier.predict(X_test)

# Calculate accuracy for each set
accuracy_train = accuracy_score(y_train_resampled, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

# Print classification reports for both sets
print("Training Classification Report:")
print(classification_report(y_train_resampled, y_pred_train))
print("Training Accuracy:", accuracy_train)

print("\nTest Classification Report:")
print(classification_report(y_test, y_pred_test))
print("Test Accuracy:", accuracy_test)

# Train different classifiers and evaluate
classifiers = [
    ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("Support Vector Classifier", SVC(random_state=42)),
    ("Decision Tree", DecisionTreeClassifier(random_state=42))
]

for name, classifier in classifiers:
    # Train the model
    classifier.fit(X_train_resampled, y_train_resampled)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy:", accuracy)

    # Calculate classification report
    report = classification_report(y_test, y_pred)
    print(f"\n{name} Test Classification Report:")
    print(report)
