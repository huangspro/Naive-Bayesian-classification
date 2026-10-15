
# Import required libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Import classification models

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier
)
from sklearn.naive_bayes import GaussianNB

# 01. Load and preprocess dataset

df = pd.read_csv("data.csv")

# Drop irrelevant columns to get feature matrix X and target y

drop_cols = ["id", "partition", "in_train_1000", "Class"]
X = df.drop(columns=drop_cols)
y = df["Class"]

# 02. Split data into train and test sets with stratification

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,  # 020% of data for testing
    random_state=42,  # Fixed random seed for reproducibility
    stratify=y  # Maintain class distribution in train/test sets
)

# 03. Initialize all classification models to compare

model_dict = {
    "LR": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier,
    "SVM": SVC,
    "DT": DecisionTreeClassifier(random_state=42),
    "RF": RandomForestClassifier(random_state=42),
    "ET": ExtraTreesClassifier(random_state=42),
    "GB": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "NB": GaussianNB
}

# 04. Train models and evaluate test set accuracy

test_scores = {}
for model_name, model in model_dict.items():

    # Tree-based models don't need standardization

    if model_name in ["DT", "RF", "ET", "GB", "AdaBoost"]:
        clf_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),  # Fill missing values with median
            model
        )
    else:

        # Distance-based, linear, and probability models need standardization

        clf_pipeline = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler,  # Feature standardization
            model
        )
    clf_pipeline.fit(X_train, y_train)
    test_scores[model_name] = clf_pipeline.score(X_test, y_test)

# 05. Display results sorted by accuracy

print("Model Test Accuracy Rankings:")
print(pd.Series(test_scores).sort_values(ascending=False).round(4))