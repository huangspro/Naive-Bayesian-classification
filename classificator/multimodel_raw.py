import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("../data/bean_raw.csv")
drop_cols = ["id", "partition", "in_train_1000", "Class"]
X, y = df.drop(columns=drop_cols), df["Class"]

train_mask = df["partition"] == "TRAIN"
test_mask = df["partition"] == "TEST"
X_train, y_train = X[train_mask], y[train_mask]
X_test, y_test = X[test_mask], y[test_mask]

models = {
    "LR": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(),
    "DT": DecisionTreeClassifier(random_state=42),
    "RF": RandomForestClassifier(random_state=42),
    "ET": ExtraTreesClassifier(random_state=42),
    "GB": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "NB": GaussianNB()
}

scores = {}
for name, model in models.items():
    print(name)
    steps = [SimpleImputer(strategy="median")]
    if name not in ["DT", "RF", "ET", "GB", "AdaBoost"]:
        steps.append(StandardScaler())
    clf = make_pipeline(*steps, model)
    clf.fit(X_train, y_train)
    scores[name] = clf.score(X_test, y_test)

print(pd.Series(scores).sort_values(ascending=False).round(4))
