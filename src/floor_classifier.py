# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report

# class FloorClassifier:
#     def __init__(self, model_type='rf'):
#         if model_type == 'rf':
#             self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#         else:
#             raise ValueError(f"Unsupported model type: {model_type}")

#     def fit(self, X_train, y_train):
#         self.model.fit(X_train, y_train)

#     def predict(self, X):
#         return self.model.predict(X)

#     def evaluate(self, X_test, y_test):
#         preds = self.predict(X_test)
#         acc = accuracy_score(y_test, preds)
#         report = classification_report(y_test, preds, zero_division=0)
#         return acc, report

#     def save(self, filepath='models/floor_model.pkl'):
#         joblib.dump(self.model, filepath)

#     def load(self, filepath='models/floor_model.pkl'):
#         self.model = joblib.load(filepath)










from sklearn.ensemble import RandomForestClassifier

def train_floor_classifier(X_train, y_floor_train):
    """
    Trains Stage 1 Floor Classifier (Random Forest) to distinguish
    Ground Floor vs. First Floor.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_floor_train)
    return model