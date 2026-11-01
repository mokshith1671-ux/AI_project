# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import accuracy_score, classification_report

# class RPLocalizer:
#     def __init__(self, model_type='rf'):
#         self.model_type = model_type
#         if model_type == 'rf':
#             self.model = RandomForestClassifier(n_estimators=100, random_state=42)
#         elif model_type == 'knn':
#             self.model = KNeighborsClassifier(n_neighbors=3, weights='distance')
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

#     def save(self, filepath='models/rp_model.pkl'):
#         joblib.dump(self.model, filepath)

#     def load(self, filepath='models/rp_model.pkl'):
#         self.model = joblib.load(filepath)










from sklearn.ensemble import RandomForestClassifier

def train_rp_localizer(X_train, y_rp_train):
    """
    Trains Stage 2 RP Localizer (Random Forest) to predict specific
    Reference Point nodes within a floor.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_rp_train)
    return model