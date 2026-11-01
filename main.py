import os
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.data_preprocessing import load_and_preprocess_data
from src.floor_classifier import train_floor_classifier
from src.rp_localizer import train_rp_localizer
from src.visualization import save_visualizations

# -------------------------------------------------------------
# 1. Terminal Printing Function
# -------------------------------------------------------------
def display_terminal_results(y_floor_true, y_floor_pred, y_rp_true, y_rp_pred, rp_labels):
    print("\n" + "="*65)
    print("         PATH 3: STAGE 1 - FLOOR CLASSIFIER EVALUATION")
    print("="*65)
    print(classification_report(y_floor_true, y_floor_pred))
    
    print("Stage 1 Floor Confusion Matrix:")
    floor_labels = ['Ground Floor', 'First Floor']
    cm_floor = confusion_matrix(y_floor_true, y_floor_pred, labels=floor_labels)
    df_floor = pd.DataFrame(
        cm_floor, 
        index=[f"Actual {l}" for l in floor_labels], 
        columns=[f"Pred {l}" for l in floor_labels]
    )
    print(df_floor)
    
    print("\n" + "="*65)
    print("         PATH 3: STAGE 2 - RP LOCALIZER (SESSION 4 TEST SET)")
    print("="*65)
    print(classification_report(y_rp_true, y_rp_pred))
    
    print("Stage 2 RP Confusion Matrix:")
    cm_rp = confusion_matrix(y_rp_true, y_rp_pred, labels=rp_labels)
    df_rp = pd.DataFrame(
        cm_rp, 
        index=[f"Actual {l}" for l in rp_labels], 
        columns=[f"Pred {l}" for l in rp_labels]
    )
    print(df_rp)
    print("="*65 + "\n")

# -------------------------------------------------------------
# 2. Main Execution Flow
# -------------------------------------------------------------
def main():
    print("Loading and preprocessing Wi-Fi fingerprint sessions...")
    # Load feature vectors and labels from preprocessed datasets
    X_train, y_floor_train, y_rp_train, X_test, y_floor_test, y_rp_test, feature_names, unique_rps = load_and_preprocess_data()

    # Create directories for models and results
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Stage 1: Train Floor Classifier
    print("Training Stage 1 (Floor Classifier)...")
    floor_model = train_floor_classifier(X_train, y_floor_train)
    y_floor_pred = floor_model.predict(X_test)
    joblib.dump(floor_model, "models/floor_model.pkl")

    # Stage 2: Train RP Localizer
    print("Training Stage 2 (RP Localizer)...")
    rp_model = train_rp_localizer(X_train, y_rp_train)
    y_rp_pred = rp_model.predict(X_test)
    joblib.dump(rp_model, "models/rp_model.pkl")

    # 1. Print formatted classification reports & confusion matrices in terminal
    display_terminal_results(
        y_floor_true=y_floor_test, 
        y_floor_pred=y_floor_pred, 
        y_rp_true=y_rp_test, 
        y_rp_pred=y_rp_pred, 
        rp_labels=unique_rps
    )

    # 2. Export visualization figures to results/ folder
    save_visualizations(
        y_floor_true=y_floor_test,
        y_floor_pred=y_floor_pred,
        y_rp_true=y_rp_test,
        y_rp_pred=y_rp_pred,
        rp_labels=unique_rps,
        model_rp=rp_model,
        feature_names=feature_names,
        output_dir="results"
    )
    
    print("Artifacts updated in 'models/' and plots saved in 'results/'.")

if __name__ == "__main__":
    main()