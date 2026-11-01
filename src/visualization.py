import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def save_visualizations(y_floor_true, y_floor_pred, y_rp_true, y_rp_pred, rp_labels, model_rp, feature_names, output_dir="results"):
    """
    Generates and saves performance charts for the two-stage model.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Stage 1: Floor Confusion Matrix
    plt.figure(figsize=(5, 4))
    cm_floor = confusion_matrix(y_floor_true, y_floor_pred, labels=['Ground Floor', 'First Floor'])
    sns.heatmap(cm_floor, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ground Floor', 'First Floor'], 
                yticklabels=['Ground Floor', 'First Floor'])
    plt.title('Stage 1: Floor Classifier Confusion Matrix')
    plt.xlabel('Predicted Floor')
    plt.ylabel('Actual Floor')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "floor_confusion_matrix.png"), dpi=300)
    plt.close()

    # 2. Stage 2: RP Localizer Confusion Matrix
    plt.figure(figsize=(9, 7))
    cm_rp = confusion_matrix(y_rp_true, y_rp_pred, labels=rp_labels)
    sns.heatmap(cm_rp, annot=True, fmt='d', cmap='Greens', xticklabels=rp_labels, yticklabels=rp_labels)
    plt.title('Stage 2: RP Localizer Confusion Matrix (Session 4 Test Set)')
    plt.xlabel('Predicted RP')
    plt.ylabel('Actual RP')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rp_confusion_matrix.png"), dpi=300)
    plt.close()

    # 3. Top Access Point Feature Importances
    if hasattr(model_rp, 'feature_importances_'):
        importances = model_rp.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # Top 10 APs
        
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette='viridis')
        plt.title('Top 10 Infrastructure APs by Feature Importance')
        plt.xlabel('Relative Importance')
        plt.ylabel('BSSID / AP')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ap_feature_importance.png"), dpi=300)
        plt.close()

    print(f"Visualizations saved to '{output_dir}/'")