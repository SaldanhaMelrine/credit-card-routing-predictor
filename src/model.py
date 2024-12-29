from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os

def train_model(X_train, y_train):
    output_path=r"C:/Users/dell/Desktop/M.Sc. Data Science/Case Study Model Engineering/credit-card-routing-predictor/models/model.pkl"
    model = RandomForestClassifier(random_state=42, n_estimators=100)
    model.fit(X_train, y_train)
    joblib.dump(model, output_path)
    print(f"Advanced Model saved to: {output_path}")
    return model

def train_baseline_model(X_train, y_train):
    output_path=r"C:/Users/dell/Desktop/M.Sc. Data Science/Case Study Model Engineering/credit-card-routing-predictor/models/baseline_model.pkl"
    baseline_model = LogisticRegression(random_state=42, max_iter=1000)
    baseline_model.fit(X_train, y_train)
    joblib.dump(baseline_model, output_path)
    print(f"Baseline Model saved to: {output_path}")
    return baseline_model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)[:, 1]

    metrics = {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1 Score': f1_score(y, y_pred),
        'ROC AUC': roc_auc_score(y, y_pred_prob)
    }
    return metrics

def recommend_psp(df_features, df_original, model, fee_structure, threshold=0.6):
    df_original = df_original.copy()  # Avoid modifying the original DataFrame

    # Add predicted success probabilities
    df_original['predicted_success_probability'] = model.predict_proba(df_features)[:, 1]

    # Compute efficiency scores and apply threshold
    df_original['efficiency_score'] = df_original.apply(
        lambda row: row['predicted_success_probability'] / (fee_structure[row['PSP']]['success'] + 1e-6)
        if row['predicted_success_probability'] >= threshold else None,
        axis=1
    )

    # Rank PSPs by efficiency score
    df_original['psp_rank'] = df_original.groupby(df_original.index)['efficiency_score'].rank(ascending=False)
    df_original['recommended_psp'] = df_original['psp_rank'] == 1

    # Filter out rows with NaN efficiency scores
    df_original = df_original[df_original['efficiency_score'].notna()]

    # Add success rate and cost for the recommended PSP
    df_recommendation = df_original[df_original['recommended_psp']].copy()
    df_recommendation['predicted_psp'] = df_recommendation['PSP']
    df_recommendation['psp_success_percentage'] = df_recommendation['predicted_success_probability'] * 100
    df_recommendation['cost'] = df_recommendation.apply(
        lambda row: fee_structure[row['PSP']]['success'] if row['predicted_success_probability'] >= threshold
        else fee_structure[row['PSP']]['fail'],
        axis=1
    )

    return df_recommendation

