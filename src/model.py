from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib
import os

def train_model(X_train, y_train):

    output_path=r"C:/Users/dell/Desktop/M.Sc. Data Science/Case Study Model Engineering/credit-card-routing-predictor/models/model.pkl"

    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=50)  # Reduced n_estimators for faster training

    # parameter grid for tuning
    param_grid = {
        'max_depth': [None, 10, 20, 30],  
        'min_samples_split': [2, 5, 10],   
        'min_samples_leaf': [1, 2, 4],     
        'max_features': ['sqrt', 'log2'],  
        'bootstrap': [True, False]  
    }

    # GridSearchCV to find the best combination of hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    joblib.dump(model, output_path)
    print(f"Advanced Model saved to: {output_path}")

    return best_model


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


def recommend_psp(df_features, df_original, model, fee_structure, le_psp, performance_gap=0.05):
    df_original = df_original.copy()  

    encoded_fee_structure = {}
    for i, psp_name in enumerate(le_psp.classes_):
        success_fee = fee_structure[psp_name]['success']
        encoded_fee_structure[i] = success_fee

    # Initialize lists to store the results for each transaction
    recommended_psps = []
    psp_success_probabilities = []
    psp_costs = []

    # Looping through each transaction in the dataset
    for idx, row in df_original.iterrows():
        # Features for current transaction
        transaction_features = df_features.loc[[idx]]  # Keep the row as DataFrame

        # Initializing a list to store the predicted success probabilities for each PSP for this transaction
        success_probabilities_for_transaction = {}

        post_success_count = 0 # For post_success_rate calculation

        # Loop through each PSP and predict the success probability for this transaction
        for psp in df_original['PSP'].unique():
            psp_avg_cumulative_fee = row['avg_cumulative_fee']
            psp_avg_reattempts = row['avg_psp_reattempts']
            psp_success_rate = row['psp_success_rate']
            psp_country_success_rate = row['psp_country_success_rate']
            
            psp_success_cost = encoded_fee_structure[psp]

            # Adding static features to the transaction features for the current PSP
            transaction_with_static_features = transaction_features.copy()
            transaction_with_static_features['avg_cumulative_fee'] = psp_avg_cumulative_fee
            transaction_with_static_features['avg_psp_reattempts'] = psp_avg_reattempts
            transaction_with_static_features['psp_success_rate'] = psp_success_rate
            transaction_with_static_features['psp_country_success_rate'] = psp_country_success_rate
            transaction_with_static_features['PSP'] = psp
            transaction_with_static_features['cumulative_fee'] = psp_success_cost
            transaction_with_static_features['attempt_number'] = 1

            # Predicting the success probability for this transaction and PSP
            predicted_probabilities = model.predict_proba(transaction_with_static_features)
            predicted_success_probability = predicted_probabilities[0, 1]

            success_probabilities_for_transaction[psp] = predicted_success_probability

        # Primary Selection
        best_psp = max(success_probabilities_for_transaction, key=success_probabilities_for_transaction.get)
        best_success_probability = success_probabilities_for_transaction[best_psp]

        # Secondary Selection for Cost Optimization
        best_cost = encoded_fee_structure[best_psp]
        for psp, success_prob in success_probabilities_for_transaction.items():
            if success_prob == best_success_probability:
                psp_cost = encoded_fee_structure[psp]
                if psp_cost < best_cost:
                    best_cost = psp_cost
                    best_psp = psp

        # Append the results for this transaction
        recommended_psps.append(le_psp.classes_[best_psp])
        psp_success_probabilities.append(best_success_probability)
        psp_costs.append(best_cost)

    # recommended PSP, success probability, and cost are added to the DataFrame
    df_original['recommended_psp'] = recommended_psps
    df_original['psp_success_probability'] = psp_success_probabilities
    df_original['psp_cost'] = psp_costs

    # Success Metrics Calculation
    post_success_count = df_original[df_original['psp_success_probability'] > 0.86].shape[0] #Threshold setting
    post_success_rate = post_success_count / df_original.shape[0]
    post_model_fees = df_original['psp_cost'].mean()

    # Return the DataFrame with the recommended PSP for each transaction
    df_recommendation = df_original[['amount', 'recommended_psp', 'psp_success_probability', 'psp_cost']]
    
    return df_recommendation, post_success_rate, post_model_fees
