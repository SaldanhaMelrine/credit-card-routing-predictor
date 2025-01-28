import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load the Dataset
def load_data(file_path):
    df = pd.read_excel(file_path)
    # Drop 'Unnamed' columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

# Preprocess Data
def preprocess_data(df):

    # Handling missing values
    if df.isnull().any().any(): 
        print("Missing values detected. Handling missing values...")
        print(df.isnull().sum())  # Display count of missing values per column
        df['amount'] = df['amount'].fillna(df['amount'].median())
        df['PSP'] = df['PSP'].fillna('Unknown')
        df['success'] = df['success'].fillna(0)
        df.fillna(df.mode().iloc[0], inplace=True)
        print("Missing values handled.")
    else:
        print("No Missing values detected.")
    
    # Ensure the timestamp column is in datetime format
    df['tmsp'] = pd.to_datetime(df['tmsp'])

    # Sort the data by country, amount, and timestamp for logical grouping
    df = df.sort_values(by=['country', 'amount', 'tmsp'])

    # Create the attempt_group column
    df['attempt_group'] = (
        (df['country'] != df['country'].shift(1)) |
        (df['amount'] != df['amount'].shift(1)) |
        ((df['tmsp'] - df['tmsp'].shift(1)).dt.total_seconds() > 60)
    ).cumsum()

    fee_structure = {
        'Moneycard': {'success': 5, 'fail': 2},
        'Goldcard': {'success': 10, 'fail': 5},
        'UK_Card': {'success': 3, 'fail': 1},
        'Simplecard': {'success': 1, 'fail': 0.5}
    }

    # Fee calculation for each transaction based on PSP and success
    df['fee'] = df.apply(
        lambda row: fee_structure[row['PSP']]['success'] if row['success'] == 1
        else fee_structure[row['PSP']]['fail'], axis=1
    )

    # Cumulative fees and attempt numbers calculation
    df['cumulative_fee'] = df.groupby('attempt_group')['fee'].cumsum()  # Cumulative fee
    df['attempt_number'] = df.groupby('attempt_group').cumcount() + 1  # Attempt number within group

    # PSP success rates 
    psp_success_rate = df.groupby('PSP')['success'].mean().rename('psp_success_rate')
    psp_country_success_rate = df.groupby(['PSP', 'country'])['success'].mean().rename('psp_country_success_rate')
    df = df.merge(psp_success_rate, on='PSP', how='left')
    df = df.merge(psp_country_success_rate, on=['PSP', 'country'], how='left')

    # PSP costs
    df['psp_success_cost'] = df['PSP'].map(lambda x: fee_structure[x]['success'])
    df['psp_failure_cost'] = df['PSP'].map(lambda x: fee_structure[x]['fail'])

    # Calculate avg_cumulative_fee per PSP
    df['avg_cumulative_fee'] = df.groupby('PSP')['cumulative_fee'].transform('mean')

    # Calculate avg_psp_reattempts per PSP
    # Reattempts: Failed transaction followed by a successful transaction for the same PSP
    df['reattempt'] = (df['success'] == 0) & (df['success'].shift(-1) == 1)
    df['avg_psp_reattempts'] = df.groupby('PSP')['reattempt'].transform('mean')

    #Create amount bins
    bins = [0, 100, 500, float('inf')]  # Low, Mid, High ranges
    labels = ['Low', 'Mid', 'High']
    df['amount_range'] = pd.cut(df['amount'], bins=bins, labels=labels, right=False)

    # Label Encoding for 'PSP' using sklearn's LabelEncoder
    le_psp = LabelEncoder()
    df['PSP'] = le_psp.fit_transform(df['PSP'])  # Encode PSP column

    # Print the mapping of PSP to codes
    print("PSP Encoding Mapping (Labels -> Codes):")
    for i, name in enumerate(le_psp.classes_):
        print(f"Code {i}: {name}")

    # Label Encoding for all categorical columns
    le = LabelEncoder()
    categorical_columns = ['country', 'card']  
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])

        # Print the mapping of each category to the encoded label
        print(f"\n{column} Encoding Mapping (Labels -> Codes):")
        for i, name in enumerate(le.classes_):
            print(f"Code {i}: {name}")

    # Calculate Pre-Prediction Success Rate
    pre_success_rate = df[df['success'] == 1].shape[0] / df.shape[0]  

    # Calculation for fee Optimization
    pre_model_fees = df['psp_success_cost'].mean()

    return df, le_psp, pre_success_rate, pre_model_fees

# Features and Target
def define_features_and_target(df):
    features = [
        'amount', 'avg_psp_reattempts', 'avg_cumulative_fee', 'cumulative_fee', 'attempt_number',
        'psp_success_rate', 'psp_country_success_rate', 'psp_success_cost', 'psp_failure_cost',
        'PSP','country', 'card','3D_secured'
    ] 

    target = 'success'
    
    return df[features], df[target]

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=50)  # Reduced n_estimators for faster training

    # Set the parameter grid for tuning
    param_grid = {
        'max_depth': [None, 10, 20, 30],   # Depth of trees
        'min_samples_split': [2, 5, 10],   # Minimum samples required to split a node
        'min_samples_leaf': [1, 2, 4],     # Minimum samples required at a leaf node
        'max_features': ['sqrt', 'log2'],  # Features to consider for splitting a node
        'bootstrap': [True, False]  # Whether to use bootstrap sampling
    }

    # Use GridSearchCV to find the best combination of hyperparameters
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    return best_model

def train_baseline_model(X_train, y_train):
    baseline_model = LogisticRegression(random_state=42, max_iter=1000)
    baseline_model.fit(X_train, y_train)
    return baseline_model

# Evaluate the Model
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

# Plot Evaluation Metrics
def plot_roc_curve(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

def plot_precision_recall_curve(model, X_test, y_test):
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)

    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='accuracy')
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Training Score")
    plt.plot(train_sizes, test_mean, label="Validation Score")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

# Recommend PSP
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

def main():
    print("Starting the process...")

    # File path to your dataset
    file_path = r"C:/Users/dell/Desktop/M.Sc. Data Science/Case Study Model Engineering/credit-card-routing-predictor/data/raw/PSP_Jan_Feb_2019.xlsx"

    # Fee structure for PSPs
    fee_structure = {
        'Moneycard': {'success': 5, 'fail': 2},
        'Goldcard': {'success': 10, 'fail': 5},
        'UK_Card': {'success': 3, 'fail': 1},
        'Simplecard': {'success': 1, 'fail': 0.5}
    }

    # Load and preprocess the dataset
    print("Loading data...")
    df = load_data(file_path)
    print(f"Data loaded. Shape: {df.shape}")

    print("Preprocessing data...")
    df, le_psp, pre_success_rate, pre_model_fees = preprocess_data(df)
    print(f"Data preprocessed. Shape: {df.shape}")

    # Define features and target
    print("Defining features and target...")
    X, y = define_features_and_target(df)
    print(f"Features and target defined. Features shape: {X.shape}, Target shape: {y.shape}")

    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Data split complete.")

    # Train the model
    print("Training baseline and advanced model...")
    model = train_model(X_train, y_train)
    baseline_model = train_baseline_model(X_train, y_train)
    print("Model training complete.")

    # Evaluate the model
    print("Evaluating model on training data...")
    train_metrics = evaluate_model(model, X_train, y_train)
    baseline_train_metrics = evaluate_model(baseline_model, X_train, y_train)
    print("Random Forest Training Performance:", train_metrics)
    print("Baseline Training Performance:", baseline_train_metrics)

    print("Evaluating model on testing data...")
    test_metrics = evaluate_model(model, X_test, y_test)
    baseline_test_metrics = evaluate_model(baseline_model, X_test, y_test)
    print("Random Forest Testing Performance:", test_metrics)
    print("Baseline Testing Performance:", baseline_test_metrics)

    # Cross-Validation Scores
    print("Performing cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())

    # Plotting visualizations
    print("Plotting visualizations...")
    print("ROC Curve:")
    plot_roc_curve(model, X_test, y_test)

    print("Precision-Recall Curve:")
    plot_precision_recall_curve(model, X_test, y_test)

    print("Learning Curve:")
    plot_learning_curve(model, X, y)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Failure", "Success"])
    disp.plot(cmap="Blues", values_format='d')
    plt.title("Confusion Matrix")
   
    # Save the plot
    output_path= r"C:/Users/dell/Desktop/M.Sc. Data Science/Case Study Model Engineering/credit-card-routing-predictor/results/confusion_matrix.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Confusion matrix saved to: {output_path}")
    plt.close()

    # Feature Importance Mapping
    feature_importances = model.feature_importances_
    features = X.columns  

    # DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': feature_importances
    })

    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    print("Feature Importance:")
    print(feature_importance_df)


    # Apply the model and recommend PSPs
    print("Generating PSP recommendations...")
    recommendations, post_success_rate, post_model_fees = recommend_psp(X_test, df[df.index.isin(X_test.index)], model, fee_structure, le_psp)

    # Displaying PSP Recommendations
    print("PSP Recommendations:")

    # Recommendation output
    print(recommendations[['amount', 'recommended_psp', 'psp_success_probability', 'psp_cost']])

    #Success Rate metric
    print(f"Pre-Prediction Success Rate: {pre_success_rate:.2f}")
    print(f"Post-Prediction Success Rate : {post_success_rate:.2f}")

    # Transaction fee metric
    print(f"Pre-Prediction Average Transaction Fee: {pre_model_fees:.2f}")
    print(f"Post-Prediction Average Transaction Fee: {post_model_fees:.2f}")

    print("Process complete.")


if __name__ == "__main__":
    main()
