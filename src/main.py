from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from data_loader import load_data
from data_preprocessing import preprocess_data
from model import train_model, train_baseline_model, evaluate_model, recommend_psp
from feature_engineering import define_features_and_target
from data_visualization import plot_roc_curve, plot_precision_recall_curve, plot_learning_curve
import matplotlib.pyplot as plt
import os

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
    df = preprocess_data(df)
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

    # Apply the model and recommend PSPs
    print("Generating PSP recommendations...")
    recommendations = recommend_psp(
        df_features=X_test,
        df_original=df[df.index.isin(X_test.index)],
        model=model,
        fee_structure=fee_structure,
        threshold=0.6
    )

    # Displaying PSP Recommendations
    print("PSP Recommendations:")
    print(recommendations[['amount', 'predicted_psp', 'psp_success_percentage', 'cost']])

    print("Process complete.")


if __name__ == "__main__":
    main()