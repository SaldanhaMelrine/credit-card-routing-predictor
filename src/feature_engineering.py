# Features and Target
def define_features_and_target(df):

    output_path= r"C:/Users/dell/Desktop/M.Sc. Data Science/Case Study Model Engineering/credit-card-routing-predictor/data/processed/processed_data.xlsx"

    features = [
        'amount', 'avg_psp_reattempts', 'avg_cumulative_fee', 'cumulative_fee', 'attempt_number',
        'psp_success_rate', 'psp_country_success_rate', 'psp_success_cost', 'psp_failure_cost',
        'PSP','country', 'card','3D_secured'
    ] 

    target = 'success'

    df.to_excel(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

    return df[features], df[target]

