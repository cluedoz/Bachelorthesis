import pandas as pd
from sklearn.metrics import precision_score, recall_score

# Read data from CSV file
input_file = '/scratch/meditron/FastChat/Skripts Sean/data/predicted_icd_codes.csv'
output_file = '/scratch/meditron/FastChat/Skripts Sean/data/results_codiESP.csv'

# Load the data into a DataFrame
df = pd.read_csv(input_file)

# Function to calculate precision and recall for each row
def calculate_metrics(row):
    if pd.isna(row['Correct ICD Codes']) or pd.isna(row['Generated ICD Codes']):
        return pd.Series({'Precision': 0, 'Recall': 0, 'TP': 0, 'FP': 0, 'FN': 0})
    
    correct_codes = set(row['Correct ICD Codes'].lower().split(', '))
    generated_codes = set(row['Generated ICD Codes'].lower().split(', '))
    
    tp = len(correct_codes & generated_codes)
    fp = len(generated_codes - correct_codes)
    fn = len(correct_codes - generated_codes)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return pd.Series({'Precision': precision, 'Recall': recall, 'TP': tp, 'FP': fp, 'FN': fn})

# Apply the function to each row
metrics = df.apply(calculate_metrics, axis=1)
df = pd.concat([df, metrics], axis=1)

# Calculate micro-averaged precision and recall
total_tp = metrics['TP'].sum()
total_fp = metrics['FP'].sum()
total_fn = metrics['FN'].sum()

micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

# Calculate macro-averaged precision and recall
macro_precision = metrics['Precision'].mean()
macro_recall = metrics['Recall'].mean()

# Print debug information
print(f"Micro Precision: {micro_precision}, Micro Recall: {micro_recall}")
print(f"Macro Precision: {macro_precision}, Macro Recall: {macro_recall}")

# Create a DataFrame for micro and macro averages
summary_df = pd.DataFrame({
    'ID': ['Micro', 'Macro'],
    'Precision': [micro_precision, macro_precision],
    'Recall': [micro_recall, macro_recall],
    'TP': [total_tp, None],
    'FP': [total_fp, None],
    'FN': [total_fn, None]
})

# Concatenate the original DataFrame with the summary DataFrame
result_df = pd.concat([df, summary_df], ignore_index=True)

# Save the results to a new CSV file
result_df.to_csv(output_file, index=False)

print(f'Results saved to {output_file}')
