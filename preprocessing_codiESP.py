import os
import pandas as pd
 
# Define the folder path and file path
text_files_folder = '/scratch/meditron/FastChat/Skripts Sean/data/final_dataset_v4_to_publish/test/text_files_en'
icd_codes_file = '/scratch/meditron/FastChat/Skripts Sean/data/final_dataset_v4_to_publish/test/testD.tsv'  # Replace with your actual file path
output_csv_file = '/scratch/meditron/FastChat/Skripts Sean/data/preprocessed_CodiESP_file.csv'
 
# Read the ICD codes into a DataFrame
icd_codes_df = pd.read_csv(icd_codes_file, sep='\t', header=None, names=['ID', 'ICD_Code'])

# Create a dictionary to store the clinical texts
clinical_texts = {}

# Read the clinical text files
for filename in os.listdir(text_files_folder):
    if filename.endswith('.txt'):
        file_id = filename.split('.')[0]
        with open(os.path.join(text_files_folder, filename), 'r') as file:
            clinical_texts[file_id] = file.read()

# Function to count words in a text
def count_words(text):
    return len(text.split())

# Combine the clinical texts and ICD codes into a single DataFrame
combined_data = []

for file_id, text in clinical_texts.items():
    icd_codes = icd_codes_df[icd_codes_df['ID'] == file_id]['ICD_Code'].tolist()
    word_count = count_words(text)
    combined_data.append([file_id, text, ', '.join(icd_codes), word_count])

# Convert the combined data into a DataFrame
combined_df = pd.DataFrame(combined_data, columns=['ID', 'Case Note', 'ICD Codes', 'Length'])

# Save the DataFrame to a CSV file
combined_df.to_csv(output_csv_file, index=False)

print(f"Combined clinical notes and ICD codes have been saved to {output_csv_file}")