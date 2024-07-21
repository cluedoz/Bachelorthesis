import pandas as pd

# Pfade zu den CSV-Dateien
reference_file = '/scratch/meditron/FastChat/Skripts Sean/Auswertungen/reference_icd_codes.csv'
input_file = '/scratch/meditron/FastChat/Skripts Sean/predicted_icd_codes.csv'
output_file = '/scratch/meditron/FastChat/Skripts Sean/Auswertungen/results_icd_code_validation.csv'

# Laden der Daten in DataFrames
ref_df = pd.read_csv(reference_file)
df = pd.read_csv(input_file)

# Funktion zum Bereinigen der ICD-Codes
def clean_code(code):
    return code.replace('.', '').strip().upper()

# Bereinigen der Referenzcodes
reference_codes = ref_df['Code'].apply(clean_code)

# Bereinigen der generierten ICD-Codes und Überprüfung
def count_valid_generated_codes(row, ref_codes):
    if pd.isna(row['Generated ICD Codes']):
        return 0, []
    generated_codes = row['Generated ICD Codes'].split(', ')
    cleaned_codes = [clean_code(code) for code in generated_codes]
    valid_codes = [code for code in cleaned_codes if code in ref_codes.values]
    return len(valid_codes), valid_codes

# Funktion zur Anwendung auf jede Zeile des DataFrames
def apply_count_valid_generated_codes(row, ref_codes):
    count, valid_codes = count_valid_generated_codes(row, ref_codes)
    return pd.Series([count, valid_codes])

# Anwendung der Funktion auf jede Zeile
df[['Valid Generated ICD Codes Count', 'Valid Generated ICD Codes List']] = df.apply(lambda row: apply_count_valid_generated_codes(row, reference_codes), axis=1)

# Gesamtanzahl der gültigen generierten ICD-Codes berechnen
total_valid_generated_icd_codes = df['Valid Generated ICD Codes Count'].sum()

# Anzahl der eindeutigen gültigen generierten ICD-Codes berechnen
unique_valid_generated_icd_codes = pd.Series([code for sublist in df['Valid Generated ICD Codes List'] for code in sublist]).nunique()

# Ergebnisse ausgeben
print(f"Total Valid Generated ICD Codes: {total_valid_generated_icd_codes}")
print(f"Unique Valid Generated ICD Codes: {unique_valid_generated_icd_codes}")

# Optional: Ergebnisse in eine neue CSV-Datei speichern
df.to_csv(output_file, index=False)

print(f'Results saved to {output_file}')
