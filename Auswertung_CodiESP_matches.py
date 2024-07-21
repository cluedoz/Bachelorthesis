import pandas as pd
import os

# Daten einlesen und überprüfen
input_file_path = '/scratch/meditron/FastChat/Skripts Sean/data/predicted_icd_codes.csv'
df = pd.read_csv(input_file_path)

def calculate_matches(correct_codes, generated_codes):
    if pd.isna(correct_codes) or pd.isna(generated_codes):
        return 0, 0, 0, 0
    
    # Funktion zum Entfernen von Punkten und Umwandeln in Kleinbuchstaben
    def clean_code(code):
        return code.replace('.', '').strip().lower()
    
    correct_set = set(clean_code(code) for code in str(correct_codes).split(','))
    generated_set = set(clean_code(code) for code in str(generated_codes).split(','))

    exact_match = len(correct_set & generated_set)
    first_digit_match = len({code[0] for code in correct_set} & {code[0] for code in generated_set})
    first_two_digits_match = len({code[:2] for code in correct_set} & {code[:2] for code in generated_set})
    first_three_digits_match = len({code[:3] for code in correct_set} & {code[:3] for code in generated_set})
    
    return exact_match, first_digit_match, first_two_digits_match, first_three_digits_match


# Ergebnisse berechnen
results = df.apply(lambda row: calculate_matches(row["Correct ICD Codes"], row["Generated ICD Codes"]), axis=1)
df[['Exact Matches', 'First Digit Matches', 'First Two Digits Matches', 'First Three Digits Matches']] = pd.DataFrame(results.tolist(), index=df.index)

# Gesamtzahlen berechnen
total_exact_matches = df['Exact Matches'].sum()
total_first_digit_matches = df['First Digit Matches'].sum()
total_first_two_digits_matches = df['First Two Digits Matches'].sum()
total_first_three_digits_matches = df['First Three Digits Matches'].sum()
    
total_cases = len(df)

# Ausgaben in csv speichern
output_file_path = '/scratch/meditron/FastChat/Skripts Sean/data/results_codiESP_matches.csv'
output_dir = os.path.dirname(output_file_path)
df.to_csv(output_file_path, index=False)

