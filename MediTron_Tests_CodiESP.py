import os
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

# Token für Hugging Face
mytoken = "use_your_own_token"
os.environ["HF_TOKEN"] = mytoken
os.environ["HF_HUB_CACHE"] = "/scratch/meditron/llm_cache"

# Auswahl eines einzigen GPUs, falls der andere belegt ist. "0" für die Auswahl des ersten GPU, "1" für den zweiten GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Tokenizer und Modell laden
tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b", use_auth_token=mytoken)
tokenizer.pad_token_id = tokenizer.eos_token_id

def load_model(device):
    return AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b", use_auth_token=mytoken).to(device)

def generate_icd10_codes(input_text, device):
    prompt = (
        f"You are a clinical coder. Your task is to extract ICD-10 diagnosis codes from discharge summaries.\n\n"
        f"Follow the example precisely:\n"
        f"Example of a Discharge summary: A 56-year-old woman with mild and unconfirmed upper right hemiarcade pain thirteen months ago, but who had worsened in the last week. "
        f"In the inspection only mesiodistal fissure of the crown of the upper right premolar was observed. "
        f"In the radiographic examination we found nothing. The emergency treatment was: camera opening of this premolar and placement of a medical "
        f"device inside it (Cresophene, Septodont) temporarily covered with a cement (Cavit, Espe). Dental pain disappeared and later it was planned to "
        f"make an endodontic and placement of a metal-porcelain protection crown.\n"
        f"Extracted ICD-10 codes from the example: R52, K08.89\n\n"
        f"Extract ICD-10 diagnosis codes from the following discharge summary: {input_text}\n"
        f"Extracted ICD-10 codes:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True, padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    try:
        model = load_model(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=2.5,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory. Switching to CPU.")
        device = torch.device("cpu")
        model = load_model(device)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=50,
                num_return_sequences=1,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=2.5,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extrahiere die ICD-10 Codes aus der generierten Antwort
    if "Extracted ICD-10 codes:" in generated_text:
        extracted_text = generated_text.split("Extracted ICD-10 codes:")[-1].strip()
        # Verwenden eines regulären Ausdrucks, um die Codes zu extrahieren
        import re
        extracted_codes = re.findall(r'\b[A-Z]\d{2}\.\d{1,2}\b', extracted_text)
    else:
        extracted_codes = []

    return extracted_codes

# Pfad zur CSV-Datei
csv_path = "/scratch/meditron/FastChat/Skripts Sean/data/preprocessed_CodiESP_file.csv"

# CSV-Datei laden
df = pd.read_csv(csv_path)

# 50 zufällige Zeilen auswählen
random_rows = df.sample(n=50)

# Ausgabe-Datei
output_path = "/scratch/meditron/FastChat/Skripts Sean/data/predicted_icd_codes.csv"

# Liste zum Speichern der Ergebnisse
results = []

for _, row in random_rows.iterrows():
    input_text = row["Case Note"]
    correct_icd_codes = row["ICD Codes"]
    note_id = row["ID"]
    note_length = row["Length"]

    # ICD-10-Codes generieren
    generated_codes = generate_icd10_codes(input_text, device)

    # Ergebnisse speichern
    results.append({
        "ID": note_id,
        "Correct ICD Codes": correct_icd_codes,
        "Generated ICD Codes": ", ".join(generated_codes),
        "Length": note_length
    })

# DataFrame mit den Ergebnissen erstellen
results_df = pd.DataFrame(results)

# Ergebnisse in die CSV-Datei speichern
results_df.to_csv(output_path, index=False)

print("Generated ICD-10 codes for 50 random cases and saved to file.")
