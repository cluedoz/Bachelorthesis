from fairseq.models.transformer_lm import TransformerLanguageModel

# Konstanten für die Pfade und Parameter
DATA_DIR = "/scratch/BioLLM/BioGPT/data/PubMed/data-bin"
MODEL_DIR = "/scratch/BioLLM/BioGPT/checkpoints/Pre-trained-BioGPT"
MODEL_FILE = "checkpoint.pt"
BPE_CODES = "/scratch/BioLLM/BioGPT/data/bpecodes"
LENPEN = 1
MIN_LEN = 10
MAX_LEN = 50

# Parameter für Tests
BEAM = 5
REPETITION_PENALTY = 2.5
TOP_K = 50
TOP_P = 0.9
TEMPERATURE = 0.7

# Modell laden
model = TransformerLanguageModel.from_pretrained(
    MODEL_DIR,
    MODEL_FILE,
    DATA_DIR,
    tokenizer='moses',
    bpe='fastbpe',
    bpe_codes=BPE_CODES,
    min_len=MIN_LEN,
    max_len_b=MAX_LEN,
    lenpen=LENPEN,
    max_tokens=12000
)

if model.cfg.common.fp16:
    print('Converting to float 16')
    model.half()
model.cuda()

def generate_icd_codes(clinical_text):
    example_text = (
        "You are a clinical coder. Your job is to extract ICD-10 diagnosis codes from clinical text.\n"
        "Example clinical text: A 7-year-old male presented with a large facial mass due to marked right facial hypoplasia. "
        "Median NMR confirmed mandibular ankylosis accompanied by high secondary tissue retraction and microretrognathia, allowing only a maximum interincisal "
        "opening of 4 mm in addition to a class II malocclusion. Surgical resection of the ankylotic block of the right temporomandibular joint was performed "
        "by osteotomy at the level of the condylar neck and right coronoidectomy, followed by reconstruction with chondrocostal graft. "
        "At the time of final discharge, the patient maintained an oral opening of 3 cm.\n\n"
        "Example ICD-10 diagnosis codes from the example: Q67.0, Q18.8, M26.19, M26.4, M26.611"
    )
    prompt = (
        f"{example_text}\n\n"
        f"Here is a new clinical text. What are the CD-10 diagnosis codes for this text?\n\n"
        f"Clinical text: {clinical_text}\n"
        f"ICD-10 diagnosis codes for this clinical text:"
    )
    src_tokens = model.encode(prompt)
    
    generated = model.generate(
        [src_tokens], 
        max_len_a=MAX_LEN, 
        lenpen=LENPEN, 
        beam=BEAM,
        repetition_penalty=REPETITION_PENALTY, 
        top_k=TOP_K, 
        top_p=TOP_P,
        temperature=TEMPERATURE,
        sampling=True
    )

    tokens = generated[0][0]['tokens']

    output = model.decode(tokens)
    return output

# Beispiel für klinischen Text
clinical_text = (
    "A nine-year-old boy presented with an erythematous, crusted lesion with raised edges, erythematous, notably more active than the center of the lesion, "
    "about 10 cm, located in the right thigh. Treated with hydration disappears within a few weeks. However, at months she presents similar lesions of smaller "
    "size scattered in thorax, neck and thigh.\n"
    "A culture of the lesions was performed, with negative results. Given the intense pruritus and spontaneous resolution with hydration, the dermatologist "
    "responds that the most likely etiology is atopic eczema. Therefore, treatment with topical corticosteroids is recommended."
)

# ICD-Kodierungen generieren mit Einstellungen
icd_codes = generate_icd_codes(clinical_text)
print(f"Model's output: {icd_codes}")
