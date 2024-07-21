# Bachelorthesis
Dieses Respository enthält alle nötigen Files zur Nachvollziehung der Bachelorthesis "Automatisierte Diagnosekodierung mittels LLM's"
## Struktur

Es sind im Total zwölf Files welche folgend erklärt werden.

- BioGPT_tests_icd_coding.py 
	-  Dieses File enthält die Konfiguration des BioGPT Modell 
- preprocessing_codiESP.py
	- Diese File enthält die Vorverarbeitung des CodiESP-Datensatzes
- preprocessed_CodiESP_file.csv
	- Dieses File enthält die Ergebnisse der Vorverarbeitung
- MediTron_Tests_CodiESP.py
	- Dieses File enthält die finale Konfiguration des Meditron Modell
- predicted_icd_codes.csv
	- Dieses File enthält die Ergebnisse des Tests mit Meditron mit 50 zufällig ausgewählten Datensätzen
- Auswertung_CodiESP_matches.py
	- Diese File enthält den Code zur Auswertung der Übereinstimmigkeiten
- auswertung_codiESP_precision_recall.py
	- Diese File enthält den Code zur Auswertung von Precision und Recall
- results_codiESP_matches.csv
	- Diese File enthält die Ergebnisse der Auswertung der Übereinstimmigkeiten
- results_codiESP_precision_recall.csv
	- Diese File enthält die Ergebnisse der Auswertung von Precision und Recall
- tests_valid_icd_codes.py
	- Diese File enthält den Code zur Auswertung ob die generierten Kodes valide sind
- results_icd_code_validation.csv
	- Diese File enthält die Ergebnisse des Tests auf Validität
- Auswertung_meditron_final.xlsx
	- Dieses File enthält alle Auswertungen kombiniert in Excel

## Note
Allenfalls müssen Pfade angepasst werden und Token für die Benutzung von Modellen eingefügt werden.
Die verarbeiteten Datensätze sind open-source und müssen sich selbst beschafft werden. 
