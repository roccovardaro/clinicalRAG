#!/bin/bash

# Assicurati di essere nella root del progetto (ClinicalRAG)
# e che il virtual environment sia attivato, se lo utilizzi.

echo "Inizio il download dei modelli..."
echo "-----------------------------------"

echo "[1/2] Scaricamento del modello BiomedCLIP (phaseOne)..."
# Usiamo PYTHONPATH=. per far sì che i moduli riescano a trovare paths.py
PYTHONPATH=. python3 -c "
from src.phaseOne.model import download_and_save_model_locally
print('Avvio download BiomedCLIP...')
download_and_save_model_locally()
print('Download BiomedCLIP completato con successo!')
"

echo ""

echo "[2/2] Scaricamento del modello MedCPTCrossEncoder (phaseTwo)..."
PYTHONPATH=. python3 -c "
from src.phaseTwo.model import download_and_save_model_locally
print('Avvio download MedCPTCrossEncoder...')
download_and_save_model_locally()
print('Download MedCPTCrossEncoder completato con successo!')
"

echo "-----------------------------------"
echo "Tutti i modelli sono stati scaricati correttamente nella cartella 'models/'."
