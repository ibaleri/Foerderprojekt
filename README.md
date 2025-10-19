# Förderprojekt KI-Agent

## Was macht das Tool?

Ein KI-gestützter Recherche-Assistent für Förderprojekte. Das Tool kombiniert semantische Suche mit GPT-4o, um präzise Antworten auf Fragen zu Forschungsprojekten zu liefern.

**Funktionsweise:**
1. Nutzer stellt eine Frage (z.B. "Welche Projekte beschäftigen sich mit Wasserstoff?")
2. GPT-4o extrahiert relevante Suchbegriffe
3. Semantische Suche (Embeddings) findet die 100 relevantesten Projekte
4. Top 30 Projekte werden an GPT-4o übergeben
5. GPT-4o beantwortet die Frage basierend auf diesen Projekten

## Voraussetzungen

- Python 3.8 oder höher
- OpenAI API Key (GPT-4o Zugang)
- Minimum 8GB RAM (für Embeddings)

## Installation

1. **Virtual Environment erstellen und aktivieren:**

   ```bash
   # Navigieren Sie zum Projektordner
   cd C:\Users\Ibale\PycharmProjects\pythonProject

   # Virtual Environment erstellen (falls noch nicht vorhanden)
   python -m venv .venv

   # Virtual Environment aktivieren
   # Auf Windows:
   .venv\Scripts\activate

   # Auf Linux/Mac:
   source .venv/bin/activate
   ```

   Nach der Aktivierung sollte `(.venv)` vor Ihrer Kommandozeile erscheinen.

2. **Abhängigkeiten installieren:**
```bash
pip install pandas numpy sentence-transformers faiss-cpu requests gradio
```

3. **Projektdatenbank herunterladen (projekte.csv):**

   a. Öffnen Sie https://foerderportal.bund.de/foekat/jsp/SucheAction.do?actionMode=searchmask

   b. Führen Sie eine leere Suche aus (ohne Filter einzugeben)

   c. Es werden alle Projekte aufgelistet

   d. Klicken Sie auf "Ausgabe als Textdatei"

   e. Der Browser lädt die `projekte.csv` herunter

   f. Verschieben Sie die `projekte.csv` in den Projektordner (`.venv/`)

4. **API Key konfigurieren:**

Öffnen Sie `Foerderprojekt_test.py` und tragen Sie Ihren OpenAI API Key ein (Zeile 35):
```python
OPENAI_API_KEY: str = "sk-proj-IHR_API_KEY_HIER"
```

## Starten

```bash
python Foerderprojekt_test.py
```

Die Gradio-Oberfläche öffnet sich automatisch im Browser unter `http://localhost:7860`

## Verwendung

1. Frage in das Textfeld eingeben
2. "Frage stellen" klicken
3. Ergebnis wird angezeigt mit:
   - KI-generierte Antwort
   - Liste relevanter Projekte (mit Filter-Funktion)

## Konfiguration

In `Foerderprojekt_test.py` Klasse `Config` (Zeile 18-36):

```python
EMBEDDINGS_FILE: str = 'test_2_embeddings_bge-m3.npy'  # Wo Embeddings gespeichert werden
MODEL_NAME: str = 'BAAI/bge-m3'                         # Embedding-Modell
CSV_FILE: str = 'projekte.csv'                         # Projektdatenbank
OPENAI_API_KEY: str = "HIER_API_KEY"                   # Ihr OpenAI Key
```

## Erste Verwendung

Beim ersten Start werden automatisch Embeddings erstellt (dauert ca. 30-60 Minuten). Diese werden gespeichert und bei zukünftigen Starts wiederverwendet.

