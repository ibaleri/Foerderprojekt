import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests
import gradio as gr
import os
from typing import Tuple
from dataclasses import dataclass, field
import logging

#Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Config:
    EMBEDDINGS_FILE: str = 'test_2_embeddings_bge-m3.npy'
    MODEL_NAME: str = 'BAAI/bge-m3'
    CSV_FILE: str = 'projekte.csv'
    DISPLAY_COLUMNS: list = field(default_factory=lambda: [
        'FKZ', 'Zuwendungsempfänger', 'Thema', 'Fördersumme in EUR',
        'Laufzeit von', 'Laufzeit bis', 'Bundesland_Empfänger'
    ])
    CSV_COLUMNS: list = field(default_factory=lambda: [
        'FKZ', 'Ressort', 'Referat', 'PT', 'Arb.-Einh.',
        'Zuwendungsempfänger', 'Gemeindekennziffer_Empfänger', 'Stadt_Gemeinde_Empfänger',
        'Ort_Empfänger', 'Bundesland_Empfänger', 'Staat_Empfänger', 'Ausführende Stelle',
        'Gemeindekennziffer_Ausführend', 'Stadt_Gemeinde_Ausführend', 'Ort_Ausführend',
        'Bundesland_Ausführend', 'Staat_Ausführend', 'Thema', 'Leistungsplansystematik',
        'Klartext Leistungsplansystematik', 'Laufzeit von', 'Laufzeit bis', 'Fördersumme in EUR',
        'Förderprofil', 'Verbundprojekt', 'Förderart', 'Zusätzliche Spalte'
    ])
    OPENAI_API_KEY: str = ""
    OPENAI_API_URL: str = "https://api.openai.com/v1/chat/completions"


class ProjectSearchEngine:
    def __init__(self, config: Config):
        self.config = config
        self.df = None
        self.model = SentenceTransformer(self.config.MODEL_NAME)
        self.index = None
        self.projekt_embeddings = None
        self.api_key = self.config.OPENAI_API_KEY

    def Lade_Daten_und_Überprüfe_ob_fehler(self, csv_file: str, expected_columns: int) -> pd.DataFrame:
        try:
            logger.info("Lade Daten und prüfe auf Spaltenfehler...")
            df = pd.read_csv(csv_file, sep=';', encoding='latin1', header=None, low_memory=False)
            if df.shape[1] != expected_columns:
                raise ValueError(f"Erwartet {expected_columns} Spalten, aber gefunden {df.shape[1]} Spalten.")
            return df
        except Exception as e:
            logger.error(f"Fehler beim Einlesen der CSV-Datei: {e}")
            raise

    def load_data(self) -> None:
        try:
            logger.info("Lade Daten...")
            self.df = self.Lade_Daten_und_Überprüfe_ob_fehler(self.config.CSV_FILE, len(self.config.CSV_COLUMNS))
            self.df.columns = self.config.CSV_COLUMNS
            self.Daten_bereinigen()
        except Exception as e:
            logger.error(f"Fehler beim Einlesen der CSV-Datei: {e}")
            raise

    def Daten_bereinigen(self) -> None:
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                self.df[column] = self.df[column].str.replace('="', '', regex=False)
                self.df[column] = self.df[column].str.replace('"', '', regex=False)
                self.df[column] = self.df[column].str.lower()

        self.df['Fördersumme in EUR'] = pd.to_numeric(
            self.df['Fördersumme in EUR'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False),
            errors='coerce'
        )

        date_columns = ['Laufzeit von', 'Laufzeit bis']
        for col in date_columns:
            self.df[col] = pd.to_datetime(self.df[col], format='%d.%m.%Y', errors='coerce')
            self.df[col] = self.df[col].dt.strftime('%d.%m.%Y')

    def Lade_oder_Erstelle_Embedded_Datei(self) -> None:
        if os.path.exists(self.config.EMBEDDINGS_FILE):
            logger.info("Lade gespeicherte Embeddings...")
            self.projekt_embeddings = np.load(self.config.EMBEDDINGS_FILE)
        else:
            logger.info("Erstelle neue Embeddings...")
            combined_texts = (
                "Zuwendungsempfänger: " + self.df['Zuwendungsempfänger'].fillna('') +
                ". Thema: " + self.df['Thema'].fillna('') +
                ". Klartext: " + self.df['Klartext Leistungsplansystematik'].fillna('')
            )
            self.projekt_embeddings = self.model.encode(combined_texts.tolist(), convert_to_numpy=True)
            self.projekt_embeddings = self.projekt_embeddings / np.linalg.norm(self.projekt_embeddings, axis=1, keepdims=True)
            np.save(self.config.EMBEDDINGS_FILE, self.projekt_embeddings)
            logger.info("Embeddings wurden gespeichert.")

    def Initialisiere_Suchindex(self) -> None:
        try:
            if self.projekt_embeddings is None:
                raise ValueError("Embeddings sind nicht geladen oder erstellt.")
            d = self.projekt_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.projekt_embeddings)
            logger.info("FAISS-Index wurde erfolgreich initialisiert.")
        except Exception as e:
            logger.error(f"Fehler bei der Initialisierung des FAISS-Indexes: {e}")
            raise

    def Extrahiere_Wichtige_Begriffe(self, frage: str) -> list:

        try:
            prompt = f"Extrahiere nur die wichtigsten Schlüsselwörter aus der folgenden Frage, die für eine semantische Suche nach Themen relevant sind:'{frage}'. Ignoriere allgemeine Wörter wie 'Themen', 'Auflistung', 'Frage' oder andere unbedeutende Begriffe. Gib nur spezifische thematische Begriffe zurück, getrennt durch Kommas."

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.6,
                "max_tokens": 2000
            }

            response = requests.post(self.config.OPENAI_API_URL, headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                keywords = response_data['choices'][0]['message']['content'].strip()
                return [kw.strip() for kw in keywords.split(",")]
            else:
                logger.error(f"OpenAI API Fehler {response.status_code}: {response.text}")
                return []

        except Exception as e:
            logger.error(f"Fehler bei der Schlagwort-Extraktion: {e}")
            return []

    def Semantische_Suche(self, keywords: list, k: int = 100) -> pd.DataFrame:

        query = " ".join(keywords)
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        distances, indices = self.index.search(query_embedding, k)
        results = self.df.iloc[indices[0]]
        return results.head(k)

    def Beantworte_Frage(self, frage: str) -> str:
        try:
            # 1. Schlagwortextraktion
            keywords = self.Extrahiere_Wichtige_Begriffe(frage)
            if not keywords:
                return "Ich konnte keine relevanten Begriffe extrahieren."

            logger.info(f"Extrahierte Begriffe: {keywords}")

            # 2. Semantische Suche
            relevant_projects = self.Semantische_Suche(keywords, k=100)
            if relevant_projects.empty:
                return "Keine relevanten Ergebnisse gefunden."

            # 3. Begrenze auf 30 Projekte für GPT-4o
            relevant_projects_for_llm = relevant_projects.head(30)

            # 4. Ergebnisse zusammenstellen mit allen wichtigen Informationen
            projects_summary = "\n".join([
                f"FKZ: {row['FKZ']}, Empfänger: {row['Zuwendungsempfänger']}, Thema: {row['Thema']}, Fördersumme: {row['Fördersumme in EUR']:.2f} EUR"
                if pd.notna(row['Fördersumme in EUR'])
                else f"FKZ: {row['FKZ']}, Empfänger: {row['Zuwendungsempfänger']}, Thema: {row['Thema']}, Fördersumme: nicht verfügbar"
                for _, row in relevant_projects_for_llm.iterrows()
            ])
            prompt = f"Die folgenden Projekte könnten relevant sein:\n{projects_summary}\n\nBeantworte die Frage basierend auf diesen Projekten:\n{frage}\n\nWenn die Frage eine Sortierung verlangt (z.B. nach Fördersumme), sortiere die Projekte entsprechend. Wenn du die Antwort nicht kennst, sage einfach, dass du es nicht weißt. Füge dabei die relevanten Informationen (FKZ, Empfänger, Fördersumme) zu den Projekten hinzu."

            # Log Prompt-Größe
            prompt_chars = len(prompt)
            prompt_tokens_estimate = prompt_chars / 4
            logger.info(f"Prompt-Größe: {prompt_chars} Zeichen, geschätzt {prompt_tokens_estimate:.0f} Tokens")
            logger.info(f"Anzahl Projekte im Prompt: {len(relevant_projects_for_llm)}")

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 8000
            }

            response = requests.post(self.config.OPENAI_API_URL, headers=headers, json=data)

            if response.status_code == 200:
                response_data = response.json()
                answer = response_data['choices'][0]['message']['content'].strip()
                return answer
            else:
                logger.error(f"OpenAI API Fehler {response.status_code}: {response.text}")
                return "Entschuldigung, ich konnte deine Anfrage momentan nicht bearbeiten."

        except Exception as e:
            logger.error(f"Fehler bei der Verarbeitung der Frage: {e}")
            return f"Ein Fehler ist aufgetreten: {str(e)}"


def Gradio_Oberfläche(search_engine: ProjectSearchEngine) -> gr.Blocks:
    with gr.Blocks(css="""
        #dataframe_output {
            width: 100%;
            height: 500px !important;
            max-height: 500px !important;
            overflow-y: scroll;
            overflow-x: scroll;
        }
        .table-wrap {
            max-height: 500px !important;
            overflow: visible !important;
        }
        table {
            width: 100%;
        }
    """) as demo:
        gr.Markdown("# Forschungsprojekt-KI-Agent")

        # Eingabefelder
        query_input = gr.Textbox(
            label="Ihre Frage",
            placeholder="Stellen Sie eine Frage zu den Forschungsprojekten"
        )
        submit_button = gr.Button("Frage stellen")
        answer_output = gr.Textbox(
            label="Antwort",
            interactive=False,
            elem_id="answer_output"
        )

        # Eingabefeld für Filter
        filter_input = gr.Textbox(
            label="Filter nach Thema",
            placeholder="Geben Sie ein Schlüsselwort ein, um nach Themen zu filtern (dynamisch)",
            interactive=True
        )

        dataframe_output = gr.Dataframe(
            label="Relevante Projekte",
            interactive=False,
            elem_id="dataframe_output",
            wrap=True
        )

        # Variable um die zuletzt gefundenen Projekte zu speichern (als Liste von Dicts)
        state_relevant_projects = gr.State(value=[])

        #  Frage stellen und relevante Projekte anzeigen
        def process_user_query(query, current_state):
            answer = search_engine.Beantworte_Frage(query)
            keywords = search_engine.Extrahiere_Wichtige_Begriffe(query)
            relevant_projects = search_engine.Semantische_Suche(keywords, k=100)

            # Nur die gewünschten Spalten auswählen
            relevant_projects = relevant_projects[search_engine.config.DISPLAY_COLUMNS]

            # Konvertiere den DataFrame in eine Liste von Dictionaries für den State
            relevant_projects_dict = relevant_projects.to_dict(orient='records')

            return answer, relevant_projects, relevant_projects_dict

        #dynamische Filterfunktion
        def filter_results(filter_keyword, relevant_projects_dict):
            if not relevant_projects_dict:
                return pd.DataFrame()

            relevant_projects = pd.DataFrame(relevant_projects_dict)

            if filter_keyword:
                filtered_projects = relevant_projects[
                    relevant_projects['Thema'].str.contains(filter_keyword, case=False, na=False)
                ]
            else:
                filtered_projects = relevant_projects

            return filtered_projects

        # Suche verbinden
        submit_button.click(
            fn=process_user_query,
            inputs=[query_input, state_relevant_projects],
            outputs=[answer_output, dataframe_output, state_relevant_projects]
        )

        #Dynamischer Filter
        filter_input.change(
            fn=filter_results,
            inputs=[filter_input, state_relevant_projects],
            outputs=[dataframe_output]
        )

    return demo


def main():
    config = Config()
    search_engine = ProjectSearchEngine(config)
    search_engine.load_data()
    search_engine.Lade_oder_Erstelle_Embedded_Datei()
    search_engine.Initialisiere_Suchindex()

    demo = Gradio_Oberfläche(search_engine)
    demo.launch()


if __name__ == "__main__":
    main()
