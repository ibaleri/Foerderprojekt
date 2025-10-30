"""
Test 2: Evaluierung verschiedener Embedding-Modelle
Vergleicht 3 Modelle anhand von Ground Truth Queries
"""
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time
import os
from typing import Dict, List

# Konfiguration: 3 Modelle zum Testen
MODELS = {
    'e5-base-sts-en-de': 'danielheinz/e5-base-sts-en-de',  # Aktuelles Modell (768 dim, DE-tuned)
    'bge-m3': 'BAAI/bge-m3',  # State-of-the-art 2025 (1024 dim, 8192 tokens, multilingual)
    'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',  # Lightweight (384 dim, schnell)
}

class EmbeddingModelTester:
    def __init__(self, csv_file: str, ground_truth_file: str):
        self.csv_file = csv_file
        self.ground_truth_file = ground_truth_file
        self.df = None
        self.ground_truth = None
        self.results = []
        self.all_test_runs = []

    def load_data(self):
        """Lade Projekte und Ground Truth"""
        print("Lade Daten...")
        # Projekte laden
        CSV_COLUMNS = [
            'FKZ', 'Ressort', 'Referat', 'PT', 'Arb.-Einh.',
            'Zuwendungsempfänger', 'Gemeindekennziffer_Empfänger', 'Stadt_Gemeinde_Empfänger',
            'Ort_Empfänger', 'Bundesland_Empfänger', 'Staat_Empfänger', 'Ausführende Stelle',
            'Gemeindekennziffer_Ausführend', 'Stadt_Gemeinde_Ausführend', 'Ort_Ausführend',
            'Bundesland_Ausführend', 'Staat_Ausführend', 'Thema', 'Leistungsplansystematik',
            'Klartext Leistungsplansystematik', 'Laufzeit von', 'Laufzeit bis', 'Fördersumme in EUR',
            'Förderprofil', 'Verbundprojekt', 'Förderart', 'Zusätzliche Spalte'
        ]
        self.df = pd.read_csv(self.csv_file, sep=';', encoding='latin1', header=None, low_memory=False)
        self.df.columns = CSV_COLUMNS

        # Bereinigung
        for column in self.df.columns:
            if self.df[column].dtype == 'object':
                self.df[column] = self.df[column].str.replace('="', '', regex=False)
                self.df[column] = self.df[column].str.replace('"', '', regex=False)
                self.df[column] = self.df[column].str.lower()

        # Ground Truth laden
        self.ground_truth = pd.read_csv(self.ground_truth_file)
        print(f"Geladen: {len(self.df)} Projekte, {len(self.ground_truth)} Ground Truth Queries")

    def test_model(self, model_name: str, model_path: str) -> Dict:
        """Teste ein einzelnes Modell"""
        print(f"\n{'='*60}")
        print(f"Teste Modell: {model_name}")
        print(f"{'='*60}")

        # Lade Modell
        print("Lade Modell...")
        model = SentenceTransformer(model_path)

        # Erstelle Embeddings (oder lade vorhandene für e5-base)
        embedding_file = f'test_2_embeddings_{model_name}.npy'

        # Für e5-base: Versuche project_embeddings_test.npy zu verwenden
        if model_name == 'e5-base-sts-en-de' and os.path.exists('project_embeddings_test.npy'):
            print("Lade vorhandene Embeddings von project_embeddings_test.npy...")
            start_time = time.time()
            embeddings = np.load('project_embeddings_test.npy')
            embedding_time = time.time() - start_time
            print(f"  Geladen in {embedding_time:.2f}s (statt ~30-60 Min neu erstellen!)")
        elif os.path.exists(embedding_file):
            print(f"Lade gespeicherte Embeddings von {embedding_file}...")
            start_time = time.time()
            embeddings = np.load(embedding_file)
            embedding_time = time.time() - start_time
            print(f"  Geladen in {embedding_time:.2f}s")
        else:
            print("Erstelle Embeddings...")
            start_time = time.time()
            combined_texts = (
                "Zuwendungsempfänger: " + self.df['Zuwendungsempfänger'].fillna('') +
                ". Thema: " + self.df['Thema'].fillna('') +
                ". Klartext: " + self.df['Klartext Leistungsplansystematik'].fillna('')
            )
            embeddings = model.encode(combined_texts.tolist(), convert_to_numpy=True, show_progress_bar=True)
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            embedding_time = time.time() - start_time

            # Speichere für zukünftige Nutzung
            np.save(embedding_file, embeddings)
            print(f"  Embeddings gespeichert als {embedding_file}")

        # FAISS Index erstellen
        print("Erstelle FAISS Index...")
        d = embeddings.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(embeddings)

        # Teste Queries
        print("Teste Queries...")
        print(f"{'='*80}")
        print(f"{'Nr':<4} {'Query':<45} {'Erwartet':<15} {'Gefunden':<10} {'Zeit(s)':<8}")
        print(f"{'='*80}")

        hits = 0
        total_query_time = 0

        for idx, row in self.ground_truth.iterrows():
            query = row['query']
            expected_fkz = row['expected_fkz']

            # Query Embedding
            start_time = time.time()
            query_embedding = model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

            # Suche Top-10
            distances, indices = index.search(query_embedding, 250)
            query_time = time.time() - start_time
            total_query_time += query_time

            # Prüfe ob expected_fkz in Top-10 (CASE-INSENSITIVE!)
            top_10_fkz = self.df.iloc[indices[0]]['FKZ'].tolist()
            # Vergleich in lowercase (da DataFrame lowercase ist)
            expected_fkz_lower = str(expected_fkz).lower()
            is_hit = expected_fkz_lower in [str(fkz).lower() for fkz in top_10_fkz]
            if is_hit:
                hits += 1

            # Speichere Test-Run
            test_run = {
                'Modell': model_name,
                'Query-Nr': idx + 1,
                'Query': query,
                'Erwartete FKZ': expected_fkz,
                'Treffer': 'JA' if is_hit else 'NEIN',
                'Query-Zeit (s)': round(query_time, 3)
            }
            self.all_test_runs.append(test_run)

            # Print für jeden Versuch
            hit_status = "JA" if is_hit else "NEIN"
            query_short = query[:43] + ".." if len(query) > 45 else query
            print(f"{idx+1:<4} {query_short:<45} {expected_fkz:<15} {hit_status:<10} {query_time:<8.3f}")

        print(f"{'='*80}")

        # Berechne Metriken
        hit_rate = (hits / len(self.ground_truth)) * 100
        avg_query_time = total_query_time / len(self.ground_truth)

        result = {
            'Modell': model_name,
            'Embedding-Dimension': d,
            'Embedding-Zeit (s)': round(embedding_time, 2),
            'Trefferquote (%)': round(hit_rate, 2),
            'Ø Query-Zeit (s)': round(avg_query_time, 3),
            'Hits': hits,
            'Total Queries': len(self.ground_truth)
        }

        print(f"\nErgebnisse:")
        print(f"  Trefferquote: {hit_rate:.2f}%")
        print(f"  Ø Query-Zeit: {avg_query_time:.3f}s")
        print(f"  Hits: {hits}/{len(self.ground_truth)}")

        return result

    def run_all_tests(self):
        """Teste alle Modelle"""
        self.load_data()

        for model_name, model_path in MODELS.items():
            result = self.test_model(model_name, model_path)
            self.results.append(result)

        # Speichere Ergebnisse
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('test_2_results_embedding_models.csv', index=False, encoding='utf-8')

        # Speichere alle Test-Runs
        test_runs_df = pd.DataFrame(self.all_test_runs)
        test_runs_df.to_csv('test_2_all_test_runs.csv', index=False, encoding='utf-8')

        # Ausgabe: Komplette Liste aller Test-Runs
        print(f"\n{'='*100}")
        print("KOMPLETTE LISTE ALLER TEST-RUNS")
        print(f"{'='*100}")
        print(test_runs_df.to_string(index=False))

        # Ausgabe: Summary
        print(f"\n{'='*100}")
        print("SUMMARY - FINALE ERGEBNISSE")
        print(f"{'='*100}")
        print(results_df.to_string(index=False))
        print(f"\nErgebnisse gespeichert:")
        print(f"  - Test-Runs: test_2_all_test_runs.csv")
        print(f"  - Summary: test_2_results_embedding_models.csv")


if __name__ == "__main__":
    tester = EmbeddingModelTester('projekte.csv', 'ground_truth.csv')
    tester.run_all_tests()
