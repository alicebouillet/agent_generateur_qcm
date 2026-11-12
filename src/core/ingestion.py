"""
Pipeline d'ingestion RAG pour les syllabus de formation d'ingénieurs.
Lit les fichiers Markdown, extrait les métadonnées structurées,
découpe les documents et les indexe dans ChromaDB avec Ollama Embeddings.
"""

import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
SYLLABUS_DATA_DIRECTORY = os.getenv("SYLLABUS_DATA_DIRECTORY", "./data/syllabus")


def extract_syllabus_metadata(content: str, filename: str) -> dict[str, Any]:
    """
    Extrait les métadonnées clés d'un fichier de syllabus Markdown via regex.
    """
    metadata: dict[str, Any] = {
        "source": filename,
        "code_ue": "INCONNU",
        "filiere": "Générale",
        "semestre": "S8",
        "ects": 0,
    }

    # Recherche Code UE (ex: - **Code UE** : SD_321)
    code_match = re.search(r"\*\*Code UE\*\*\s*:\s*([A-Z0-9_-]+)", content)
    if code_match:
        metadata["code_ue"] = code_match.group(1).strip()

    # Recherche Filière
    filiere_match = re.search(r"\*\*Filière\*\*\s*:\s*([^\n\r]+)", content)
    if filiere_match:
        metadata["filiere"] = filiere_match.group(1).strip()

    # Recherche Semestre
    semestre_match = re.search(r"\*\*Semestre\*\*\s*:\s*([A-Za-z0-9]+)", content)
    if semestre_match:
        metadata["semestre"] = semestre_match.group(1).strip()

    # Recherche ECTS
    ects_match = re.search(r"\*\*Crédits ECTS\*\*\s*:\s*(\d+)", content)
    if ects_match:
        metadata["ects"] = int(ects_match.group(1).strip())

    return metadata


def load_and_chunk_syllabus(directory_path: str = SYLLABUS_DATA_DIRECTORY) -> list[Document]:
    """
    Charge tous les fichiers Markdown de syllabus et crée des documents LangChain
    avec les métadonnées adéquates pour le filtrage vectoriel.
    """
    path = Path(directory_path)
    if not path.exists():
        raise FileNotFoundError(f"Le dossier de syllabus n'existe pas : {directory_path}")

    documents: list[Document] = []

    for file_path in path.glob("*.md"):
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        meta = extract_syllabus_metadata(content, file_path.name)

        # Découpage sémantique par grande section (titres niveau 2 ##)
        sections = re.split(r"(?=\n##\s+)", content)

        for idx, section in enumerate(sections):
            section_clean = section.strip()
            if not section_clean:
                continue

            chunk_meta = meta.copy()
            chunk_meta["chunk_id"] = f"{meta['code_ue']}_chunk_{idx}"

            # Détection du titre de section
            first_line = section_clean.split("\n")[0]
            if first_line.startswith("##"):
                chunk_meta["section_title"] = first_line.replace("##", "").strip()
            else:
                chunk_meta["section_title"] = "En-tête & Informations Générales"

            documents.append(Document(page_content=section_clean, metadata=chunk_meta))

    return documents


def get_vector_store(
    persist_directory: str = CHROMA_PERSIST_DIRECTORY, reindex_if_empty: bool = True
) -> Chroma:
    """
    Initialise et retourne le VectorStore ChromaDB avec les embeddings locaux Ollama.
    Si la base est vide ou inexistante, indexe automatiquement les syllabus disponibles.
    """
    embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)

    needs_indexing = False
    if not os.path.exists(persist_directory):
        needs_indexing = True
    else:
        try:
            temp_vs = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            if hasattr(temp_vs, "_collection") and temp_vs._collection.count() == 0:
                needs_indexing = True
        except Exception:
            needs_indexing = True

    if reindex_if_empty and needs_indexing:
        print(f"📦 Indexation initiale des syllabus depuis {SYLLABUS_DATA_DIRECTORY}...")
        docs = load_and_chunk_syllabus()
        vectorstore = Chroma.from_documents(
            documents=docs, embedding=embeddings, persist_directory=persist_directory
        )
        print(f"✅ {len(docs)} segments de syllabus indexés avec succès dans ChromaDB.")
        return vectorstore

    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


if __name__ == "__main__":
    import sys

    # Support UTF-8 console Windows pour les émojis
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print(f"🚀 Lancement de l'ingestion des syllabus dans ChromaDB ({CHROMA_PERSIST_DIRECTORY})...")
    vectorstore = get_vector_store(reindex_if_empty=True)
    count = vectorstore._collection.count() if hasattr(vectorstore, "_collection") else "N/A"
    print(f"✅ Ingestion terminée ! {count} segments indexés dans la base vectorielle.")
