"""
Outils métiers et calculs déterministes pour l'agent de démonstration.
Séance 5 : Sprint Dev et Prompt Engineering Défensif pour SLMs.
"""

from typing import Any

from src.core.ingestion import get_vector_store

# Référentiel de cours pour le calcul déterministe
COURS_CATALOGUE: dict[str, dict[str, Any]] = {
    "SD_321": {"ects": 4, "filiere": "Data", "semestre": "S8", "titre": "Machine Learning Hackathon"},
    "SD_322": {"ects": 4, "filiere": "Data", "semestre": "S8", "titre": "Big Data & Smart Data"},
    "SD_327": {"ects": 2, "filiere": "Data", "semestre": "S8", "titre": "MLOps & Déploiement"},
    "SD_217": {"ects": 4, "filiere": "Data", "semestre": "S7", "titre": "Économétrie & Bayes"},
    "TC_101": {"ects": 5, "filiere": "Tronc", "semestre": "S7", "titre": "Mathématiques & Stats"},
    "SSI_107": {"ects": 3, "filiere": "Cyber", "semestre": "S7", "titre": "Sécurité Réseaux"},
}


def calculate_ects_balance(selected_codes: list[str], target_ects: int = 30) -> dict[str, Any]:
    """
    Calcule de façon déterministe le total d'ECTS
    et le différentiel par rapport au seuil de validation.
    """
    total = 0
    details: list[str] = []

    for code in selected_codes:
        ue = code.strip().upper()
        if ue in COURS_CATALOGUE:
            pts = COURS_CATALOGUE[ue]["ects"]
            total += pts
            details.append(f"{ue} ({pts} ECTS)")

    balance = max(0, target_ects - total)
    status = "VALIDÉ" if total >= target_ects else "INCOMPLET"

    return {
        "total_ects": total,
        "target_ects": target_ects,
        "balance": balance,
        "status": status,
        "cours_retenus": ", ".join(details) if details else "Aucun cours identifié",
    }


def verifier_compatibilite_prerequis(code_ue: str, acquis_etudiant: list[str]) -> dict[str, Any]:
    """
    Interroge la section 'Prérequis' du syllabus demandé dans ChromaDB
    et retourne les données factuelles extraites de la base.
    """
    vs = get_vector_store(persist_directory="./chroma_db", reindex_if_empty=False)
    normalised_code = code_ue.strip().upper()

    try:
        docs = vs.similarity_search(
            query="Prérequis Obligatoires",
            k=1,
            filter={"code_ue": normalised_code},
        )
    except Exception as e:
        return {
            "statut": "ERREUR",
            "code_ue": normalised_code,
            "message": f"Erreur lors de l'accès à la base documentaire : {e}",
        }

    if not docs:
        return {
            "statut": "INTROUVABLE",
            "code_ue": normalised_code,
            "message": f"Le syllabus du cours '{normalised_code}' est introuvable.",
        }

    return {
        "statut": "SUCCES",
        "code_ue": normalised_code,
        "texte_prerequis": docs[0].page_content,
        "acquis_etudiant": acquis_etudiant,
    }
