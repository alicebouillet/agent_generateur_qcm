"""Outils Python Déterministes pour l'Agent Étudiant.

RÈGLE D'OR DE L'INGÉNIERIE SLM :
Tout calcul arithmétique, manipulation d'ensembles, algorithme de graphe,
validation de format JSON ou parsing par regex DOIT être exécuté en pur Python ici.
Ne laissez jamais le SLM faire des additions ou des détections de cycles.
"""

from typing import Any


def process_domain_logic(query: str, retrieved_docs: list[Any]) -> dict[str, Any]:
    """TODO: Implémentez ici la logique métier déterministe de votre agent.

    Exemples selon votre archétype :
    - quiz_gen : validation du schéma JSON (4 options A/B/C/D, 1 seule clé correcte)


    Args:
        query: Texte de la question de l'étudiant
        retrieved_docs: Chunks documentaires extraits par ChromaDB

    Returns:
        Dictionnaire contenant les données chiffrées ou vérifiées.
    """
    return {
        "status": "success",
        "documents_analyzed": len(retrieved_docs),
        "summary": "Données traitées avec succès par les outils Python déterministes.",
    }


def format_domain_response(tool_results: dict[str, Any]) -> str:
    """Génère une réponse formatée de secours si le SLM n'est pas disponible."""
    summary = tool_results.get("summary", "")
    analyzed = tool_results.get("documents_analyzed", 0)
    return f"Traitement effectué : {summary} ({analyzed} document(s) analysé(s))."


def validate_quiz_structure(raw_quiz: dict[str, Any]) -> bool:
    """Vérifie la conformité du schéma (présence de 4 options uniques, 1 seule clé correcte)."""
    try:
        options = raw_quiz.get("options", [])
        correct = raw_quiz.get("correct", "")
        if len(options) != 4 or len(set(options)) != 4:
            return False
        return correct in options
    except Exception:
        return False

def score_quiz_submission(user_answers: dict[int, str], solution_key: dict[int, str]) -> dict[str, Any]:
    """Compare les réponses fournies à la clé de correction et calcule la note finale."""
    total = len(solution_key)
    details = {
        q: {"donnee": user_answers.get(q), "attendue": rep,
            "correct": user_answers.get(q, "").strip().upper() == rep.strip().upper()}
        for q, rep in solution_key.items()
    }
    nb_correct = sum(d["correct"] for d in details.values())
    return {
        "score": nb_correct,
        "total": total,
        "note_sur_20": round(20 * nb_correct / total, 2) if total else 0.0,
        "details": details,
    }
