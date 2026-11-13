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
    - advisor : calcul exact de la somme des crédits ECTS
    - auditor : détection de cycles de dépendances (DFS) ou vérification d'éligibilité
    - quiz_gen : validation du schéma JSON (4 options A/B/C/D, 1 seule clé correcte)
    - internship_matcher : similarité de Jaccard entre compétences du CV et de l'offre
    - workload_estimator : calcul horaire (1 ECTS = 25h, total CM+TP)
    - exam_planner : découpage en blocs Pomodoro sous contrainte d'heures

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
