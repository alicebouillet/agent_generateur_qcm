"""Outils Python déterministes de l'agent Générateur de QCM (quiz_gen).

RÈGLE D'OR DE L'INGÉNIERIE SLM :
Tout calcul, validation de format JSON ou parsing par regex est exécuté
en pur Python ici. Le SLM ne fait que rédiger les questions ; c'est ce module
qui vérifie leur structure et qui note les réponses de l'étudiant.

Format d'une question de QCM attendu partout dans ce module :
    {
        "question": "Que signifie ECTS ?",
        "options": ["...", "...", "...", "..."],   # exactement 4 textes distincts
        "correct": "B"                             # lettre A-D = position dans options
    }

Organisation du fichier :
    1. Préparation de la génération  -> process_domain_logic
    2. Lecture de la sortie du SLM    -> validate_quiz_structure, parse_quiz_output
    3. Correction du quiz             -> correct_answer_text, score_quiz_submission
    4. Réponse de secours             -> format_domain_response
"""

import json
import re
from typing import Any

# --------------------------------------------------------------------------- #
# Constantes
# --------------------------------------------------------------------------- #

# Lettres des 4 options : LETTRES[i] désigne options[i] (A -> 0, B -> 1, ...)
LETTRES = ("A", "B", "C", "D")

# Nombre de questions : valeur par défaut et bornes (un SLM 3B sature vite)
NB_QUESTIONS_DEFAUT = 3
NB_QUESTIONS_MIN = 1
NB_QUESTIONS_MAX = 10

# Code d'UE : 2 à 5 lettres, tiret optionnel, 2 à 4 chiffres (ex : INF801, MATH-204).
# Pas d'espace autorisé, sinon "FAIS 10" serait pris pour un code d'UE.
# À adapter au format réel des codes du syllabus.
REGEX_CODE_UE = re.compile(r"\b([A-Z]{2,5}-?\d{2,4})\b")

# Nombre de questions demandé : "5 questions", "10 QCM", "3 quiz"
REGEX_NB_QUESTIONS = re.compile(r"(\d+)\s*(questions?|qcm|quiz)", re.IGNORECASE)

# Bloc JSON dans la réponse du SLM : du premier [ ou { jusqu'au dernier ] ou }
REGEX_JSON = re.compile(r"\[.*\]|\{.*\}", re.DOTALL)


# --------------------------------------------------------------------------- #
# 1. Préparation de la génération (avant l'appel au SLM)
# --------------------------------------------------------------------------- #

def process_domain_logic(query: str, retrieved_docs: list[Any]) -> dict[str, Any]:
    """Prépare les paramètres de génération du QCM à partir de la requête et du RAG.

    Args:
        query: Texte de la demande de l'étudiant.
        retrieved_docs: Extraits documentaires renvoyés par ChromaDB.

    Returns:
        Dictionnaire avec :
        - status : "success" ou "no_content" (rien sur quoi baser un quiz)
        - code_ue : code d'UE détecté dans la requête, ou None
        - nb_questions : nombre de questions à générer
        - documents_analyzed : nombre d'extraits retenus
        - summary : résumé lisible
    """
    # Code d'UE éventuellement cité par l'étudiant
    ue_match = REGEX_CODE_UE.search(query.upper())
    code_ue = ue_match.group(1) if ue_match else None

    # Nombre de questions demandé, ramené dans les bornes autorisées
    nb_match = REGEX_NB_QUESTIONS.search(query)
    if nb_match:
        nb_questions = min(max(int(nb_match.group(1)), NB_QUESTIONS_MIN), NB_QUESTIONS_MAX)
    else:
        nb_questions = NB_QUESTIONS_DEFAUT

    # Si une UE est demandée, on ne garde que les extraits de cette UE
    docs = retrieved_docs
    if code_ue:
        docs = [
            d for d in retrieved_docs
            if str(d.metadata.get("code_ue", "")).upper().replace(" ", "") == code_ue
        ]

    # Aucun contenu : on le signale pour que l'agent n'appelle pas le SLM
    if not docs:
        return {
            "status": "no_content",
            "code_ue": code_ue,
            "nb_questions": nb_questions,
            "documents_analyzed": 0,
            "summary": "Aucun contenu de cours trouvé pour générer un QCM.",
        }

    return {
        "status": "success",
        "code_ue": code_ue,
        "nb_questions": nb_questions,
        "documents_analyzed": len(docs),
        "summary": f"{nb_questions} question(s) à générer sur {code_ue or 'le contenu trouvé'}.",
    }


# --------------------------------------------------------------------------- #
# 2. Lecture et validation de la sortie du SLM
# --------------------------------------------------------------------------- #

def validate_quiz_structure(raw_quiz: Any) -> bool:
    """Vérifie qu'une question respecte le schéma (voir docstring du module).

    Returns:
        True si la question est exploitable, False sinon.
    """
    if not isinstance(raw_quiz, dict):
        return False

    # Énoncé : chaîne non vide
    question = raw_quiz.get("question")
    if not isinstance(question, str) or not question.strip():
        return False

    # ① Exactement 4 options, dans une liste
    options = raw_quiz.get("options")
    if not isinstance(options, list) or len(options) != len(LETTRES):
        return False
    # ② Chaque option est un texte non vide
    if not all(isinstance(o, str) and o.strip() for o in options):
        return False
    # ③ Options toutes différentes (sans tenir compte de la casse ni des espaces)
    if len({o.strip().lower() for o in options}) != len(LETTRES):
        return False

    # ④ Bonne réponse : une lettre A-D
    correct = raw_quiz.get("correct")
    return isinstance(correct, str) and correct.strip().upper() in LETTRES


def parse_quiz_output(llm_text: str) -> list[dict[str, Any]]:
    """Extrait les questions JSON de la réponse texte du SLM.

    Le SLM entoure souvent le JSON de phrases ou de ```json ... ```, d'où
    l'extraction par regex avant json.loads.

    Returns:
        Liste des questions valides (vide si rien d'exploitable).
    """
    # 1. Isoler le bloc JSON dans le texte
    match = REGEX_JSON.search(llm_text)
    if not match:
        return []

    # 2. Convertir le texte en objets Python (JSON mal formé -> rien)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []

    # 3. Toujours travailler sur une liste (le SLM peut renvoyer une seule question)
    questions = data if isinstance(data, list) else [data]

    # 4. Ne garder que les questions conformes au schéma
    return [q for q in questions if validate_quiz_structure(q)]


# --------------------------------------------------------------------------- #
# 3. Correction du quiz
# --------------------------------------------------------------------------- #

def correct_answer_text(quiz: dict[str, Any]) -> str:
    """Renvoie le texte de la bonne réponse (ex : 'B' -> options[1]).

    À n'appeler que sur une question validée par validate_quiz_structure,
    sinon ValueError / KeyError.
    """
    index = LETTRES.index(quiz["correct"].strip().upper())
    return quiz["options"][index]


def score_quiz_submission(
    user_answers: dict[int, str], solution_key: dict[int, str]
) -> dict[str, Any]:
    """Compare les réponses de l'étudiant à la clé de correction et calcule la note.

    Args:
        user_answers: {numéro de question: lettre choisie} ; réponse absente ou None = fausse.
        solution_key: {numéro de question: bonne lettre}.

    Returns:
        score, total, note_sur_20 et le détail question par question.
    """
    total = len(solution_key)

    # Comparaison insensible à la casse et aux espaces ("b " == "B")
    details = {
        q: {
            "donnee": user_answers.get(q),
            "attendue": rep,
            "correct": (user_answers.get(q) or "").strip().upper() == rep.strip().upper(),
        }
        for q, rep in solution_key.items()
    }

    nb_correct = sum(d["correct"] for d in details.values())
    return {
        "score": nb_correct,
        "total": total,
        "note_sur_20": round(20 * nb_correct / total, 2) if total else 0.0,
        "details": details,
    }


# --------------------------------------------------------------------------- #
# 4. Réponse de secours (SLM indisponible)
# --------------------------------------------------------------------------- #

def format_domain_response(tool_results: dict[str, Any]) -> str:
    """Génère une réponse de secours à partir des résultats de process_domain_logic."""
    summary = tool_results.get("summary", "")

    # Pas de contenu : le résumé suffit, inutile de parler d'indisponibilité
    if tool_results.get("status") == "no_content":
        return summary

    analyzed = tool_results.get("documents_analyzed", 0)
    return (
        "Le générateur de QCM est momentanément indisponible. "
        f"{summary} ({analyzed} extrait(s) de cours trouvé(s))."
    )
