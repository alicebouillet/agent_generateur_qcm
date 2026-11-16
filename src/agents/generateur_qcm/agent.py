"""Starter Code Générique d'Agent Étudiant (LangGraph + RAG + SLM).

Ce fichier sert de point de départ universel pour TOUS les archétypes :
- advisor, auditor, quiz_gen, internship_matcher, workload_estimator, exam_planner.

Directives pour votre équipe :
1. Personnalisez les propriétés 'name' et 'description' ci-dessous.
2. Branchez vos filtres RAG ou vos outils Python déterministes (tools.py).
3. Soignez le prompt système envoyé au SLM local (Ollama 3B).
"""
"""
Agent de Référence : Generateur de QCM.
"""
import os
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.core.base_agent import BaseAgentPlugin
from src.core.ingestion import get_vector_store
from src.core.state import AgentState

from .tools import (
    format_domain_response,
    format_quiz_for_student,
    parse_quiz_output,
    process_domain_logic,
)


class CustomAgentPlugin(BaseAgentPlugin):
    """Agent spécialisé conçu et développé par l'équipe d'étudiants.

    Hérite obligatoirement du contrat d'interface BaseAgentPlugin.
    """

    @property
    def name(self) -> str:
        return "quiz_gen"

    @property
    def description(self) -> str:
        return (
            "Générateur d'évaluations formatives et de QCM d'entraînement. "
            "Génère des quiz à choix multiples fondés sur les compétences visées "
            "et les acquis d'apprentissage d'une UE précise, et évalue les réponses."
        )

    def create_node(self) -> Callable[[AgentState], dict[str, Any]]:
        """Retourne la fonction exécutable (nœud LangGraph) de l'agent."""
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b")

        llm = ChatOllama(base_url=ollama_url, model=model_name, temperature=0.1, num_predict=400)
        vector_store = None

        def node(state: AgentState) -> dict[str, Any]:
            nonlocal vector_store
            if vector_store is None:
                vector_store = get_vector_store(reindex_if_empty=False)

            # Extraction du dernier message reçu
            last_message = state["messages"][-1]
            user_query = str(last_message.content)

            # Étape 1 : RAG ciblé avec filtres de métadonnées (k <= 2 pour SLM 3B)
            # TODO: Ajuster vos filtres ChromaDB (ex: {"code_ue": "..."}, {"semestre": "S8"})
            relevant_docs = []
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 2})
                relevant_docs = retriever.invoke(user_query)
                context_text = "\n\n".join(
                                    f"[Extrait {doc.metadata.get('code_ue', 'UE')} - {doc.metadata.get('section_title', '')}]\n{doc.page_content}"
                                    for doc in relevant_docs
                                )
            except Exception as e :
                context_text = f"Information non disponible (Erreur RAG: {e})"
                relevant_docs = []

            # Étape 2 : Exécution des outils Python déterministes (Règle d'or SLM)
            tool_results = process_domain_logic(user_query, relevant_docs)

            # Pas de contenu de cours : on répond sans appeler le SLM (évite un quiz inventé)
            if tool_results["status"] == "no_content":
                return {
                    "messages": [AIMessage(content=f"[{self.name}] {format_domain_response(tool_results)}")],
                    "current_agent": self.name,
                }


            # Étape 3 : Génération du QCM par le SLM (sortie JSON imposée)
            nb_questions = tool_results["nb_questions"]
            system_prompt = (
                f"Tu es l'agent spécialisé '{self.name}', générateur de QCM d'entraînement.\n\n"
                f"TÂCHE : rédige exactement {nb_questions} question(s) à choix multiples "
                "portant sur le CONTEXTE DU COURS ci-dessous.\n\n"
                "RÈGLES STRICTES :\n"
                "1. Base-toi EXCLUSIVEMENT sur le contexte du cours. N'invente aucune notion.\n"
                "2. Chaque question a exactement 4 options différentes et UNE SEULE bonne réponse.\n"
                "3. 'correct' est la LETTRE de la bonne option : A = 1re, B = 2e, C = 3e, D = 4e.\n"
                "4. Varie la position de la bonne réponse d'une question à l'autre.\n"
                "5. Réponds UNIQUEMENT avec la liste JSON, sans texte avant ni après.\n\n"
                "FORMAT EXACT ATTENDU :\n"
                '[{"question": "Énoncé ?", '
                '"options": ["option 1", "option 2", "option 3", "option 4"], '
                '"correct": "B"}]\n\n'
                f"CONTEXTE DU COURS :\n{context_text}"
            )


            messages = [
                SystemMessage(content=system_prompt),
                last_message,
            ]

            try:
                response = llm.invoke(messages)
            except Exception:
                # Fallback défensif en cas d'indisponibilité temporaire du SLM
                response_text = format_domain_response(tool_results)
            else:
                # Étape 4 : Python vérifie la sortie du SLM avant de l'envoyer à l'étudiant
                quiz = parse_quiz_output(str(response.content))
                if quiz:
                    response_text = format_quiz_for_student(quiz)
                else:
                    response_text = "Je n'ai pas réussi à générer un QCM valide à partir du cours. Réessaie ou reformule ta demande."


            # Préfixage obligatoire avec l'identifiant pour la traçabilité
            final_message = AIMessage(content=f"[{self.name}] {response_text}")

            return {
                "messages": [final_message],
                "current_agent": self.name,
            }

        return node
