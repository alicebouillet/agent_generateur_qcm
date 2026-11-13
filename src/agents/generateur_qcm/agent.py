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

from .tools import format_domain_response, process_domain_logic


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
            except Exception:
                context_text = f"Information non disponible (Erreur RAG: {e})"
                relevant_docs = []

            rag_context = (
                "\n---\n".join([doc.page_content for doc in relevant_docs])
                if relevant_docs
                else "Aucun document spécifique trouvé."
            )

            # Étape 2 : Exécution des outils Python déterministes (Règle d'or SLM)
            # TODO: Appeler vos fonctions de tools.py (calculs, graphes, regex, JSON)
            tool_results = process_domain_logic(user_query, relevant_docs)

            # Étape 3 : Synthèse par le SLM avec Prompt Système défensif
            system_prompt = (
                f"Tu es l'agent spécialisé '{self.name}'.\n"
                "Ton rôle est d'apporter une réponse rigoureuse, concise et factuelle à l'étudiant.\n\n"
                "RÈGLES STRICTES :\n"
                "1. Base-toi EXCLUSIVEMENT sur les éléments du Contexte RAG et les Résultats des Outils fournis ci-dessous.\n"
                "2. N'invente aucun cours, code d'UE ou prérequis non présent dans les données.\n"
                "3. Si l'information est absente ou insuffisante, reconnais-le explicitement.\n"
                "4. Reste concis (maximum 3 à 5 phrases claires).\n\n"
                f"CONTEXTE RAG :\n{rag_context}\n\n"
                f"RÉSULTATS DES OUTILS PYTHON :\n{tool_results}"
            )

            messages = [
                SystemMessage(content=system_prompt),
                last_message,
            ]

            try:
                response = llm.invoke(messages)
                response_text = str(response.content)
            except Exception:
                # Fallback défensif en cas d'indisponibilité temporaire du SLM
                response_text = format_domain_response(tool_results)

            # Préfixage obligatoire avec l'identifiant pour la traçabilité
            final_message = AIMessage(content=f"[{self.name}] {response_text}")

            return {
                "messages": [final_message],
                "current_agent": self.name,
            }

        return node
