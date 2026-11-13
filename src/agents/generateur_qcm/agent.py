"""
Agent de Référence : Generateur de QCM.
Sert de modèle architectural pour les étudiants.
Démontre comment implémenter BaseAgentPlugin, interroger ChromaDB et générer une réponse.
"""

import os
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from src.core.base_agent import BaseAgentPlugin
from src.core.ingestion import get_vector_store
from src.core.state import AgentState


class DemoSyllabusExplorerAgent(BaseAgentPlugin):
    """
    Agent spécialisé dans la recherche documentaire et la découverte générale des cours.
    """

    @property
    def name(self) -> str:
        return "syllabus_explorer"

    @property
    def description(self) -> str:
        return (
            "Spécialiste de la recherche factuelle et générale sur les cours : "
            "volumes horaires, enseignants responsables, crédits ECTS et descriptions de cours."
        )

    def create_node(self) -> Callable[[AgentState], dict[str, Any]]:
        # Initialisation du LLM local et du retriever vectoriel
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:3b")

        llm = ChatOllama(base_url=ollama_url, model=model_name, temperature=0.0, num_predict=400)

        # Lazy loading du vector store pour éviter de ralentir le démarrage si non requis
        vector_store = None

        def node(state: AgentState) -> dict[str, Any]:
            nonlocal vector_store
            if vector_store is None:
                vector_store = get_vector_store(reindex_if_empty=False)

            # Dernier message utilisateur
            last_message = state["messages"][-1]
            user_query = str(last_message.content)

            # Étape 1 : Retrieval documentaire dans ChromaDB
            try:
                retriever = vector_store.as_retriever(search_kwargs={"k": 2})
                relevant_docs = retriever.invoke(user_query)
                context_text = "\n\n".join(
                    f"[Extrait {doc.metadata.get('code_ue', 'UE')} - {doc.metadata.get('section_title', '')}]\n{doc.page_content}"
                    for doc in relevant_docs
                )
            except Exception as e:
                context_text = f"Information non disponible (Erreur RAG: {e})"

            # Étape 2 : Génération de la réponse outillée avec prompt adapté au SLM
            system_prompt = f"""Tu es l'agent 'Syllabus Explorer', un assistant pédagogique pour une école d'ingénieur.
Réponds précisément à la question de l'étudiant en te basant EXCLUSIVEMENT sur les extraits de syllabus suivants :

=== EXTRAITS DE SYLLABUS ===
{context_text}
============================

Si l'information n'est pas dans les extraits, indique poliment que le syllabus correspondant ne semble pas mentionner cette information.
Cite systématiquement le code UE concerné (ex: SD_321, SSI_107).
"""

            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_query)]
            )

            return {
                "messages": [AIMessage(content=f"[Agent {self.name}]\n{response.content}")],
                "current_agent": self.name,
            }

        return node
