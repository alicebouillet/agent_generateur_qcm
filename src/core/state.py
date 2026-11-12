"""
Schéma de l'état partagé (State) pour le système multi-agents LangGraph.
Tous les nœuds et sous-graphes du système lisent et écrivent dans cet état.
"""

from collections.abc import Sequence
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    État global circulant dans le graphe LangGraph.

    Attributes:
        messages: Historique chronologique des messages (utilisateur, superviseur, agents).
                  Annoté avec add_messages (standard LangGraph) pour concaténer et dédupliquer les messages.
        next_agent: Nom du prochain agent sélectionné par le superviseur ('FINISH' pour terminer).
        current_agent: Nom du dernier agent ayant pris la parole.
        context_data: Dictionnaire libre pour partager des données structurées entre agents
                      (ex: liste de cours identifiés, profil étudiant, métadonnées).
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    next_agent: str | None
    current_agent: str | None
    context_data: dict[str, Any]
