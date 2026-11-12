"""
Contrat d'interface pour les plugins d'agents spécialisés.
Chaque équipe d'étudiants doit créer une classe héritant de BaseAgentPlugin.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

from langgraph.graph import END, START, StateGraph

from src.core.state import AgentState


class BaseAgentPlugin(ABC):
    """
    Classe de base abstraite (contrat d'interface) pour tout agent spécialisé.

    Pour intégrer son agent dans le système :
    1. Créer un sous-dossier dans `src/agents/<nom_agent>/`
    2. Hériter de `BaseAgentPlugin`
    3. Implémenter les propriétés `name`, `description` et la méthode `create_node()`
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Identifiant unique de l'agent (en minuscules, sans espaces, ex: 'advisor', 'auditor').
        Ce nom sera utilisé par le superviseur pour router les requêtes.
        """

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Description détaillée du rôle et des compétences de l'agent.
        Cette description est directement fournie au prompt du superviseur
        pour lui permettre de choisir judicieusement cet agent.
        """

    @abstractmethod
    def create_node(self) -> Callable[[AgentState], dict[str, Any]]:
        """
        Retourne la fonction exécutable (nœud LangGraph) de l'agent.
        Cette fonction prend en argument l'AgentState et retourne un dictionnaire
        de mise à jour (ex: {'messages': [AIMessage(...)], 'current_agent': self.name}).
        """

    def build_subgraph(self) -> StateGraph:
        """
        Optionnel : Permet à un agent avancé d'exposer un sous-graphe LangGraph complet
        au lieu d'un nœud unique. Par défaut, encapsule le create_node().
        """
        workflow = StateGraph(AgentState)
        workflow.add_node(self.name, cast(Any, self.create_node()))
        workflow.add_edge(START, self.name)
        workflow.add_edge(self.name, END)
        return workflow
