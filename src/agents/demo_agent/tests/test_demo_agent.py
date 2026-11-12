"""
Tests unitaires pour l'agent DemoSyllabusExplorerAgent.
Vérifie le respect du contrat d'interface et l'exécution du nœud.
"""

from unittest.mock import MagicMock, patch

from langchain_core.messages import HumanMessage

from src.agents.demo_agent.agent import DemoSyllabusExplorerAgent
from src.core.base_agent import BaseAgentPlugin
from src.core.state import AgentState


def test_demo_agent_contract():
    """Vérifie que l'agent respecte les propriétés de BaseAgentPlugin."""
    agent = DemoSyllabusExplorerAgent()
    assert isinstance(agent, BaseAgentPlugin)
    assert agent.name == "syllabus_explorer"
    assert len(agent.description) > 10
    assert callable(agent.create_node())


@patch("src.agents.demo_agent.agent.ChatOllama")
@patch("src.agents.demo_agent.agent.get_vector_store")
def test_demo_agent_node_execution(mock_get_vector_store, mock_chat_ollama):
    """Vérifie l'exécution du nœud avec des mocks (sans avoir besoin d'Ollama actif)."""
    # Configuration du mock vectorstore
    mock_doc = MagicMock()
    mock_doc.metadata = {"code_ue": "SD_321", "section_title": "Description"}
    mock_doc.page_content = "Cours d'apprentissage automatique."

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]

    mock_vs = MagicMock()
    mock_vs.as_retriever.return_value = mock_retriever
    mock_get_vector_store.return_value = mock_vs

    # Configuration du mock LLM
    mock_llm_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Voici les infos sur l'UE SD_321."
    mock_llm_instance.invoke.return_value = mock_response
    mock_chat_ollama.return_value = mock_llm_instance

    agent = DemoSyllabusExplorerAgent()
    node_fn = agent.create_node()

    initial_state: AgentState = {
        "messages": [HumanMessage(content="Parle-moi du cours SD_321")],
        "next_agent": None,
        "current_agent": None,
        "context_data": {},
    }

    result = node_fn(initial_state)

    assert "messages" in result
    assert result["current_agent"] == "syllabus_explorer"
    assert "SD_321" in result["messages"][0].content
