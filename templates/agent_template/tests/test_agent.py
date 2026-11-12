"""Tests unitaires avec Mocking pour l'Agent Étudiant.

Ces tests s'exécutent en < 100 ms via pytest, sans GPU ni connexion réseau.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from ..agent import CustomAgentPlugin
from ..tools import format_domain_response, process_domain_logic


@pytest.fixture
def agent():
    return CustomAgentPlugin()


def test_plugin_properties(agent):
    """Vérifie le respect du contrat BaseAgentPlugin."""
    assert isinstance(agent.name, str) and len(agent.name.strip()) > 0
    assert agent.name == agent.name.lower(), "Le nom doit être en minuscules"
    assert " " not in agent.name, "Le nom ne doit pas contenir d'espaces"
    assert len(agent.description) >= 15, "La description doit comporter au moins 15 caractères"
    assert callable(agent.create_node()), "create_node() doit retourner une fonction"


def test_deterministic_tools():
    """Vérifie le comportement de l'outil déterministe."""
    mock_docs = [MagicMock(page_content="Cours IA")]
    res = process_domain_logic("requête test", mock_docs)
    assert isinstance(res, dict)
    assert res["status"] == "success"
    assert res["documents_analyzed"] == 1
    fallback = format_domain_response(res)
    assert "1 document(s)" in fallback


@patch("templates.agent_template.agent.ChatOllama")
@patch("templates.agent_template.agent.get_vector_store")
def test_agent_node_execution(mock_get_vector_store, mock_chat_ollama, agent):
    """Vérifie le flux d'exécution du nœud LangGraph avec mock du SLM et de ChromaDB."""
    # Mocking du Document ChromaDB
    mock_doc = MagicMock()
    mock_doc.metadata = {"code_ue": "IA_101", "section_title": "Description"}
    mock_doc.page_content = "Notions fondamentales d'intelligence artificielle."

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc]

    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = mock_retriever
    mock_get_vector_store.return_value = mock_vector_store

    # Mocking du LLM Ollama
    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = AIMessage(content="Réponse synthétisée par l'agent.")
    mock_chat_ollama.return_value = mock_llm_instance

    node = agent.create_node()
    state = {
        "messages": [HumanMessage(content="Bonjour, pouvez-vous m'aider ?")],
        "next_agent": None,
        "current_agent": None,
        "context_data": {},
    }

    output = node(state)

    assert "messages" in output
    assert len(output["messages"]) == 1
    assert output["current_agent"] == agent.name
    assert f"[{agent.name}]" in output["messages"][0].content
