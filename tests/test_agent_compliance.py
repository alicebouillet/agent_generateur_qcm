"""
Test de conformité CI : Vérifie que tous les agents présents dans src/agents/
respectent le contrat BaseAgentPlugin et n'ont pas de régressions structurelles.
"""

from src.core.base_agent import BaseAgentPlugin
from src.core.registry import AgentRegistry


def test_all_registered_agents_comply_with_base_plugin():
    """Vérifie que chaque agent découvert hérite bien de BaseAgentPlugin."""
    registry = AgentRegistry(agents_package_path="src/agents")
    plugins = registry.discover_and_load_agents()

    assert len(plugins) > 0, "Aucun agent n'a été découvert dans src/agents/ !"

    for name, plugin in plugins.items():
        assert isinstance(plugin, BaseAgentPlugin), (
            f"L'agent '{name}' n'hérite pas de BaseAgentPlugin !"
        )
        assert isinstance(plugin.name, str) and len(plugin.name.strip()) > 0, (
            f"L'agent '{name}' a un nom invalide."
        )
        assert plugin.name == plugin.name.lower(), (
            f"Le nom de l'agent '{plugin.name}' doit être en minuscules."
        )
        assert " " not in plugin.name, (
            f"Le nom de l'agent '{plugin.name}' ne doit pas contenir d'espaces."
        )
        assert isinstance(plugin.description, str) and len(plugin.description.strip()) >= 15, (
            f"La description de l'agent '{name}' est trop courte. Décrivez précisément ses compétences pour le superviseur."
        )
        assert callable(plugin.create_node()), (
            f"create_node() de l'agent '{name}' doit retourner une fonction exécutable."
        )


def test_registry_prompt_generation():
    """Vérifie que le prompt généré pour le superviseur est cohérent."""
    registry = AgentRegistry(agents_package_path="src/agents")
    registry.discover_and_load_agents()
    prompt = registry.get_descriptions_prompt()
    plugins = registry.list_plugins()
    assert len(plugins) > 0, "Le registry doit découvrir au moins un agent."
    for p in plugins:
        assert f"**{p.name}**" in prompt
