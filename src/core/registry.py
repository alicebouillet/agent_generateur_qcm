"""
Registre dynamique des agents.
Découvre automatiquement les sous-dossiers dans `src/agents/`
et instancie les classes qui implémentent BaseAgentPlugin.
"""

import importlib
import inspect
from pathlib import Path

from src.core.base_agent import BaseAgentPlugin


class AgentRegistry:
    """
    Gestionnaire centralisé pour la découverte, le chargement
    et l'exposition des plugins d'agents spécialisés.
    """

    def __init__(self, agents_dir: str = "src/agents", agents_package_path: str | None = None):
        self.agents_dir = Path(agents_package_path or agents_dir)
        self._plugins: dict[str, BaseAgentPlugin] = {}

    def discover_and_load_agents(self) -> dict[str, BaseAgentPlugin]:
        """
        Scanne le dossier des agents et charge toutes les classes héritant de BaseAgentPlugin.
        """
        if not self.agents_dir.exists():
            return self._plugins

        for agent_folder in self.agents_dir.iterdir():
            if agent_folder.is_dir() and not agent_folder.name.startswith((".", "_")):
                self._load_agent_from_folder(agent_folder)

        return self._plugins

    def _load_agent_from_folder(self, folder: Path):
        """
        Tente de charger un agent depuis son module `agent.py` ou `__init__.py`.
        """
        module_path = None
        if (folder / "agent.py").exists():
            module_path = f"src.agents.{folder.name}.agent"
        elif (folder / "__init__.py").exists():
            module_path = f"src.agents.{folder.name}"

        if not module_path:
            return

        try:
            module = importlib.import_module(module_path)
            self._register_classes_from_module(module)
        except Exception as e:
            print(f"❌ Erreur lors du chargement de l'agent '{folder.name}': {e}")

    def _register_classes_from_module(self, module):
        for _, attr in inspect.getmembers(module):
            if (
                inspect.isclass(attr)
                and issubclass(attr, BaseAgentPlugin)
                and attr is not BaseAgentPlugin
            ):
                instance = attr()
                if instance.name in self._plugins:
                    if type(self._plugins[instance.name]) is attr:
                        continue
                    print(f"⚠️ Conflit détecté pour le nom d'agent '{instance.name}'. Remplacement.")
                self._plugins[instance.name] = instance

    def get_plugin(self, name: str) -> BaseAgentPlugin | None:
        return self._plugins.get(name)

    def list_plugins(self) -> list[BaseAgentPlugin]:
        return list(self._plugins.values())

    def get_descriptions_prompt(self) -> str:
        """
        Génère la documentation textuelle de tous les agents disponibles
        pour le prompt système du superviseur.
        """
        if not self._plugins:
            return "Aucun agent spécialisé n'est enregistré pour le moment."

        descriptions = []
        for name, plugin in self._plugins.items():
            descriptions.append(f"- **{name}** : {plugin.description}")
        return "\n".join(descriptions)
