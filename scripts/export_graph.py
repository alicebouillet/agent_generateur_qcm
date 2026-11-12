#!/usr/bin/env python3
"""
📊 Exporteur de Graphe Multi-Agents (Mermaid & PNG)
Génère les visualisations d'architecture du superviseur LangGraph et des sous-agents.

Usage:
    uv run python scripts/export_graph.py [--output FILE] [--include-templates] [--stdout]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Forcer UTF-8 sur Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from src.core.registry import AgentRegistry  # noqa: E402
from src.orchestrator.supervisor import build_multiagent_graph  # noqa: E402

console = Console(legacy_windows=False if sys.platform == "win32" else None)

DEFAULT_OUTPUT_MD = Path("docs/graph_multiagent.md")
DEFAULT_OUTPUT_PNG = Path("docs/graph_multiagent.png")


def export_graph(
    output_md_path: Path = DEFAULT_OUTPUT_MD,
    output_png_path: Path = DEFAULT_OUTPUT_PNG,
    include_templates: bool = False,
    to_stdout: bool = False,
) -> int:
    """
    Découvre les agents, compile le graphe et exporte le schéma Mermaid.
    """
    registry = AgentRegistry()
    registry.discover_and_load_agents()

    if include_templates:
        # Optionnel : charger également les templates d'agents pour prévisualisation
        templates_path = ROOT_DIR / "templates"
        if templates_path.exists():
            import importlib.util

            for t_dir in sorted(templates_path.iterdir()):
                if t_dir.is_dir() and (t_dir / "agent.py").exists():
                    try:
                        agent_file = t_dir / "agent.py"
                        spec = importlib.util.spec_from_file_location(
                            f"templates.{t_dir.name}.agent", agent_file
                        )
                        if spec and spec.loader:
                            mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(mod)
                            for attr_name in dir(mod):
                                attr = getattr(mod, attr_name)
                                if (
                                    isinstance(attr, type)
                                    and hasattr(attr, "name")
                                    and attr_name not in ("BaseAgentPlugin",)
                                ):
                                    try:
                                        instance = attr()
                                        if hasattr(instance, "name") and instance.name:
                                            registry._plugins[instance.name] = instance
                                    except Exception:
                                        pass
                    except Exception as e:
                        console.print(f"[dim yellow]Ignoré {t_dir.name}: {e}[/dim yellow]")

    app = build_multiagent_graph(registry)
    graph = app.get_graph()

    mermaid_raw = graph.draw_mermaid()
    plugin_names = list(registry._plugins.keys())

    # Document markdown complet prêt pour rendu GitHub / VS Code
    markdown_content = f"""# 🗺️ Cartographie Visuelle du Système Multi-Agents

Ce document illustre la topologie dynamique compilée par **LangGraph**.
Chaque agent développé par les équipes étudiantes est automatiquement découvert par le [`AgentRegistry`](../src/core/registry.py) et greffé comme un nœud du graphe orienté.

---

## 📊 Topologie des Nœuds ({len(plugin_names)} agent(s) actif(s))

```mermaid
{mermaid_raw}
```

---

## 🔄 Analyse du Flux d'Exécution & Cycle de Décision

1. **Point d'Entrée (`__start__`)** : La requête utilisateur est injectée dans le dictionnaire d'état partagé [`AgentState`](../src/core/state.py).
2. **Superviseur Central (`supervisor`)** :
   - Évalue la conversation courante via le modèle local (`qwen2.5:3b`).
   - Analyse sémantiquement les descriptions des agents enregistrés.
   - Émet un choix de délégation structuré via son schéma Pydantic `SupervisorDecision`.
3. **Nœuds Agents Spécialisés** ({', '.join(f'`{p}`' for p in plugin_names)}) :
   - Reçoivent l'état, consultent leurs outils métiers ou la base vectorielle ChromaDB.
   - Produisent un message d'analyse préfixé par `[Agent <nom>]`.
   - **Règle d'or LangGraph** : Tout agent retourne systématiquement son message au superviseur pour validation ou clôture.
4. **Sortie Déterministe (`__end__`)** : Lorsque le superviseur estime que la question a reçu une réponse complète ou si un guardrail s'active, il route vers `FINISH` pour terminer l'interaction.
"""

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    console.print(
        Panel.fit(
            f"[bold green]✔ Diagramme Mermaid exporté avec succès dans :[/bold green] {output_md_path}\n"
            f"[dim]Agents représentés : {', '.join(plugin_names)}[/dim]",
            border_style="green",
        )
    )

    # Tentative d'export image PNG si les dépendances le permettent
    try:
        png_bytes = graph.draw_mermaid_png()
        with open(output_png_path, "wb") as f:
            f.write(png_bytes)
        console.print(f"[green]✔ Schéma PNG généré dans :[/green] {output_png_path}")
    except Exception as e:
        console.print(
            f"[dim]ℹ Rendu PNG binaire ignoré ({e}). Le format Markdown/Mermaid reste disponible et rendu nativement sur GitHub/VS Code.[/dim]"
        )

    if to_stdout:
        console.print("\n[bold cyan]Code Mermaid généré :[/bold cyan]\n")
        console.print(Syntax(mermaid_raw, "mermaid", theme="monokai", line_numbers=True))

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Génère et exporte le diagramme Mermaid du système multi-agents LangGraph."
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Chemin du fichier Markdown généré (défaut: docs/graph_multiagent.md).",
    )
    parser.add_argument(
        "--include-templates",
        "-t",
        action="store_true",
        help="Inclut les agents des répertoires templates/ dans la topologie visualisée.",
    )
    parser.add_argument(
        "--stdout",
        "-s",
        action="store_true",
        help="Affiche directement le code Mermaid dans le terminal.",
    )

    args = parser.parse_args()
    sys.exit(
        export_graph(
            output_md_path=args.output,
            include_templates=args.include_templates,
            to_stdout=args.stdout,
        )
    )


if __name__ == "__main__":
    main()
