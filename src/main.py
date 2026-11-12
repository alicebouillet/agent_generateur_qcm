"""
Point d'entrée principal (CLI interactif).
Permet de tester et de discuter avec le système multi-agents LangGraph en local.
"""

import sys
from typing import cast

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src.core.registry import AgentRegistry
from src.core.state import AgentState
from src.orchestrator.supervisor import build_multiagent_graph
from src.core.observability import get_observability_callbacks

load_dotenv()

# Forcer la gestion UTF-8 sur Windows pour éviter UnicodeEncodeError avec les émojis Rich
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

console = Console(legacy_windows=False if sys.platform == "win32" else None)


def main():
    console.print(
        Panel.fit(
            "[bold cyan]🎓 Système Multi-Agents RAG - Syllabus École d'Ingénieur[/bold cyan]\n"
            "[dim]LangGraph + Ollama (SLMs locaux) + ChromaDB[/dim]",
            border_style="cyan",
        )
    )

    # Initialisation du registre et découverte des agents
    registry = AgentRegistry()
    plugins = registry.discover_and_load_agents()

    console.print(f"\n[green]✔ {len(plugins)} agent(s) découvert(s) :[/green]")
    for name, p in plugins.items():
        console.print(f"  • [bold yellow]{name}[/bold yellow] : [dim]{p.description}[/dim]")

    console.print("\n[blue]⏳ Compilation du graphe LangGraph...[/blue]")
    try:
        app = build_multiagent_graph(registry)
        console.print("[green]✔ Graphe multi-agents compilé avec succès ![/green]\n")
    except Exception as e:
        console.print(f"[bold red]❌ Erreur lors de la compilation du graphe :[/bold red] {e}")
        sys.exit(1)

    console.print("[dim]Tapez votre question (ou 'exit' pour quitter) :[/dim]\n")

    # Boucle de dialogue
    state: AgentState = {
        "messages": [],
        "next_agent": None,
        "current_agent": None,
        "context_data": {},
    }

    while True:
        try:
            user_input = console.input("[bold green]Étudiant > [/bold green]").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("[yellow]Au revoir ![/yellow]")
                break

            state["messages"] = list(state["messages"]) + [HumanMessage(content=user_input)]
            state["next_agent"] = None
            state["current_agent"] = None
            if "context_data" in state:
                state["context_data"].pop("pending_agents", None)
                state["context_data"].pop("visited_agents", None)

            console.print("[dim]🤖 Orchestration en cours...[/dim]")

            # Observabilité en direct (Rich Console + Langfuse si configuré)
            callbacks = get_observability_callbacks(
                trace_name="InteractionEtudiant",
                enable_console=True,
                verbose=True,
            )

            # Exécution du graphe avec garde-fou de récursion et traçage
            final_output = app.invoke(
                state,
                config={"recursion_limit": 10, "callbacks": callbacks},
            )
            state = cast(AgentState, final_output)

            # Affichage du dernier message produit
            if state["messages"]:
                last_msg = state["messages"][-1]
                console.print("\n" + "-" * 50)
                console.print(Markdown(str(last_msg.content)))
                console.print("-" * 50 + "\n")

            # Affichage de l'arbre d'exécution d'observabilité
            for cb in callbacks:
                if hasattr(cb, "render_summary_panel"):
                    console.print(cb.render_summary_panel())

        except KeyboardInterrupt:
            console.print("\n[yellow]Session interrompue.[/yellow]")
            break
        except Exception as err:
            console.print(f"[bold red]Erreur d'exécution :[/bold red] {err}")


if __name__ == "__main__":
    main()
