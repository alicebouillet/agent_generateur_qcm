"""
Module d'Observabilité Détaillée pour Système Multi-Agents LangGraph.
Capture fidèlement les transitions de nœuds LangGraph (filtrage du bruit interne),
les décisions de routage, les extractions RAG et les métriques LLM.
"""

import os
import sys
import time
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree

# Configuration sécurisée pour Windows (évite les erreurs UnicodeEncodeError sur CP1252)
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: S110
        pass

console = Console(legacy_windows=False, safe_box=True)


class DetailedConsoleTraceHandler(BaseCallbackHandler):
    """
    Gestionnaire d'observabilité console de haute précision.
    Filtre le bruit interne de LangChain pour ne conserver QUE les véritables
    étapes du graphe LangGraph (Supervisor, Agents, Outils, RAG, LLM).
    """

    def __init__(self, trace_name: str = "InteractionMultiAgents", verbose: bool = True):
        super().__init__()
        self.trace_name = trace_name
        self.verbose = verbose
        self.start_time = time.perf_counter()
        self.active_runs: dict[UUID, dict[str, Any]] = {}
        self.recorded_steps: list[dict[str, Any]] = []
        self._current_top_step: dict[str, Any] | None = None

    def on_chain_start(
        self,
        serialized: dict[str, Any] | None,
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        metadata = metadata or {}
        tags = tags or []
        node_name = metadata.get("langgraph_node")

        # Règle stricte : Seuls les nœuds étiquetés 'graph:step:X' représentent les étapes du StateGraph
        step_tag = next((t for t in tags if t.startswith("graph:step:")), None)
        if not step_tag or not node_name:
            return

        step_info = {  # type: ignore
            "run_id": run_id,
            "node_name": node_name,
            "step_tag": step_tag,
            "start_time": time.perf_counter(),
            "inputs": inputs,
            "details": [],
            "llm_time": 0.0,
            "tokens": 0,
        }
        self.active_runs[run_id] = step_info
        self._current_top_step = step_info

        if self.verbose:
            step_num = step_tag.replace("graph:step:", "")
            console.print(
                f"[dim]• [Observabilité] Étape {step_num} : Entrée dans le nœud [bold cyan]'{node_name}'[/bold cyan]...[/dim]"
            )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if run_id not in self.active_runs:
            return

        step = self.active_runs.pop(run_id)
        elapsed = time.perf_counter() - step["start_time"]
        step["elapsed"] = elapsed
        step["outputs"] = outputs

        node_name = step["node_name"]
        next_agent = outputs.get("next_agent") if isinstance(outputs, dict) else None
        context_data = outputs.get("context_data", {}) if isinstance(outputs, dict) else {}
        pending = context_data.get("pending_agents", [])

        if node_name == "supervisor":
            if next_agent and next_agent != "FINISH":
                plan_desc = f"Agent cible : [bold yellow]{next_agent}[/bold yellow]"
                if pending:
                    plan_desc += f" (suivi de {pending})"
                step["summary"] = f"Planification : {plan_desc}"
            else:
                visited = context_data.get("visited_agents", [])
                if visited:
                    step["summary"] = (
                        f"Clôture [bold magenta]FINISH[/bold magenta] (Agents intervenus : {visited})"
                    )
                else:
                    step["summary"] = (
                        "Clôture immédiate [bold magenta]FINISH[/bold magenta] (Garde-fou / Hors-sujet)"
                    )
        else:
            # Extraction d'un aperçu de la réponse pour le diagnostic
            preview = ""
            messages = outputs.get("messages", []) if isinstance(outputs, dict) else []
            if messages:
                content = str(messages[-1].content)
                first_line = content.strip().split("\n")[0]
                preview = f" -> {first_line[:80]}"
            step["summary"] = f"Réponse produite ({elapsed:.2f}s){preview}"

        self.recorded_steps.append(step)

        if self.verbose:
            console.print(
                f"[dim]✔ [Observabilité] Nœud [bold]{node_name}[/bold] terminé en "
                f"[bold]{elapsed:.2f}s[/bold] | {step['summary']}[/dim]"
            )

    def on_llm_start(
        self,
        serialized: dict[str, Any] | None,
        prompts: list[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.active_runs[run_id] = {"type": "llm", "start_time": time.perf_counter()}

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if run_id in self.active_runs:
            llm_run = self.active_runs.pop(run_id)
            elapsed = time.perf_counter() - llm_run["start_time"]
            total_tokens = 0
            if response.llm_output and "token_usage" in response.llm_output:
                total_tokens = response.llm_output["token_usage"].get("total_tokens", 0)

            if self._current_top_step:
                self._current_top_step["llm_time"] += elapsed
                self._current_top_step["tokens"] += total_tokens
                desc = f"Inférence LLM : {elapsed:.2f}s"
                if total_tokens:
                    desc += f" (~{total_tokens} tokens)"
                self._current_top_step["details"].append(desc)

    def on_tool_start(
        self,
        serialized: dict[str, Any] | None,
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = kwargs.get("name", "Tool")
        self.active_runs[run_id] = {
            "type": "tool",
            "name": tool_name,
            "input": input_str,
            "start_time": time.perf_counter(),
        }
        if self._current_top_step:
            self._current_top_step["details"].append(
                f"Outil invoqué : '{tool_name}' ({input_str[:60]})"
            )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        if run_id in self.active_runs:
            tool_run = self.active_runs.pop(run_id)
            elapsed = time.perf_counter() - tool_run["start_time"]
            tool_name = tool_run["name"]
            if self._current_top_step:
                self._current_top_step["details"].append(
                    f"Outil '{tool_name}' terminé ({elapsed:.2f}s)"
                )

    def render_summary_panel(self) -> Panel:
        """Génère l'arbre Rich détaillé et explicite des étapes réelles."""
        total_time = time.perf_counter() - self.start_time

        tree = Tree(
            f"[bold cyan]🔍 Trace d'Observabilité LangGraph[/bold cyan] [dim]({total_time:.2f}s total)[/dim]"
        )

        if not self.recorded_steps:
            tree.add("[dim]Aucune étape enregistrée.[/dim]")

        for i, step in enumerate(self.recorded_steps, 1):
            node = step["node_name"]
            elapsed = step["elapsed"]
            summary = step.get("summary", "")

            if node == "supervisor":
                branch = tree.add(
                    f"[bold yellow]Étape {i} : Superviseur[/bold yellow] [dim]({elapsed:.2f}s)[/dim]\n"
                    f"  └── 🔀 [cyan]{summary}[/cyan]"
                )
            else:
                branch = tree.add(
                    f"[bold green]Étape {i} : Agent Expert '{node}'[/bold green] [dim]({elapsed:.2f}s)[/dim]\n"
                    f"  └── 📝 [white]{summary}[/white]"
                )

            for d in step.get("details", []):
                branch.add(f"[dim]• {d}[/dim]")

        return Panel(
            tree,
            title="[bold green]📊 Diagnostic d'Exécution Multi-Agents[/bold green]",
            border_style="cyan",
            expand=False,
        )


def get_observability_callbacks(
    trace_name: str = "MultiAgentExecution",
    session_id: str | None = None,
    user_id: str | None = None,
    enable_console: bool = True,
    verbose: bool = True,
) -> list[BaseCallbackHandler]:
    """
    Instancie et retourne la liste des gestionnaires d'observabilité.
    """
    callbacks: list[BaseCallbackHandler] = []

    if enable_console:
        callbacks.append(DetailedConsoleTraceHandler(trace_name=trace_name, verbose=verbose))

    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

    if langfuse_public_key and langfuse_secret_key:
        try:
            from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

            lf_handler = LangfuseCallbackHandler(
                public_key=langfuse_public_key,
                secret_key=langfuse_secret_key,  # type: ignore
                host=langfuse_host,  # type: ignore
                session_id=session_id,  # type: ignore
                user_id=user_id,  # type: ignore
            )
            callbacks.append(lf_handler)
            if verbose:
                console.print("[dim]✔ [Observabilité] Télémétrie Langfuse connectée.[/dim]")
        except Exception as e:
            if verbose:
                console.print(f"[yellow]⚠️ [Observabilité] Langfuse non connecté : {e}[/yellow]")

    return callbacks
