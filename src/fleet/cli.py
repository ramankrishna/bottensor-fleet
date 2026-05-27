"""fleet — bottensor-fleet command-line interface."""
from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="fleet",
    help="bottensor-fleet — graph-native multi-agent runtime.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        from importlib.metadata import version as _pkg_version
        typer.echo(f"bottensor-fleet {_pkg_version('bottensor-fleet')}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", callback=_version_callback, is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Graph-native multi-agent fleet."""

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_BACKEND_ALIASES: dict[str, str] = {
    "anthropic": "claude",
    "claude":    "claude",
    "openai":    "openai",
    "gpt":       "openai",
    "ollama":    "ollama",
    "mlx":       "mlx",
}


def _resolve_backend(prefix: str) -> str:
    """Map user-facing provider prefixes to polyrt backend names."""
    return _BACKEND_ALIASES.get(prefix.lower(), prefix.lower())


def _load_graph_from(path_or_module: str):  # type: ignore[return]
    """Load a Graph or CompiledGraph from a file path or dotted module name."""
    import importlib
    import importlib.util
    import os

    if path_or_module.endswith(".py") or "/" in path_or_module or os.sep in path_or_module:
        spec = importlib.util.spec_from_file_location(
            "_fleet_dyn", os.path.abspath(path_or_module)
        )
        if spec is None or spec.loader is None:
            raise typer.BadParameter(f"Cannot load file: {path_or_module!r}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        mod = importlib.import_module(path_or_module)

    if hasattr(mod, "create_graph"):
        return mod.create_graph()
    if hasattr(mod, "graph"):
        return mod.graph
    raise typer.BadParameter(
        f"{path_or_module!r} must export 'graph: Graph' or 'create_graph() -> Graph'"
    )


# ---------------------------------------------------------------------------
# fleet new <name>
# ---------------------------------------------------------------------------

_GRAPH_TEMPLATE = '''\
"""Graph: {name} — scaffolded by `fleet new {name}`."""
from __future__ import annotations

import asyncio

from fleet import Agent, Graph
from fleet.core.state import GraphState

# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

agent = Agent(
    name="agent",
    goal="Complete the user goal.",
    model="anthropic/claude-sonnet-4-6",  # or openai/gpt-4o, ollama/llama3, …
    tools=["web_search"],                  # remove if no tools needed
)

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

graph = (
    Graph("{name}")
    .add_node("agent", agent.step)
    .set_entry("agent")
    .set_exit("agent")
    .compile()
)

# ---------------------------------------------------------------------------
# Run directly: python {name}.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    state = GraphState(goal="What is the capital of France?")
    final = asyncio.run(graph.run(state))
    print(final.messages[-1].content)
'''


@app.command()
def new(name: str = typer.Argument(..., help="Name of the new graph")) -> None:
    """Scaffold a new graph file from the built-in template."""
    import os

    filename = f"{name}.py"
    if os.path.exists(filename):
        console.print(f"[yellow]File already exists:[/yellow] {filename}")
        raise typer.Exit(1)

    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(_GRAPH_TEMPLATE.format(name=name))

    console.print(f"[green]Created[/green] {filename}")
    console.print(f"  Edit it, then run:  [bold]fleet run {filename} --goal '...'[/bold]")


# ---------------------------------------------------------------------------
# fleet run <graph_file> --goal "..." [--backend sqlite|redis]
# ---------------------------------------------------------------------------

@app.command()
def run(
    graph_file: str = typer.Argument(..., help="Graph file path or dotted module"),
    goal: str = typer.Option(..., "--goal", "-g", help="Top-level objective"),
    backend: str = typer.Option("sqlite", "--backend", "-b", help="Checkpoint backend"),
) -> None:
    """Run a graph to completion, streaming progress to the terminal."""
    import asyncio
    import os
    from importlib.metadata import version as _pkg_version

    from fleet.core.graph import Graph
    from fleet.core.scheduler import EventBus
    from fleet.core.state import GraphState

    async def _run() -> None:
        raw = _load_graph_from(graph_file)
        cg = raw.compile(backend=backend) if isinstance(raw, Graph) else raw

        bus = EventBus()
        cg._event_bus = bus

        async def _on_event(evt: dict) -> None:
            node = evt.get("node", "")
            step = evt.get("step", "")
            evt_type = evt.get("type", "?")
            console.print(f"  [dim]{step:>3}[/dim]  [cyan]{evt_type}[/cyan]  {node}")

        bus.subscribe(_on_event)

        state = GraphState(
            goal=goal,
            metadata={
                "graph_module": graph_file,
                "graph_source": os.path.abspath(graph_file),
                "fleet_version": _pkg_version("bottensor-fleet"),
            },
        )
        console.print(f"[bold]graph[/bold]   {graph_file}")
        console.print(f"[bold]goal[/bold]    {goal}")
        console.print(f"[bold]backend[/bold] {backend}")
        console.rule()

        result = await cg.run(state)

        console.rule()
        run_id = result.metadata.get("run_id", "—")
        console.print(f"[green]✓ done[/green]   run_id={run_id}  messages={len(result.messages)}")

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# fleet ui [--port 8765]
# ---------------------------------------------------------------------------

@app.command()
def ui(port: int = typer.Option(8765, "--port", "-p", help="Listen port")) -> None:
    """Launch the Fleet API server and open the browser UI."""
    import threading
    import time
    import webbrowser

    import uvicorn

    from fleet.server.app import create_app

    url = f"http://localhost:{port}"
    console.print(f"[bold green]Fleet UI[/bold green] → {url}")
    console.print("Press [bold]Ctrl-C[/bold] to stop.\n")

    def _open_browser() -> None:
        time.sleep(1.2)
        webbrowser.open(url)

    threading.Thread(target=_open_browser, daemon=True).start()
    uvicorn.run(create_app(), host="0.0.0.0", port=port, log_level="info")


# ---------------------------------------------------------------------------
# fleet add-agent <graph_file> --name X --model anthropic/claude-sonnet-4-6
# ---------------------------------------------------------------------------

_AGENT_SNIPPET = '''

# ── Agent: {name}  (model: {backend}/{model}) ─────────────────────────────
from fleet.agents.agent import Agent
from fleet.providers.client import FleetLLM
from fleet.core.state import GraphState as _GS

_llm_{name} = FleetLLM("{backend}", "{model}")
_agent_{name} = Agent(_llm_{name})


async def {name}(state: _GS) -> _GS:
    """Agent node: {name}."""
    return await _agent_{name}.step(state)
'''


@app.command("add-agent")
def add_agent(
    graph_file: str = typer.Argument(..., help="Graph file to append to"),
    name: str = typer.Option(..., "--name", "-n", help="Node / function name"),
    model: str = typer.Option(
        "anthropic/claude-sonnet-4-6", "--model", "-m", help="backend/model identifier"
    ),
) -> None:
    """Append an Agent node to an existing graph file."""
    if "/" in model:
        raw_backend, model_name = model.split("/", 1)
    else:
        raw_backend, model_name = model, model

    backend_name = _resolve_backend(raw_backend)
    snippet = _AGENT_SNIPPET.format(name=name, backend=backend_name, model=model_name)
    with open(graph_file, "a", encoding="utf-8") as fh:
        fh.write(snippet)

    console.print(f"[green]Appended[/green] agent '{name}' to {graph_file}")
    console.print(
        f"  Add it to your graph:  "
        f"[bold].add_node('{name}', {name})[/bold]"
    )


# ---------------------------------------------------------------------------
# fleet ls
# ---------------------------------------------------------------------------

@app.command("ls")
def list_runs() -> None:
    """List past runs stored in the default SQLite checkpoint."""
    import asyncio

    from rich.table import Table

    from fleet.core.checkpoint import SQLiteCheckpoint

    async def _list() -> list[dict[str, str]]:
        return await SQLiteCheckpoint().list_runs_rich()

    rows = asyncio.run(_list())
    if not rows:
        console.print("[dim]No past runs found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Run ID", style="cyan", no_wrap=True)
    table.add_column("Saved at", style="dim", no_wrap=True)
    table.add_column("Goal")
    for r in rows:
        saved = r["saved_at"][:19].replace("T", " ")  # ISO to "YYYY-MM-DD HH:MM:SS"
        goal = r["goal"][:60] + ("…" if len(r["goal"]) > 60 else "")
        table.add_row(r["run_id"], saved, goal)
    console.print(table)


# ---------------------------------------------------------------------------
# fleet replay <run_id> [--backend sqlite|redis]
# ---------------------------------------------------------------------------

@app.command()
def replay(
    run_id: str = typer.Argument(..., help="Run ID to replay"),
    goal: str | None = typer.Option(None, "--goal", help="Override the original goal"),
) -> None:
    """Re-run a previous graph using its persisted source path."""
    import asyncio
    import importlib.util
    from pathlib import Path

    from fleet.core.checkpoint import SQLiteCheckpoint
    from fleet.core.state import GraphState

    async def _replay() -> None:
        ckpt = SQLiteCheckpoint()
        state = await ckpt.load(run_id)
        if state is None:
            typer.echo(f"Run not found: {run_id}", err=True)
            raise typer.Exit(1)

        source_path = state.metadata.get("graph_source")
        if not source_path:
            typer.echo(
                f"Run {run_id} has no graph_source metadata (created with an older fleet "
                "version). Cannot replay.",
                err=True,
            )
            raise typer.Exit(1)

        if not Path(source_path).exists():
            typer.echo(f"Graph source file no longer exists: {source_path}", err=True)
            raise typer.Exit(1)

        spec = importlib.util.spec_from_file_location("_replay_module", source_path)
        if spec is None or spec.loader is None:
            typer.echo(f"Could not load module from {source_path}", err=True)
            raise typer.Exit(1)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        graph = getattr(module, "graph", None)
        if graph is None:
            typer.echo(
                f"Module {source_path} has no top-level `graph` variable. "
                "Replay requires the convention `graph = (...).compile()`.",
                err=True,
            )
            raise typer.Exit(1)

        original_goal = state.goal
        new_state = GraphState(goal=goal or original_goal)
        new_state.metadata["replayed_from"] = run_id

        typer.echo(f"Replaying {run_id} (goal: {new_state.goal!r})")
        final = await graph.run(new_state)
        typer.echo(f"\nReplay complete. New run_id: {final.metadata.get('run_id')}")
        if final.messages:
            typer.echo(f"\nFinal output:\n{final.messages[-1].content}")

    asyncio.run(_replay())
