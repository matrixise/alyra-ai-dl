#!/usr/bin/env python3
"""
Simple CLI to test Bio_ClinicalBERT + Ollama interaction.

Usage:
    python apps/cli.py
    python apps/cli.py --model models/my-model
    python apps/cli.py --no-llm
    python apps/cli.py --threshold 0.7

Architecture:
    User input -> Bio_ClinicalBERT (prediction) -> Ollama (explanation)
"""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from alyra_ai_dl import (
    DEFAULT_MODEL_PATH,
    create_classifier,
    detect_device,
    predict_with_threshold,
)
from apps.llm_processor import generate_response

app = typer.Typer(help="Medical Symptom Analyzer CLI")
console = Console()


@app.command()
def analyze(
    model: Annotated[
        Path,
        typer.Option(
            "--model",
            "-m",
            help="Path to the trained model",
            exists=True,
            dir_okay=True,
        ),
    ] = DEFAULT_MODEL_PATH,
    threshold: Annotated[
        float,
        typer.Option("--threshold", "-t", help="Confidence threshold (0.0-1.0)"),
    ] = 0.55,
    no_llm: Annotated[
        bool,
        typer.Option("--no-llm", help="Skip LLM response generation (BERT only)"),
    ] = False,
    llm_backend: Annotated[
        str,
        typer.Option(
            "--llm-backend", help="LLM backend to use (ollama, openai, lightning)"
        ),
    ] = "ollama",
    ollama_url: Annotated[
        str,
        typer.Option("--ollama-url", "-u", help="Ollama server URL"),
    ] = "http://localhost:11434",
    ollama_model: Annotated[
        str,
        typer.Option("--ollama-model", help="Ollama model name"),
    ] = "llama3",
):
    """Interactive symptom analyzer using Bio_ClinicalBERT + Ollama."""
    # Load model
    console.print("[bold blue]Loading Bio_ClinicalBERT model...[/bold blue]")

    # Handle model path (model is already a Path from typer)
    model_path = Path.cwd() / model if not model.is_absolute() else model

    # Detect and display device
    device = detect_device()
    console.print(f"[dim]Using device: {device.value}[/dim]")

    # Create classifier
    try:
        classifier = create_classifier(
            model_path=model_path,
            device=device,
            top_k=None,  # Return all probabilities
        )
        console.print("[bold green]✓ Model loaded successfully[/bold green]\n")
    except Exception as e:
        console.print(f"[bold red]Error loading model:[/bold red] {e}")
        console.print(f"[yellow]Model path: {model_path}[/yellow]")
        console.print(
            "[yellow]Make sure the model is a valid transformers model[/yellow]"
        )
        raise typer.Exit(1) from e

    console.print(
        Panel(
            "[bold]Medical Symptom Analyzer[/bold]\n"
            "Type your symptoms and press Enter.\n"
            "Type [bold red]quit[/bold red] to exit.",
            title="Welcome",
            border_style="blue",
        )
    )
    console.print()

    while True:
        try:
            user_input = console.input("[bold green]You:[/bold green] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold]Goodbye![/bold]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            console.print("[bold]Goodbye![/bold]")
            break

        # Step 1: BERT prediction
        console.print("\n[bold cyan][BERT][/bold cyan] Analyzing symptoms...")
        prediction = predict_with_threshold(
            classifier,
            user_input,
            threshold=threshold,
        )

        # Display results in a table
        table = Table(
            title="BERT Analysis", show_header=True, header_style="bold magenta"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Prediction", prediction["disease"])
        table.add_row("Confidence", f"{prediction['confidence']:.1%}")

        console.print(table)

        if "suggestion" in prediction:
            console.print(f"[yellow]Note:[/yellow] {prediction['suggestion']}")

        # Show top probabilities
        if "all_probs" in prediction:
            prob_table = Table(title="All Probabilities", show_header=True)
            prob_table.add_column("Disease", style="cyan")
            prob_table.add_column("Probability", style="green")

            sorted_probs = sorted(
                prediction["all_probs"].items(), key=lambda x: x[1], reverse=True
            )
            for disease, prob in sorted_probs:
                prob_table.add_row(disease, f"{prob:.1%}")

            console.print(prob_table)

        # Step 2: LLM response (optional)
        if not no_llm:
            console.print(
                f"\n[bold cyan][{llm_backend.upper()}][/bold cyan] Generating response..."
            )
            try:
                response = generate_response(
                    user_input,
                    prediction,
                    backend=llm_backend,
                    base_url=ollama_url,
                    model=ollama_model,
                )
                console.print(
                    Panel(
                        response,
                        title="[bold]Assistant[/bold]",
                        border_style="green",
                    )
                )
            except ValueError as e:
                # API key missing
                console.print(f"[bold red][{llm_backend.upper()}] Configuration Error[/bold red]")
                console.print(f"[yellow]{e}[/yellow]")
            except ConnectionError:
                console.print(f"[bold red][{llm_backend.upper()}] Connection Error[/bold red]")
                if llm_backend == "ollama":
                    console.print(
                        f"[yellow]Cannot connect to Ollama at {ollama_url}[/yellow]"
                    )
                    console.print(
                        "[yellow]Make sure Ollama is running: ollama serve[/yellow]"
                    )
                else:
                    console.print(
                        f"[yellow]Cannot connect to {llm_backend} service[/yellow]"
                    )
                    console.print(
                        f"[yellow]Check your {llm_backend.upper()}_API_KEY environment variable[/yellow]"
                    )
            except Exception as e:
                console.print(f"[bold red][{llm_backend.upper()}] Error:[/bold red] {e}")
                if llm_backend == "ollama":
                    console.print(
                        f"[yellow]Check '{ollama_model}' is installed: ollama list[/yellow]"
                    )

        console.print("\n" + "-" * 60 + "\n")


if __name__ == "__main__":
    app()
