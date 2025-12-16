#!/usr/bin/env python3
"""
Analyse et visualisation des logs d'entraînement.

Usage:
    .venv/bin/python analyze_training_logs.py models/symptom_classifier
    .venv/bin/python analyze_training_logs.py --compare models/symptom_classifier models/symptom_classifier_combined
    .venv/bin/python analyze_training_logs.py models/symptom_classifier --output analysis_plots
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def load_training_history(model_path: Path) -> list[dict]:
    """
    Load training history from a model directory.

    Tries to load from:
    1. metrics/training_history.json (if available)
    2. checkpoint-*/trainer_state.json (latest checkpoint)

    Args:
        model_path: Path to the model directory

    Returns:
        List of log entries

    Raises:
        FileNotFoundError: If no training history is found
    """
    # Try training_history.json first
    history_file = model_path / "metrics" / "training_history.json"
    if history_file.exists():
        with open(history_file) as f:
            return json.load(f)

    # Fall back to latest checkpoint
    checkpoints = sorted(model_path.glob("checkpoint-*"))
    if checkpoints:
        trainer_state_file = checkpoints[-1] / "trainer_state.json"
        if trainer_state_file.exists():
            with open(trainer_state_file) as f:
                state = json.load(f)
                return state.get("log_history", [])

    raise FileNotFoundError(
        f"No training history found in {model_path}\n"
        f"Looked for:\n"
        f"  - {history_file}\n"
        f"  - checkpoint-*/trainer_state.json"
    )


def extract_metrics(history: list[dict]) -> dict[str, pd.DataFrame]:
    """
    Extract different metric types from training history.

    Args:
        history: Raw log history

    Returns:
        Dictionary with DataFrames for train and eval metrics
    """
    train_logs = []
    eval_logs = []

    for entry in history:
        if "loss" in entry:  # Training step
            train_logs.append(entry)
        elif "eval_loss" in entry:  # Evaluation step
            eval_logs.append(entry)

    return {
        "train": pd.DataFrame(train_logs) if train_logs else pd.DataFrame(),
        "eval": pd.DataFrame(eval_logs) if eval_logs else pd.DataFrame(),
    }


def plot_loss_curves(metrics: dict[str, pd.DataFrame], output_path: Path, model_name: str = "Model"):
    """Plot training and validation loss curves."""
    _fig, ax = plt.subplots(figsize=(12, 6))

    if not metrics["train"].empty and "loss" in metrics["train"].columns:
        ax.plot(
            metrics["train"]["step"],
            metrics["train"]["loss"],
            label="Training Loss",
            alpha=0.7,
            linewidth=1.5,
        )

    if not metrics["eval"].empty and "eval_loss" in metrics["eval"].columns:
        ax.plot(
            metrics["eval"]["step"],
            metrics["eval"]["eval_loss"],
            label="Validation Loss",
            marker="o",
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Training and Validation Loss - {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"  ✓ Loss curves saved to: {output_path / 'loss_curves.png'}")


def plot_learning_rate(metrics: dict[str, pd.DataFrame], output_path: Path, model_name: str = "Model"):
    """Plot learning rate schedule."""
    if metrics["train"].empty or "learning_rate" not in metrics["train"].columns:
        console.print("  ⊘ Learning rate data not available")
        return

    _fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        metrics["train"]["step"],
        metrics["train"]["learning_rate"],
        color="green",
        linewidth=2,
    )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Learning Rate", fontsize=12)
    ax.set_title(f"Learning Rate Schedule - {model_name}", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path / "learning_rate.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"  ✓ Learning rate plot saved to: {output_path / 'learning_rate.png'}")


def plot_metrics(metrics: dict[str, pd.DataFrame], output_path: Path, model_name: str = "Model"):
    """Plot validation metrics (accuracy, F1, precision, recall)."""
    if metrics["eval"].empty:
        console.print("  ⊘ Validation metrics not available")
        return

    eval_metrics = [
        "eval_accuracy",
        "eval_f1_macro",
        "eval_precision_macro",
        "eval_recall_macro",
    ]
    available_metrics = [m for m in eval_metrics if m in metrics["eval"].columns]

    if not available_metrics:
        console.print("  ⊘ No standard validation metrics found")
        return

    _fig, ax = plt.subplots(figsize=(12, 6))

    for metric in available_metrics:
        label = metric.replace("eval_", "").replace("_", " ").title()
        ax.plot(
            metrics["eval"]["step"],
            metrics["eval"][metric],
            marker="o",
            label=label,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"Validation Metrics - {model_name}", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path / "validation_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"  ✓ Validation metrics plot saved to: {output_path / 'validation_metrics.png'}")


def print_summary_table(history: list[dict], model_name: str = "Model"):
    """Print summary statistics of training."""
    metrics_data = extract_metrics(history)

    console.print(f"\n[bold cyan]Training Summary - {model_name}[/bold cyan]")

    table = Table(title="Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    # Total steps
    if not metrics_data["train"].empty:
        total_steps = metrics_data["train"]["step"].max()
        table.add_row("Total Training Steps", f"{int(total_steps):,}")

    # Best validation metrics
    if not metrics_data["eval"].empty:
        eval_df = metrics_data["eval"]

        if "eval_loss" in eval_df.columns:
            best_loss = eval_df["eval_loss"].min()
            best_loss_step = eval_df.loc[eval_df["eval_loss"].idxmin(), "step"]
            table.add_row("Best Validation Loss", f"{best_loss:.4f} (step {int(best_loss_step)})")

        if "eval_accuracy" in eval_df.columns:
            best_acc = eval_df["eval_accuracy"].max()
            best_acc_step = eval_df.loc[eval_df["eval_accuracy"].idxmax(), "step"]
            table.add_row("Best Accuracy", f"{best_acc:.4f} (step {int(best_acc_step)})")

        if "eval_f1_macro" in eval_df.columns:
            best_f1 = eval_df["eval_f1_macro"].max()
            best_f1_step = eval_df.loc[eval_df["eval_f1_macro"].idxmax(), "step"]
            table.add_row("Best F1 Macro", f"{best_f1:.4f} (step {int(best_f1_step)})")

        # Final metrics
        final_row = eval_df.iloc[-1]
        table.add_row("─" * 25, "─" * 30)
        table.add_row("Final Validation Loss", f"{final_row.get('eval_loss', 0):.4f}")

        if "eval_accuracy" in final_row:
            table.add_row("Final Accuracy", f"{final_row['eval_accuracy']:.4f}")
        if "eval_f1_macro" in final_row:
            table.add_row("Final F1 Macro", f"{final_row['eval_f1_macro']:.4f}")

    console.print(table)


def compare_runs(model_paths: list[Path], output_path: Path):
    """Compare multiple training runs."""
    console.print("\n[bold magenta]Comparing Multiple Runs[/bold magenta]")

    _fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    colors = ["blue", "red", "green", "orange", "purple"]
    comparison_data = []

    for idx, model_path in enumerate(model_paths):
        try:
            history = load_training_history(model_path)
            metrics = extract_metrics(history)
            model_name = model_path.name
            color = colors[idx % len(colors)]

            # Training loss
            if not metrics["train"].empty and "loss" in metrics["train"].columns:
                axes[0, 0].plot(
                    metrics["train"]["step"],
                    metrics["train"]["loss"],
                    label=model_name,
                    alpha=0.7,
                    color=color,
                )

            # Validation loss
            if not metrics["eval"].empty and "eval_loss" in metrics["eval"].columns:
                axes[0, 1].plot(
                    metrics["eval"]["step"],
                    metrics["eval"]["eval_loss"],
                    label=model_name,
                    marker="o",
                    color=color,
                )

            # Accuracy
            if not metrics["eval"].empty and "eval_accuracy" in metrics["eval"].columns:
                axes[1, 0].plot(
                    metrics["eval"]["step"],
                    metrics["eval"]["eval_accuracy"],
                    label=model_name,
                    marker="o",
                    color=color,
                )

            # F1 Score
            if not metrics["eval"].empty and "eval_f1_macro" in metrics["eval"].columns:
                axes[1, 1].plot(
                    metrics["eval"]["step"],
                    metrics["eval"]["eval_f1_macro"],
                    label=model_name,
                    marker="o",
                    color=color,
                )

            # Collect final metrics for comparison table
            if not metrics["eval"].empty:
                final = metrics["eval"].iloc[-1]
                comparison_data.append(
                    {
                        "Model": model_name,
                        "Final Loss": final.get("eval_loss", None),
                        "Final Accuracy": final.get("eval_accuracy", None),
                        "Final F1": final.get("eval_f1_macro", None),
                    }
                )

        except FileNotFoundError as e:
            console.print(f"[yellow]Warning: {e}[/yellow]")
            continue

    # Configure subplots
    axes[0, 0].set_title("Training Loss", fontsize=12)
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Validation Loss", fontsize=12)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Validation Accuracy", fontsize=12)
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1.05)

    axes[1, 1].set_title("Validation F1 Score", fontsize=12)
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("F1 Macro")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(output_path / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    console.print(f"\n  ✓ Comparison plot saved to: {output_path / 'comparison.png'}")

    # Print comparison table
    if comparison_data:
        table = Table(title="Model Comparison - Final Metrics")
        table.add_column("Model", style="cyan")
        table.add_column("Loss", justify="right", style="yellow")
        table.add_column("Accuracy", justify="right", style="green")
        table.add_column("F1 Macro", justify="right", style="magenta")

        for row in comparison_data:
            table.add_row(
                row["Model"],
                f"{row['Final Loss']:.4f}" if row["Final Loss"] is not None else "N/A",
                f"{row['Final Accuracy']:.4f}" if row["Final Accuracy"] is not None else "N/A",
                f"{row['Final F1']:.4f}" if row["Final F1"] is not None else "N/A",
            )

        console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize training logs from Bio_ClinicalBERT models")
    parser.add_argument(
        "model_paths",
        nargs="+",
        type=Path,
        help="Path(s) to model directory/directories",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("analysis_plots"),
        help="Output directory for plots (default: analysis_plots)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare multiple models (requires 2+ model paths)",
    )

    args = parser.parse_args()

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    console.print("\n[bold cyan]Training Log Analysis[/bold cyan]")
    console.print("=" * 60)

    # Comparison mode
    if args.compare or len(args.model_paths) > 1:
        if len(args.model_paths) < 2:
            console.print("[red]Error: Comparison requires at least 2 model paths[/red]")
            return 1

        compare_runs(args.model_paths, args.output)

    # Single model analysis
    else:
        model_path = args.model_paths[0]

        if not model_path.exists():
            console.print(f"[red]Error: Model path not found: {model_path}[/red]")
            return 1

        console.print(f"\n[green]Analyzing model:[/green] {model_path}")

        try:
            # Load history
            history = load_training_history(model_path)
            console.print(f"  Loaded {len(history)} log entries")

            # Extract metrics
            metrics = extract_metrics(history)

            # Print summary
            print_summary_table(history, model_path.name)

            # Generate plots
            console.print("\n[bold]Generating plots...[/bold]")
            plot_loss_curves(metrics, args.output, model_path.name)
            plot_learning_rate(metrics, args.output, model_path.name)
            plot_metrics(metrics, args.output, model_path.name)

            console.print("\n[bold green]✓ Analysis complete![/bold green]")
            console.print(f"[dim]Plots saved to: {args.output}[/dim]")

        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1

    return 0


if __name__ == "__main__":
    exit(main())
