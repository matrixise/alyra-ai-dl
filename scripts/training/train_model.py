#!/usr/bin/env python3
"""
Script d'entraînement pour Bio_ClinicalBERT sur la classification de maladies.

Usage:
    .venv/bin/python diseases_dl/scripts/train_model.py "panic disorder,pneumonia,cystitis"
    .venv/bin/python diseases_dl/scripts/train_model.py "anxiety,depression" --epochs 5 --augment
    .venv/bin/python diseases_dl/scripts/train_model.py "trouble panique,pneumonie,cystite" --language fr
"""

import json
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

warnings.filterwarnings("ignore")

app = typer.Typer(
    help="Train Bio_ClinicalBERT for disease classification from symptoms"
)
console = Console()

# Default paths
DEFAULT_DATA_PATH = Path(__file__).parent.parent / "data" / "diseases_symptoms.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "models" / "symptom_classifier"


# ============================================================================
# Data Augmentation Functions
# ============================================================================


def shuffle_symptoms(text: str) -> str:
    """Randomly shuffle the order of symptoms."""
    symptoms = [s.strip() for s in text.split(",")]
    random.shuffle(symptoms)
    return ", ".join(symptoms)


def drop_random_symptom(text: str, p: float = 0.2) -> str:
    """Randomly drop symptoms with probability p."""
    symptoms = [s.strip() for s in text.split(",")]
    if len(symptoms) <= 1:
        return text
    kept = [s for s in symptoms if random.random() > p]
    if not kept:  # Keep at least one
        kept = [random.choice(symptoms)]
    return ", ".join(kept)


def get_templates(template_file: Path) -> list[str]:
    """
    Get augmentation templates from a template file.

    Args:
        template_file: Path to the template JSON file

    Returns:
        List of template strings

    Raises:
        FileNotFoundError: If the template file doesn't exist
        ValueError: If the JSON file is invalid
    """
    if not template_file.exists():
        raise FileNotFoundError(f"Template file not found: {template_file}")

    try:
        with open(template_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data["templates"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid template file {template_file}: {e}")


def add_template(text: str, template_file: Path) -> str:
    """Add natural language template around symptoms."""
    templates = get_templates(template_file)
    template = random.choice(templates)
    return template.format(text)


def augment_example(
    text: str, label: int, n_augmentations: int = 3, template_file: Path | None = None
) -> list:
    """Generate augmented versions of an example."""
    augmented = []
    for _ in range(n_augmentations):
        aug_text = text
        # Apply augmentations with some probability
        if random.random() > 0.3:
            aug_text = shuffle_symptoms(aug_text)
        if random.random() > 0.5:
            aug_text = drop_random_symptom(aug_text)
        if random.random() > 0.5 and template_file:
            aug_text = add_template(aug_text, template_file)
        augmented.append({"symptoms": aug_text, "label": label})
    return augmented


# ============================================================================
# Helper Functions
# ============================================================================


def create_hf_dataset(df: pd.DataFrame) -> HFDataset:
    """Convert DataFrame to HuggingFace Dataset."""
    return HFDataset.from_dict(
        {
            "text": df["symptoms"].tolist(),
            "label": df["label"].tolist(),
        }
    )


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
        "precision_macro": precision_score(labels, predictions, average="macro"),
        "recall_macro": recall_score(labels, predictions, average="macro"),
    }


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list,
    output_path: Path,
):
    """Save confusion matrix as PNG."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
# Main Training Function
# ============================================================================


@app.command()
def train(
    diseases: str = typer.Argument(
        ..., help="Comma-separated list of diseases to classify"
    ),
    data_path: Path = typer.Option(
        DEFAULT_DATA_PATH,
        "--data-path",
        "-d",
        help="Path to CSV file with diseases and symptoms",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directory to save trained model",
    ),
    model_name: str = typer.Option(
        "emilyalsentzer/Bio_ClinicalBERT",
        "--model-name",
        "-m",
        help="Pretrained model name from HuggingFace",
    ),
    epochs: int = typer.Option(3, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(
        16, "--batch-size", "-b", help="Training batch size"
    ),
    learning_rate: float = typer.Option(
        2e-5, "--learning-rate", "-lr", help="Learning rate"
    ),
    max_length: int = typer.Option(
        128, "--max-length", help="Maximum token sequence length"
    ),
    augment: bool = typer.Option(
        True, "--augment/--no-augment", help="Enable data augmentation"
    ),
    n_augmentations: int = typer.Option(
        3, "--n-augmentations", help="Number of augmentations per example"
    ),
    test_size: float = typer.Option(
        0.15, "--test-size", help="Test set proportion (0-1)"
    ),
    val_size: float = typer.Option(
        0.15, "--val-size", help="Validation set proportion (0-1)"
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility"),
    device: str = typer.Option(
        "auto", "--device", help="Device to use (auto/cuda/mps/cpu)"
    ),
    template_file: Path | None = typer.Option(
        None,
        "--template-file",
        "-t",
        help="Path to augmentation template JSON file (e.g., config/augmentation_templates_fr_physician.json). If not specified, no templates will be applied.",
    ),
    save_confusion_matrix_flag: bool = typer.Option(
        False,
        "--save-confusion-matrix/--no-save-confusion-matrix",
        help="Save confusion matrix during training (requires matplotlib)",
    ),
) -> None:
    """
    Train Bio_ClinicalBERT for disease classification from symptom descriptions.

    The model is fine-tuned on a dataset of symptom-disease pairs and saved
    for later inference.
    """
    # ========================================================================
    # 1. Setup
    # ========================================================================
    console.print("\n[bold cyan]Bio_ClinicalBERT Training Pipeline[/bold cyan]")
    console.print("=" * 60)

    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Parse diseases
    target_diseases = [d.strip() for d in diseases.split(",")]

    # Validate: need at least 2 diseases for classification
    if len(target_diseases) < 2:
        console.print("[red]Error: Classification requires at least 2 diseases[/red]")
        console.print("[yellow]Tip: Provide comma-separated diseases like:[/yellow]")
        console.print('  "panic disorder,pneumonia,cystitis"')
        raise typer.Exit(1)

    # Validate template file if provided
    if template_file:
        try:
            templates = get_templates(template_file)
            console.print(f"[yellow]Template file:[/yellow] {template_file}")
            console.print(f"[yellow]Loaded {len(templates)} templates[/yellow]")
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)
    else:
        console.print(
            "[yellow]No template file specified - templates will not be applied during augmentation[/yellow]"
        )

    console.print(f"\n[green]Target diseases ({len(target_diseases)}):[/green]")
    for i, disease in enumerate(target_diseases, 1):
        console.print(f"  {i}. {disease}")

    # Detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    console.print(f"\n[yellow]Using device:[/yellow] {device}")

    # ========================================================================
    # 2. Load and Filter Data
    # ========================================================================
    console.print(f"\n[bold]Loading dataset from {data_path}...[/bold]")

    if not data_path.exists():
        console.print(f"[red]Error: File not found: {data_path}[/red]")
        raise typer.Exit(1)

    df = pd.read_csv(data_path)
    console.print(f"  Total examples: {len(df):,}")
    console.print(f"  Total diseases: {df['disease'].nunique():,}")

    # Filter by target diseases
    df_filtered = df[df["disease"].isin(target_diseases)].copy()

    if len(df_filtered) == 0:
        console.print("[red]Error: No examples found for specified diseases[/red]")
        raise typer.Exit(1)

    # Display class distribution
    class_dist = df_filtered["disease"].value_counts()
    table = Table(title="Class Distribution")
    table.add_column("Disease", style="cyan")
    table.add_column("Examples", justify="right", style="green")
    for disease, count in class_dist.items():
        table.add_row(disease, f"{count:,}")
    console.print(table)

    # Validate: need at least 10 examples per class for proper train/val/test split
    min_examples = class_dist.min()
    if min_examples < 10:
        console.print(
            f"[red]Error: At least one disease has only {min_examples} examples[/red]"
        )
        console.print(
            "[yellow]Need at least 10 examples per disease for train/val/test split[/yellow]"
        )
        raise typer.Exit(1)

    # Create label mappings
    label2id = {label: i for i, label in enumerate(target_diseases)}
    id2label = {i: label for label, i in label2id.items()}
    df_filtered["label"] = df_filtered["disease"].map(label2id)

    # ========================================================================
    # 3. Train/Val/Test Split
    # ========================================================================
    console.print(f"\n[bold]Splitting dataset...[/bold]")

    train_df, temp_df = train_test_split(
        df_filtered,
        test_size=(test_size + val_size),
        stratify=df_filtered["label"],
        random_state=seed,
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_size / (test_size + val_size),
        stratify=temp_df["label"],
        random_state=seed,
    )

    console.print(f"  Train: {len(train_df):,} examples")
    console.print(f"  Validation: {len(val_df):,} examples")
    console.print(f"  Test: {len(test_df):,} examples")

    # ========================================================================
    # 4. Data Augmentation
    # ========================================================================
    if augment:
        console.print(
            f"\n[bold]Applying data augmentation (x{n_augmentations})...[/bold]"
        )

        augmented_rows = []
        for _, row in train_df.iterrows():
            # Keep original
            augmented_rows.append({"symptoms": row["symptoms"], "label": row["label"]})
            # Add augmented versions
            augmented_rows.extend(
                augment_example(
                    row["symptoms"], row["label"], n_augmentations, template_file
                )
            )

        train_aug_df = pd.DataFrame(augmented_rows)
        console.print(
            f"  Original: {len(train_df):,} → Augmented: {len(train_aug_df):,}"
        )
    else:
        train_aug_df = train_df

    # ========================================================================
    # 5. Tokenization
    # ========================================================================
    console.print(f"\n[bold]Loading tokenizer: {model_name}...[/bold]")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    console.print(f"  Vocabulary size: {tokenizer.vocab_size:,}")

    # Create HuggingFace datasets
    train_dataset = create_hf_dataset(train_aug_df)
    val_dataset = create_hf_dataset(val_df)
    test_dataset = create_hf_dataset(test_df)

    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
    )

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    console.print(f"  Tokenizing (max_length={max_length})...")
    tokenized_datasets = dataset_dict.map(tokenize_function, batched=True)

    # ========================================================================
    # 6. Model Training
    # ========================================================================
    console.print(f"\n[bold]Initializing model...[/bold]")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(target_diseases),
        id2label=id2label,
        label2id=label2id,
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        seed=seed,
        push_to_hub=False,
        use_mps_device=True if device == "mps" else False,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    console.print(f"\n[bold green]Starting training ({epochs} epochs)...[/bold green]")
    console.print("=" * 60)

    train_result = trainer.train()

    console.print("\n[bold green]Training complete![/bold green]")
    console.print(f"  Final training loss: {train_result.training_loss:.4f}")

    # ========================================================================
    # 7. Evaluation on Test Set
    # ========================================================================
    console.print(f"\n[bold]Evaluating on test set...[/bold]")

    test_results = trainer.evaluate(tokenized_datasets["test"])

    # Display results
    metrics_table = Table(title="Test Set Results")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", justify="right", style="green")

    for key, value in test_results.items():
        if key.startswith("eval_"):
            metric_name = key.replace("eval_", "").replace("_", " ").title()
            if isinstance(value, float):
                metrics_table.add_row(metric_name, f"{value:.4f}")

    console.print(metrics_table)

    # Get predictions for confusion matrix and classification report
    predictions = trainer.predict(tokenized_datasets["test"])
    y_pred = np.argmax(predictions.predictions, axis=-1)
    y_true = predictions.label_ids

    # Classification report
    console.print("\n[bold]Classification Report:[/bold]")
    report = classification_report(
        y_true, y_pred, target_names=target_diseases, digits=4
    )
    console.print(report)

    # ========================================================================
    # 8. Save Model and Artifacts
    # ========================================================================
    console.print(f"\n[bold]Saving model and artifacts...[/bold]")

    # Create output directories
    final_model_path = output_dir / "final"
    metrics_path = output_dir / "metrics"
    final_model_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    # Save model and tokenizer
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    console.print(f"  ✓ Model saved to: {final_model_path}")

    # Save label mappings
    with open(final_model_path / "label_mappings.json", "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    console.print(f"  ✓ Label mappings saved")

    # Save training metrics
    with open(metrics_path / "training_metrics.json", "w") as f:
        json.dump(
            {
                "training_loss": train_result.training_loss,
                "train_runtime": train_result.metrics["train_runtime"],
                "train_samples_per_second": train_result.metrics[
                    "train_samples_per_second"
                ],
            },
            f,
            indent=2,
        )

    # Save complete training history
    with open(metrics_path / "training_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # Save test results
    with open(metrics_path / "test_results.json", "w") as f:
        json.dump(test_results, f, indent=2)
    console.print(f"  ✓ Metrics saved to: {metrics_path}")

    # Save confusion matrix (optional)
    if save_confusion_matrix_flag:
        cm_path = metrics_path / "confusion_matrix.png"
        save_confusion_matrix(y_true, y_pred, target_diseases, cm_path)
        console.print(f"  ✓ Confusion matrix saved to: {cm_path}")
    else:
        console.print(
            f"  ⊘ Confusion matrix skipped (use --save-confusion-matrix to enable)"
        )

    # ========================================================================
    # 9. Summary
    # ========================================================================
    console.print("\n" + "=" * 60)
    console.print("[bold green]Training pipeline completed successfully![/bold green]")
    console.print("=" * 60)

    summary_table = Table(title="Summary")
    summary_table.add_column("Item", style="cyan")
    summary_table.add_column("Value", style="green")

    summary_table.add_row("Diseases", str(len(target_diseases)))
    summary_table.add_row("Training examples", f"{len(train_aug_df):,}")
    summary_table.add_row("Test accuracy", f"{test_results['eval_accuracy']:.2%}")
    summary_table.add_row("Test F1-score", f"{test_results['eval_f1_macro']:.4f}")
    summary_table.add_row("Model path", str(final_model_path))

    console.print(summary_table)

    console.print("\n[dim]Use this model with:[/dim]")
    console.print(f"[dim]  from transformers import pipeline[/dim]")
    console.print(
        f'[dim]  classifier = pipeline("text-classification", model="{final_model_path}")[/dim]'
    )
    console.print(f'[dim]  result = classifier("anxiety and chest pain")[/dim]\n')


if __name__ == "__main__":
    app()
