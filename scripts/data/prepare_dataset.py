#!/usr/bin/env python
"""
Script de préparation du dataset pour l'entraînement.

Ce script:
1. Charge le dataset augmenté
2. Supprime les doublons
3. Filtre les maladies souhaitées
4. Supprime les colonnes de symptômes complètement vides
5. Crée un dataset final avec les colonnes 'disease' et 'symptoms'
"""

import pathlib
import typing

import pandas as pd
import typer
from loguru import logger

app = typer.Typer(help="Prepare dataset for model training")


@app.command()
def prepare(
    input_file: typing.Annoated[
        pathlib.Path,
        typer.Option(
            ...,
            "--input",
            "-i",
            help="Path to input CSV file (augmented dataset)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output_file: typing.Annotated[
        pathlib.Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to output CSV file (prepared dataset)",
        ),
    ],
    diseases: typing.Annotated[
        str,
        typer.Option(
            "--diseases",
            "-d",
            help="Comma-separated list of diseases to keep",
        ),
    ],
) -> None:
    """
    Prepare dataset for training by filtering diseases and creating symptom descriptions.

    Example:
        python prepare_dataset.py \\
            --input data/Final_Augmented_dataset_Diseases_and_Symptoms.csv \\
            --output data/prepared-dataset.csv \\
            --diseases "anxiety,cystitis,herniated disk,panic disorder,pneumonia,spondylolisthesis"
    """
    # Parse diseases list
    disease_list = [d.strip() for d in diseases.split(",")]
    logger.info(f"Preparing dataset with {len(disease_list)} diseases: {disease_list}")

    # 1. Load dataset
    logger.info(f"Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # 2. Remove duplicates
    logger.info("Removing duplicate rows...")
    df_unique = df.drop_duplicates()
    duplicates_removed = len(df) - len(df_unique)
    duplicated_pct = duplicates_removed / len(df)
    logger.info(f"Removed {duplicates_removed} duplicates ({duplicated_pct:.2%}), {len(df_unique)} rows remaining")

    # 3. Filter for selected diseases
    logger.info(f"Filtering for diseases: {disease_list}...")
    df_filtered = df_unique[df_unique["diseases"].isin(disease_list)]
    rows_filtered = len(df_unique) - len(df_filtered)
    logger.info(f"Removed {rows_filtered} rows, {len(df_filtered)} rows remaining with selected diseases")

    if len(df_filtered) == 0:
        logger.error("No rows left after filtering! Check disease names.")
        raise typer.Exit(code=1)

    # Display distribution of diseases
    logger.info("Disease distribution after filtering:")
    disease_counts = df_filtered["diseases"].value_counts()
    for disease, count in disease_counts.items():
        logger.info(f"  - {disease}: {count}")

    # 4. Remove columns (symptoms) that are completely zero
    logger.info("Removing symptom columns that are all zeros...")
    symptom_columns = [col for col in df_filtered.columns if col != "diseases"]
    columns_to_keep = ["diseases"]

    for col in symptom_columns:
        if df_filtered[col].sum() > 0:  # If column has at least one 1
            columns_to_keep.append(col)

    columns_removed = len(df_filtered.columns) - len(columns_to_keep)
    logger.info(
        f"Removed {columns_removed} empty symptom columns, {len(columns_to_keep) - 1} symptom columns remaining"
    )

    df_cleaned = df_filtered[columns_to_keep]

    # 5. Create output dataset with 'disease' and 'symptoms' columns
    logger.info("Creating output dataset with 'disease' and 'symptoms' columns...")

    output_rows = []
    symptom_cols = [col for col in columns_to_keep if col != "diseases"]

    for _idx, row in df_cleaned.iterrows():
        disease = row["diseases"]
        # Get all symptom names where value is 1
        symptoms = [col for col in symptom_cols if row[col] == 1]
        symptoms_text = ", ".join(symptoms)

        output_rows.append({"disease": disease, "symptoms": symptoms_text})

    df_output = pd.DataFrame(output_rows)

    # 6. Save output file
    logger.info(f"Saving prepared dataset to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_file, index=False)

    logger.success("Dataset prepared successfully!")
    logger.info(f"Final dataset: {len(df_output)} rows, 2 columns")
    logger.info(f"Output saved to: {output_file}")

    # Show sample
    logger.info("\nSample rows from prepared dataset:")
    for _i, row in df_output.head(3).iterrows():
        logger.info(f"  [{row['disease']}]: {row['symptoms'][:100]}...")


if __name__ == "__main__":
    app()
