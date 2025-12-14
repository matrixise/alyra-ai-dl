"""
Application Streamlit pour la classification de maladies à partir de symptômes.

Cette application utilise un modèle transformers pré-entraîné pour prédire
les maladies potentielles basées sur une liste de symptômes fournis par l'utilisateur.
"""

import pathlib

import pandas as pd
import streamlit as st

from alyra_ai_dl.core import DEFAULT_MODEL_PATH, create_classifier, detect_device
from alyra_ai_dl.inference import predict_with_threshold


@st.cache_resource
def load_classifier(model_path: str, top_k: int = 2):
    """
    Charge et cache le classifier transformers.

    Args:
        model_path: Chemin vers le modèle
        top_k: Nombre de prédictions à retourner

    Returns:
        Pipeline de classification caché
    """
    device = detect_device()
    return create_classifier(
        model_path=pathlib.Path(model_path),
        device=device,
        top_k=top_k,
    )


def main():
    """Fonction principale de l'application Streamlit."""
    # Configuration de la page
    st.set_page_config(
        page_title="Classificateur de Maladies",
        page_icon="🏥",
        layout="wide",
    )

    st.title("🏥 Classificateur de Maladies")
    st.write(
        "Prédisez une maladie à partir de symptômes en utilisant un pipeline transformers. "
        "Entrez les symptômes séparés par des virgules et ajustez le seuil de confiance."
    )

    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        model_path = st.text_input(
            "Chemin du Modèle",
            str(DEFAULT_MODEL_PATH),
            help="Chemin vers le répertoire du modèle entraîné",
        )

        top_k = st.slider(
            "Top K prédictions",
            min_value=1,
            max_value=5,
            value=2,
            help="Nombre de prédictions principales à retourner",
        )

        threshold = st.slider(
            "Seuil de Confiance",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            help="Score de confiance minimum pour considérer une prédiction valide",
        )

        st.divider()

        # Informations sur le device
        device = detect_device()
        st.info(f"**Dispositif:** {device.value.upper()}")

    # Charger le classifier avec cache
    try:
        with st.spinner("Chargement du modèle..."):
            classifier = load_classifier(model_path, top_k)
            st.success("✅ Modèle chargé avec succès!")

            # Afficher les maladies supportées dans la sidebar
            with st.sidebar:
                st.divider()
                st.subheader("🏥 Maladies Supportées")
                diseases = list(classifier.model.config.id2label.values())
                for disease in diseases:
                    st.markdown(f"- **{disease.replace('_', ' ').title()}**")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        return

    # Input des symptômes
    st.subheader("📝 Entrez les Symptômes")
    symptoms = st.text_area(
        "Symptômes (séparés par des virgules)",
        "hip pain, back pain, neck pain, low back pain, problems with movement, loss of sensation, leg cramps or spasms",
        height=100,
        help="Entrez les symptômes séparés par des virgules",
    )

    # Exemples de symptômes
    with st.expander("💡 Exemples de Symptômes"):
        st.markdown("""
        **Spondylolisthésis:**
        - `low back pain, problems with movement, paresthesia, leg cramps or spasms, leg weakness`

        **Hernie Discale:**
        - `arm pain, back pain, neck pain, paresthesia, shoulder pain, arm weakness`

        **Trouble Panique:**
        - `depressive or psychotic symptoms, irregular heartbeat, breathing fast`
        """)

    # Bouton de prédiction
    if st.button("🔍 Prédire la Maladie", type="primary", width="stretch"):
        if not symptoms.strip():
            st.warning("⚠️ Veuillez entrer au moins un symptôme.")
            return

        with st.spinner("Analyse des symptômes..."):
            try:
                result = predict_with_threshold(classifier, symptoms, threshold)

                # Affichage des résultats principaux
                st.subheader("📊 Résultats")

                col1, col2, col3 = st.columns(3)

                with col1:
                    disease_color = (
                        "normal" if result["disease"] != "unknown" else "off"
                    )
                    st.metric(
                        "Maladie Prédite",
                        result["disease"].title(),
                        delta=None,
                        delta_color=disease_color,
                    )

                with col2:
                    confidence_pct = f"{result['confidence']:.1%}"
                    st.metric("Confiance", confidence_pct)

                with col3:
                    threshold_pct = f"{result['threshold']:.1%}"
                    st.metric("Seuil", threshold_pct)

                # Afficher suggestion si unknown
                if result["disease"] == "unknown":
                    st.warning(f"⚠️ {result['suggestion']}")
                else:
                    st.success(
                        f"✅ Maladie identifiée avec {confidence_pct} de confiance"
                    )

                # Graphique des probabilités
                st.subheader("📈 Toutes les Probabilités")

                probs_df = pd.DataFrame(
                    list(result["all_probs"].items()),
                    columns=["Maladie", "Probabilité"],
                ).sort_values("Probabilité", ascending=False)

                # Créer un bar chart
                st.bar_chart(probs_df.set_index("Maladie"), height=300)

                # Table détaillée
                st.subheader("📋 Probabilités Détaillées")
                probs_df["Probabilité"] = probs_df["Probabilité"].apply(
                    lambda x: f"{x:.2%}"
                )
                st.dataframe(
                    probs_df,
                    hide_index=True,
                    use_container_width=True,
                )

                # Détails JSON
                with st.expander("🔍 Voir les Résultats Bruts"):
                    st.json(result)

            except Exception as e:
                st.error(f"❌ Erreur lors de la prédiction: {e}")
                st.exception(e)

    # Footer
    st.divider()
    st.caption(
        "🤖 Propulsé par Transformers Pipeline | "
        "Créé avec Streamlit | "
        f"Modèle: {model_path}"
    )


if __name__ == "__main__":
    main()
