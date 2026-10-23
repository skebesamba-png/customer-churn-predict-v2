import streamlit as st
import pandas as pd
import prince
import matplotlib.pyplot as plt
import numpy as np
import joblib


# =====================================================
# CHARGEMENT DU MODÈLE
# =====================================================

import joblib
import streamlit as st
from pathlib import Path

@st.cache_resource
def charger_modele():
    chemin_modele = (
        Path(__file__).resolve().parent / "modele_logistique.pkl"
    )

    if not chemin_modele.exists():
        raise FileNotFoundError(
            f"Le fichier du modèle est introuvable : {chemin_modele}"
        )

    return joblib.load(chemin_modele)

modele = charger_modele()
# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Predict",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# MENU
# ============================================================

st.sidebar.title("📊 CUSTOMER CHURN")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Accueil",
        "📂 Chargement des données",
         "📊 AFDM",
        "🤖 Modèles",
        "🔮 Prédiction"
    ]
)


# ============================================================
# ACCUEIL
# ============================================================

if menu == "🏠 Accueil":

    st.markdown(
        """
        <div style="text-align:center">

        <p style="
            font-size:14px;
            color:#64748b;
            letter-spacing:3px;
            font-weight:600;
        ">
            MÉMOIRE DE FIN D'ÉTUDES
        </p>

        <h1 style="
            font-size:42px;
            color:#0f172a;
        ">
            📊 CUSTOMER CHURN PREDICT
        </h1>

        <h3 style="
            color:#475569;
            font-weight:400;
        ">
            Plateforme d'analyse et de prédiction
            du désabonnement client
        </h3>

        <p style="
            font-size:16px;
            color:#64748b;
        ">
            <strong>Master 2</strong> –
            Statistique et Informatique Décisionnelle
        </p>

        <p style="
            font-size:14px;
            color:#94a3b8;
        ">
            Python • Pandas • Scikit-learn • Streamlit
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.subheader("🎯 Présentation du projet")

    st.write(
        """
        Cette application a été développée dans le cadre d'un
        mémoire de Master 2 en Statistique et Informatique
        Décisionnelle.

        Elle utilise les techniques de Machine Learning pour
        analyser le comportement des clients et prédire leur
        risque de désabonnement.
        """
    )

    st.subheader("🔎 Objectif")

    st.write(
        """
        L'objectif est de développer un modèle prédictif capable
        d'identifier les clients susceptibles de se désabonner
        et de fournir un outil d'aide à l'analyse.
        """
    )

    st.subheader("🤖 Modèle retenu pour la prédiction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Modèle",
            "Régression Logistique"
        )

    st.subheader("⚙️ Fonctionnement")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### 📂 Données")
        st.write(
            "Chargement et exploration des données clients."
        )

    with col2:
        st.markdown("### 🤖 Modèles")
        st.write(
            "Présentation et comparaison des modèles de Machine Learning."
        )

    with col3:
        st.markdown("### 🔮 Prédiction")
        st.write(
            "Prédiction du risque de désabonnement d'un client."
        )
        
    with col4:
        st.markdown("### 📊 AFDM")
        st.write(
             "Représenter la structure des individus dans un espace."
        )


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

elif menu == "📂 Chargement des données":

    st.title("📂 Chargement des données")

    st.markdown(
        """
        Importez votre fichier de données clients afin de réaliser
        une première exploration de la base avant l'analyse prédictive.
        """
    )

    st.divider()

    # ============================================================
    # IMPORTATION DU FICHIER
    # ============================================================

    st.subheader("📁 Importation du fichier")

    fichier = st.file_uploader(
        "Sélectionnez votre fichier de données",
        type=["csv", "xlsx", "xls"],
        help="Formats acceptés : CSV, XLSX et XLS"
    )

    if fichier is None:

        st.info(
            "👆 Veuillez sélectionner un fichier CSV ou Excel "
            "pour commencer l'analyse."
        )

    else:

        try:

            # ====================================================
            # LECTURE DU FICHIER
            # ====================================================

            if fichier.name.lower().endswith(".csv"):

                try:
                    df = pd.read_csv(fichier)

                except UnicodeDecodeError:

                    fichier.seek(0)

                    df = pd.read_csv(
                        fichier,
                        encoding="latin1"
                    )

            else:

                df = pd.read_excel(fichier)

            # ====================================================
            # STOCKAGE
            # ====================================================

            st.session_state["donnees"] = df
            st.session_state["nom_fichier"] = fichier.name

            st.success(
                f"✅ Fichier **{fichier.name}** chargé avec succès."
            )

            # ====================================================
            # VUE D'ENSEMBLE
            # ====================================================

            st.divider()

            st.subheader("📊 Vue d'ensemble")

            nombre_lignes = df.shape[0]
            nombre_colonnes = df.shape[1]
            valeurs_manquantes = int(
                df.isna().sum().sum()
            )
            doublons = int(
                df.duplicated().sum()
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "👥 Observations",
                    f"{nombre_lignes:,}".replace(",", " ")
                )

            with col2:
                st.metric(
                    "📋 Variables",
                    nombre_colonnes
                )

            with col3:
                st.metric(
                    "⚠️ Valeurs manquantes",
                    f"{valeurs_manquantes:,}".replace(",", " ")
                )

            with col4:
                st.metric(
                    "🔁 Doublons",
                    f"{doublons:,}".replace(",", " ")
                )

            # ====================================================
            # APERÇU
            # ====================================================

            st.divider()

            st.subheader("👀 Aperçu de la base")

            nombre_lignes_affichees = st.slider(
                "Nombre de lignes à afficher",
                min_value=5,
                max_value=min(50, len(df)),
                value=min(10, len(df)),
                step=5
            )

            st.dataframe(
                df.head(nombre_lignes_affichees),
                use_container_width=True,
                hide_index=True
            )

            # ====================================================
            # QUALITÉ DES DONNÉES
            # ====================================================

            st.divider()

            st.subheader("🔎 Qualité des données")

            taux_manquants = (
                valeurs_manquantes
                / (df.shape[0] * df.shape[1])
            ) * 100

            col1, col2 = st.columns(2)

            with col1:

                if valeurs_manquantes == 0:

                    st.success(
                        "✅ Aucune valeur manquante détectée."
                    )

                else:

                    st.warning(
                        f"⚠️ {valeurs_manquantes:,} valeur(s) "
                        "manquante(s) détectée(s)."
                    )

            with col2:

                if doublons == 0:

                    st.success(
                        "✅ Aucun doublon détecté."
                    )

                else:

                    st.warning(
                        f"⚠️ {doublons:,} ligne(s) dupliquée(s) détectée(s)."
                    )

            st.metric(
                "Taux global de valeurs manquantes",
                f"{taux_manquants:.2f} %"
            )

            # ====================================================
            # DICTIONNAIRE DES VARIABLES
            # ====================================================

            st.divider()

            st.subheader("📋 Dictionnaire des variables")

            dictionnaire = pd.DataFrame({

                "Variable": df.columns,

                "Type": df.dtypes.astype(str).values,

                "Valeurs manquantes":
                    df.isna().sum().values,

                "Taux de manquants (%)":
                    (df.isna().mean() * 100)
                    .round(2)
                    .values,

                "Valeurs uniques": [
                    df[col].nunique(dropna=True)
                    for col in df.columns
                ]
            })

            st.dataframe(
                dictionnaire,
                use_container_width=True,
                hide_index=True
            )

            # ====================================================
            # VISUALISATION DES VARIABLES
            # ====================================================

            st.divider()

            st.subheader("📊 Visualisation des variables")

            st.write(
                """
                Sélectionnez une variable pour examiner sa
                distribution, ses effectifs et ses pourcentages.
                """
            )

            variable = st.selectbox(
                "🔎 Variable à visualiser",
                df.columns,
                key="variable_visualisation"
            )

            serie = df[variable]

            # ====================================================
            # VARIABLE QUALITATIVE
            # ====================================================

            if (
                serie.dtype == "object"
                or str(serie.dtype) == "category"
                or serie.nunique(dropna=True) <= 10
            ):

                repartition = (
                    serie
                    .fillna("Valeur manquante")
                    .value_counts()
                    .reset_index()
                )

                repartition.columns = [
                    "Modalité",
                    "Effectif"
                ]

                total = repartition["Effectif"].sum()

                repartition["Pourcentage"] = (
                    repartition["Effectif"]
                    / total
                    * 100
                ).round(2)

                # ------------------------------------------------
                # INDICATEURS
                # ------------------------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Nombre de modalités",
                        repartition.shape[0]
                    )

                with col2:
                    st.metric(
                        "Effectif total",
                        f"{total:,}".replace(",", " ")
                    )

                with col3:
                    st.metric(
                        "Modalité dominante",
                        str(
                            repartition.iloc[0]["Modalité"]
                        )
                    )

                # ------------------------------------------------
                # TABLEAU
                # ------------------------------------------------

                st.markdown("#### 📋 Répartition des modalités")

                st.dataframe(
                    repartition,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Pourcentage": st.column_config.NumberColumn(
                            "Pourcentage",
                            format="%.2f %%"
                        )
                    }
                )

                # ------------------------------------------------
                # GRAPHIQUE DES EFFECTIFS
                # ------------------------------------------------

                st.markdown("#### 📊 Effectifs par modalité")

                st.bar_chart(
                    repartition.set_index("Modalité")[
                        "Effectif"
                    ]
                )

            # ====================================================
            # VARIABLE NUMÉRIQUE
            # ====================================================

            else:

                # ------------------------------------------------
                # INDICATEURS
                # ------------------------------------------------

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Moyenne",
                        f"{serie.mean():.2f}"
                    )

                with col2:
                    st.metric(
                        "Médiane",
                        f"{serie.median():.2f}"
                    )

                with col3:
                    st.metric(
                        "Minimum",
                        f"{serie.min():.2f}"
                    )

                with col4:
                    st.metric(
                        "Maximum",
                        f"{serie.max():.2f}"
                    )

                # ------------------------------------------------
                # HISTOGRAMME
                # ------------------------------------------------

                st.markdown(
                    "#### 📊 Distribution de la variable"
                )

                histogramme = pd.cut(
                    serie.dropna(),
                    bins=10
                ).value_counts().sort_index()

                histogramme_df = pd.DataFrame({
                    "Intervalle": histogramme.index.astype(str),
                    "Effectif": histogramme.values
                })

                st.bar_chart(
                    histogramme_df.set_index("Intervalle")
                )

                # ------------------------------------------------
                # STATISTIQUES
                # ------------------------------------------------

                st.markdown(
                    "#### 📋 Statistiques descriptives"
                )

                statistiques = pd.DataFrame({

                    "Indicateur": [
                        "Nombre d'observations",
                        "Valeurs manquantes",
                        "Moyenne",
                        "Médiane",
                        "Écart-type",
                        "Minimum",
                        "Maximum"
                    ],

                    "Valeur": [
                        serie.count(),
                        serie.isna().sum(),
                        round(serie.mean(), 2),
                        round(serie.median(), 2),
                        round(serie.std(), 2),
                        round(serie.min(), 2),
                        round(serie.max(), 2)
                    ]
                })

                st.dataframe(
                    statistiques,
                    use_container_width=True,
                    hide_index=True
                )

            # ====================================================
            # STATISTIQUES DESCRIPTIVES
            # ====================================================

            st.divider()

            st.subheader("📈 Statistiques descriptives")

            variables_numeriques = df.select_dtypes(
                include="number"
            )

            if not variables_numeriques.empty:

                st.dataframe(
                    variables_numeriques.describe().T,
                    use_container_width=True
                )

            else:

                st.info(
                    "Aucune variable numérique détectée "
                    "dans cette base."
                )

        except Exception as e:

            st.error(
                "❌ Impossible de charger le fichier."
            )

            st.exception(e)
            
# ============================================================
# 📊 AFDM — ANALYSE FACTORIELLE DES DONNÉES MIXTES
# ============================================================

# ============================================================
# 📊 AFDM — ANALYSE FACTORIELLE DES DONNÉES MIXTES
# ============================================================

# ============================================================
# 📊 AFDM — ANALYSE FACTORIELLE DES DONNÉES MIXTES
# ============================================================

elif menu == "📊 AFDM":

    st.title("📊 Analyse Factorielle des Données Mixtes")
    st.markdown(
        """
        L'AFDM permet d'étudier simultanément les relations entre
        variables qualitatives et quantitatives et de représenter
        la structure des individus dans un espace factoriel.
        """
    )

    # ========================================================
    # 1. VÉRIFICATION DES DONNÉES
    # ========================================================

    if "donnees" not in st.session_state:

        st.warning(
            "⚠️ Aucune donnée disponible. "
            "Veuillez d'abord charger un fichier dans "
            "« 📂 Chargement des données »."
        )

        st.stop()

    df = st.session_state["donnees"].copy()

    # ========================================================
    # 2. IMPORTS NÉCESSAIRES
    # ========================================================

    import matplotlib.pyplot as plt
    import prince

    # ========================================================
    # 3. DÉFINITION DES VARIABLES
    # ========================================================

    variables_qualitatives = [
        "sexe",
        "categorie_client",
        "smartphone",
        "multisim",
        "offre_telephonie_fixe",
        "plusieurs_ligne_telephonie_fixe",
        "type_internet",
        "services_messageries",
        "sauvegarde_contact_en_ligne",
        "acces_connexion_4G_mobile",
        "abonnement_musique",
        "abonnement_tv",
        "souscription_sva_jeux",
        "utlisation_orange_money"
    ]

    variables_quantitatives = [
        "anciennete_telco_mois",
        "depence_mensuelle_data",
        "depense_mensuelle_dredit"
    ]

    variables_actives = (
        variables_qualitatives
        + variables_quantitatives
    )

    # ========================================================
    # 4. VÉRIFICATION DES VARIABLES DISPONIBLES
    # ========================================================

    variables_disponibles = [
        variable
        for variable in variables_actives
        if variable in df.columns
    ]

    variables_absentes = [
        variable
        for variable in variables_actives
        if variable not in df.columns
    ]

    if variables_absentes:

        st.warning(
            "⚠️ Certaines variables prévues pour l'AFDM "
            "ne sont pas présentes dans les données :"
        )

        st.write(variables_absentes)

    if len(variables_disponibles) == 0:

        st.error(
            "❌ Aucune variable prévue pour l'AFDM "
            "n'est disponible dans le fichier."
        )

        st.stop()

    # ========================================================
    # 5. CONSERVATION DE LA VARIABLE CIBLE
    # ========================================================

    if "Desabonnement" in df.columns:

        desabonnement = (
            df["Desabonnement"]
            .copy()
        )

    else:

        desabonnement = None

    # ========================================================
    # 6. CONSTRUCTION DE LA BASE AFDM
    # ========================================================

    df_famd = (
        df[variables_disponibles]
        .copy()
    )

    # ========================================================
    # 7. TRAITEMENT DES VARIABLES QUALITATIVES
    # ========================================================

    qualitatives_disponibles = [
        variable
        for variable in variables_qualitatives
        if variable in df_famd.columns
    ]

    for variable in qualitatives_disponibles:

        # Conversion en chaîne
        df_famd[variable] = (
            df_famd[variable]
            .astype("object")
        )

        # Remplacement des valeurs manquantes par le mode
        if df_famd[variable].isna().any():

            mode = (
                df_famd[variable]
                .mode(dropna=True)
            )

            if len(mode) > 0:

                valeur_remplacement = mode.iloc[0]

            else:

                valeur_remplacement = "Inconnu"

            df_famd[variable] = (
                df_famd[variable]
                .fillna(valeur_remplacement)
            )

        # Conversion finale en texte
        df_famd[variable] = (
            df_famd[variable]
            .astype(str)
        )

    # ========================================================
    # 8. TRAITEMENT DES VARIABLES QUANTITATIVES
    # ========================================================

    quantitatives_disponibles = [
        variable
        for variable in variables_quantitatives
        if variable in df_famd.columns
    ]

    for variable in quantitatives_disponibles:

        # Conversion numérique
        df_famd[variable] = pd.to_numeric(
            df_famd[variable],
            errors="coerce"
        )

        # Remplacement des valeurs manquantes par la médiane
        if df_famd[variable].isna().any():

            mediane = (
                df_famd[variable]
                .median()
            )

            if pd.isna(mediane):
                mediane = 0

            df_famd[variable] = (
                df_famd[variable]
                .fillna(mediane)
            )

    # ========================================================
    # 9. PRÉSENTATION DE LA BASE AFDM
    # ========================================================

    with st.expander(
        "📋 Afficher la base utilisée pour l'AFDM"
    ):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Observations",
                f"{df_famd.shape[0]:,}"
            )

        with col2:
            st.metric(
                "Variables",
                f"{df_famd.shape[1]:,}"
            )

        with col3:
            st.metric(
                "Valeurs manquantes",
                f"{df_famd.isna().sum().sum():,}"
            )

        st.dataframe(
            df_famd.head(20),
            use_container_width=True
        )

    # ========================================================
    # 10. PARAMÈTRES DE L'AFDM
    # ========================================================

    st.markdown("---")

    st.subheader(
        "⚙️ Paramètres de l'analyse"
    )

    nombre_axes = st.slider(
        "Nombre d'axes factoriels",
        min_value=2,
        max_value=min(
            10,
            df_famd.shape[1]
        ),
        value=min(
            5,
            df_famd.shape[1]
        ),
        step=1
    )

    # ========================================================
    # 11. ESTIMATION DE L'AFDM
    # ========================================================

    try:

        with st.spinner(
            "⏳ Calcul de l'AFDM en cours..."
        ):

            afdm = prince.FAMD(
                n_components=nombre_axes,
                n_iter=10,
                copy=True,
                check_input=True,
                random_state=42
            )

            afdm = afdm.fit(
                df_famd
            )

        st.success(
            "✅ AFDM réalisée avec succès."
        )

    except Exception as e:

        st.error(
            "❌ Une erreur est survenue lors "
            "du calcul de l'AFDM."
        )

        st.code(
            str(e)
        )

        st.stop()

    # ========================================================
    # 12. VALEURS PROPRES ET INERTIE
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📈 Valeurs propres et inertie expliquée"
    )

    try:

        valeurs_propres = (
            afdm.eigenvalues_
        )

        valeurs_propres = np.asarray(
            valeurs_propres,
            dtype=float
        )

        # Sécurité
        valeurs_propres = (
            valeurs_propres[
                np.isfinite(valeurs_propres)
            ]
        )

        if len(valeurs_propres) == 0:

            raise ValueError(
                "Aucune valeur propre valide n'a été retournée."
            )

        axes = [
            f"Axe {i + 1}"
            for i in range(
                len(valeurs_propres)
            )
        ]

        inertie_df = pd.DataFrame(
            {
                "Axe": axes,
                "Valeur propre": valeurs_propres
            }
        )

        inertie_df[
            "Inertie expliquée (%)"
        ] = (
            valeurs_propres
            / valeurs_propres.sum()
            * 100
        )

        inertie_df[
            "Inertie cumulée (%)"
        ] = (
            inertie_df[
                "Inertie expliquée (%)"
            ].cumsum()
        )

        # ----------------------------------------------------
        # Tableau
        # ----------------------------------------------------

        st.dataframe(
            inertie_df.style.format(
                {
                    "Valeur propre": "{:.4f}",
                    "Inertie expliquée (%)": "{:.2f}",
                    "Inertie cumulée (%)": "{:.2f}"
                }
            ),
            use_container_width=True
        )

        # ----------------------------------------------------
        # Éboulis des valeurs propres
        # ----------------------------------------------------

        st.markdown(
            "### 📉 Éboulis des valeurs propres"
        )

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        ax.plot(
            range(
                1,
                len(valeurs_propres) + 1
            ),
            valeurs_propres,
            marker="o"
        )

        ax.set_xlabel(
            "Axes factoriels"
        )

        ax.set_ylabel(
            "Valeur propre"
        )

        ax.set_title(
            "Éboulis des valeurs propres"
        )

        ax.set_xticks(
            range(
                1,
                len(valeurs_propres) + 1
            )
        )

        ax.grid(
            alpha=0.25
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Inertie expliquée
        # ----------------------------------------------------

        st.markdown(
            "### 📊 Inertie expliquée par axe"
        )

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        ax.bar(
            axes,
            inertie_df[
                "Inertie expliquée (%)"
            ]
        )

        ax.set_xlabel(
            "Axes factoriels"
        )

        ax.set_ylabel(
            "Inertie expliquée (%)"
        )

        ax.set_title(
            "Part d'inertie expliquée par chaque axe"
        )

        ax.grid(
            axis="y",
            alpha=0.25
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

        # ----------------------------------------------------
        # Inertie cumulée
        # ----------------------------------------------------

        st.markdown(
            "### 📈 Inertie cumulée"
        )

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        ax.plot(
            range(
                1,
                len(inertie_df) + 1
            ),
            inertie_df[
                "Inertie cumulée (%)"
            ],
            marker="o"
        )

        ax.set_xlabel(
            "Axes factoriels"
        )

        ax.set_ylabel(
            "Inertie cumulée (%)"
        )

        ax.set_title(
            "Inertie cumulée expliquée"
        )

        ax.grid(
            alpha=0.25
        )

        st.pyplot(
            fig,
            use_container_width=True
        )

        plt.close(fig)

    except Exception as e:

        st.error(
            "❌ Impossible de calculer "
            "les valeurs propres."
        )

        st.code(
            str(e)
        )

    # ========================================================
    # 13. COORDONNÉES FACTORIELLES DES INDIVIDUS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "👥 Carte factorielle des individus"
    )

    try:

        coord_individus = (
            afdm.row_coordinates(
                df_famd
            )
        )

        coord_individus = pd.DataFrame(
            coord_individus
        )

        # Normalisation des noms d'axes
        coord_individus.columns = [
            f"Axe_{i + 1}"
            for i in range(
                coord_individus.shape[1]
            )
        ]

        # Conversion numérique
        coord_individus = (
            coord_individus.apply(
                pd.to_numeric,
                errors="coerce"
            )
        )

        # Diagnostic
        st.caption(
            f"{coord_individus.shape[0]:,} individus × "
            f"{coord_individus.shape[1]} axes"
        )

        # ====================================================
        # CHOIX DES AXES
        # ====================================================

        col1, col2 = st.columns(2)

        axes_disponibles = list(
            coord_individus.columns
        )

        with col1:

            axe_x = st.selectbox(
                "Axe horizontal",
                axes_disponibles,
                index=0,
                key="axe_x_individus"
            )

        with col2:

            axe_y = st.selectbox(
                "Axe vertical",
                axes_disponibles,
                index=1
                if len(axes_disponibles) > 1
                else 0,
                key="axe_y_individus"
            )

        # ====================================================
        # NETTOYAGE POUR LE GRAPHIQUE
        # ====================================================

        coord_plot_individus = (
            coord_individus
            .copy()
        )

        coord_plot_individus = (
            coord_plot_individus
            .replace(
                [np.inf, -np.inf],
                np.nan
            )
            .dropna(
                subset=[
                    axe_x,
                    axe_y
                ]
            )
        )

        # ====================================================
        # CARTE FACTORIELLE
        # ====================================================

        if coord_plot_individus.empty:

            st.error(
                "❌ La carte factorielle ne peut pas "
                "être affichée car aucune coordonnée "
                "valide n'est disponible."
            )

            st.dataframe(
                coord_individus.head(10),
                use_container_width=True
            )

        else:

            fig, ax = plt.subplots(
                figsize=(11, 7)
            )

            ax.scatter(
                coord_plot_individus[
                    axe_x
                ],
                coord_plot_individus[
                    axe_y
                ],
                alpha=0.55,
                s=35
            )

            ax.axhline(
                y=0,
                linewidth=0.8
            )

            ax.axvline(
                x=0,
                linewidth=0.8
            )

            ax.set_xlabel(
                axe_x,
                fontsize=11
            )

            ax.set_ylabel(
                axe_y,
                fontsize=11
            )

            ax.set_title(
                f"Carte factorielle des individus : "
                f"{axe_x} × {axe_y}",
                fontsize=13
            )

            ax.grid(
                alpha=0.25
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            st.caption(
                f"Individus représentés : "
                f"{len(coord_plot_individus):,} "
                f"sur {len(coord_individus):,}"
            )

    except Exception as e:

        st.error(
            "❌ Impossible de générer "
            "la carte factorielle des individus."
        )

        st.code(
            str(e)
        )

    
    # ========================================================
    # 16. CONTRIBUTION DES VARIABLES
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📊 Contributions des variables"
    )

    try:

        # ----------------------------------------------------
        # Tentative de récupération des contributions
        # ----------------------------------------------------

        contributions = (
            afdm.column_contributions_
        )

        contributions = pd.DataFrame(
            contributions
        )

        contributions.columns = [
            f"Axe_{i + 1}"
            for i in range(
                contributions.shape[1]
            )
        ]

        contributions = (
            contributions.apply(
                pd.to_numeric,
                errors="coerce"
            )
        )

        st.dataframe(
            contributions.style.format(
                "{:.4f}"
            ),
            use_container_width=True
        )

        # ----------------------------------------------------
        # Top 10 contributions
        # ----------------------------------------------------

        axes_contribution = [
            axe
            for axe in [
                "Axe_1",
                "Axe_2"
            ]
            if axe in contributions.columns
        ]

        if len(axes_contribution) >= 2:

            contribution_top = (
                contributions[
                    axes_contribution
                ]
                .sum(axis=1)
                .sort_values(
                    ascending=False
                )
                .head(10)
            )

            fig, ax = plt.subplots(
                figsize=(11, 7)
            )

            contribution_top.sort_values().plot(
                kind="barh",
                ax=ax
            )

            ax.set_xlabel(
                "Contribution cumulée"
            )

            ax.set_ylabel(
                "Variables"
            )

            ax.set_title(
                "Top 10 des variables "
                "contribuant aux axes 1 et 2"
            )

            ax.grid(
                axis="x",
                alpha=0.25
            )

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

    except Exception as e:

        st.info(
            "ℹ️ Les contributions détaillées "
            "ne sont pas disponibles avec cette "
            "configuration de Prince."
        )

    # ========================================================
    # 18. SYNTHÈSE DE L'AFDM
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📌 Synthèse de l'AFDM"
    )

    try:

        inertie_axe1 = (
            inertie_df.loc[
                inertie_df["Axe"] == "Axe 1",
                "Inertie expliquée (%)"
            ].iloc[0]
        )

        inertie_axe2 = (
            inertie_df.loc[
                inertie_df["Axe"] == "Axe 2",
                "Inertie expliquée (%)"
            ].iloc[0]
        )

        inertie_12 = (
            inertie_axe1
            + inertie_axe2
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Inertie Axe 1",
                f"{inertie_axe1:.2f} %"
            )

        with col2:

            st.metric(
                "Inertie Axe 2",
                f"{inertie_axe2:.2f} %"
            )

        with col3:

            st.metric(
                "Inertie Axes 1 + 2",
                f"{inertie_12:.2f} %"
            )

        st.markdown(
            f"""
            **Interprétation :**

            Les deux premiers axes factoriels expliquent
            conjointement **{inertie_12:.2f} %** de
            l'inertie totale.

            Le plan factoriel formé par les axes 1 et 2
            permet ainsi de visualiser une partie de la
            structure des individus et des variables.
            """
        )

    except Exception:

        pass

    # ========================================================
    # 19. EXPORT DES RÉSULTATS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "💾 Export des résultats de l'AFDM"
    )

    col1, col2, col3 = st.columns(3)

    # --------------------------------------------------------
    # Export base AFDM
    # --------------------------------------------------------

    with col1:

        csv_base = df_famd.to_csv(
            index=False,
            encoding="utf-8-sig"
        )

        st.download_button(
            label="📥 Base AFDM",
            data=csv_base,
            file_name="base_AFDM.csv",
            mime="text/csv"
        )

    # --------------------------------------------------------
    # Export coordonnées individus
    # --------------------------------------------------------

    with col2:

        csv_individus = (
            coord_individus
            .to_csv(
                index=True,
                encoding="utf-8-sig"
            )
        )

        st.download_button(
            label="📥 Coordonnées individus",
            data=csv_individus,
            file_name="coordonnees_individus_AFDM.csv",
            mime="text/csv"
        )

# ============================================================
# MODÈLES DE MACHINE LEARNING
# ============================================================

elif menu == "🤖 Modèles":

    st.title("🤖 Modèles de Machine Learning")

    st.markdown(
        """
        Cette section présente les différents modèles de Machine
        Learning utilisés pour la prédiction du désabonnement
        client ainsi que leurs performances sur l'échantillon test.
        """
    )

    st.divider()

    # ========================================================
    # MODÈLES UTILISÉS
    # ========================================================

    st.subheader("🧠 Modèles étudiés")

    modeles = [
        "Régression logistique",
        "K plus proches voisins (KNN)",
        "Naive Bayes",
        "Arbre de décision",
        "Random Forest",
        "Gradient Boosting",
        "XGBoost",
        "SVM"
    ]

    col1, col2 = st.columns(2)

    with col1:

        for modele in modeles[:4]:
            st.markdown(f"• **{modele}**")

    with col2:

        for modele in modeles[4:]:
            st.markdown(f"• **{modele}**")

    st.divider()

    # ========================================================
    # PERFORMANCES DISPONIBLES
    # ========================================================

    st.subheader("📊 Top 2 des modèles les plus performants")

    resultats = pd.DataFrame({

        "Modèle": [
            "Régression logistique",
            "Gradient Boosting"
        ],

        "Accuracy (%)": [
            88.86,
            89.08
        ],

        "Précision (%)": [
            81.93,
            84.36
        ],

        "Rappel (%)": [
            80.95,
            78.33
        ],

        "F1-score (%)": [
            81.44,
            81.23
        ],

        "AUC (%)": [
            95.20,
            95.78
        ]
    })

    st.dataframe(
        resultats,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Accuracy (%)": st.column_config.NumberColumn(
                format="%.2f %%"
            ),
            "Précision (%)": st.column_config.NumberColumn(
                format="%.2f %%"
            ),
            "Rappel (%)": st.column_config.NumberColumn(
                format="%.2f %%"
            ),
            "F1-score (%)": st.column_config.NumberColumn(
                format="%.2f %%"
            ),
            "AUC (%)": st.column_config.NumberColumn(
                format="%.2f %%"
            )
        }
    )

    st.divider()

    # ========================================================
    # VISUALISATION
    # ========================================================

    st.subheader("📈 Visualisation des performances")

    indicateur = st.selectbox(
        "Sélectionnez un indicateur",
        [
            "Accuracy (%)",
            "Précision (%)",
            "Rappel (%)",
            "F1-score (%)",
            "AUC (%)"
        ]
    )

    graphique = (
        resultats[
            ["Modèle", indicateur]
        ]
        .set_index("Modèle")
    )

    st.bar_chart(
        graphique
    )

    st.divider()

    # ========================================================
    # MODÈLE RETENU
    # ========================================================

    st.subheader("🎯 Modèle retenu")

    st.success(
        "La **Régression Logistique** est le modèle retenu "
        "pour la prédiction du désabonnement dans cette application."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "F1-score",
            "81,44 %"
        )

    with col2:

        st.metric(
            "AUC",
            "95,20 %"
        )

    with col3:

        st.metric(
            "Accuracy",
            "88,86 %"
        )

    st.markdown(
        """
        Le modèle retenu sera utilisé dans la rubrique
        **🔮 Prédiction** afin d'estimer le risque de
        désabonnement d'un client à partir de ses caractéristiques.
        """
    )

    st.divider()

    # ========================================================
    # INFORMATIONS DU MODÈLE
    # ========================================================

    st.subheader("⚙️ Paramètres du modèle retenu")

    parametres = pd.DataFrame({

        "Paramètre": [
            "Modèle",
            "Solveur",
            "C",
            "Validation"
        ],

        "Valeur": [
            "Régression Logistique",
            "lbfgs",
            "1",
            "Validation croisée stratifiée à 5 plis"
        ]
    })

    st.dataframe(
        parametres,
        use_container_width=True,
        hide_index=True
    )
# ============================================================
# PRÉDICTION
# ============================================================


# ============================================================
# PRÉDICTION DU DÉSABONNEMENT
# ============================================================

elif menu == "🔮 Prédiction":

    st.title("🔮 Prédiction du désabonnement client")

    st.markdown(
        """
        Renseignez les caractéristiques du client pour estimer
        son risque de désabonnement à l'aide du modèle entraîné.
        """
    )

    # --------------------------------------------------------
    # CHARGEMENT DU MODÈLE
    # --------------------------------------------------------
    import joblib

    # Utiliser le modèle déjà chargé au début de app.py
    modele_prediction = modele

    colonnes = list(modele_prediction.feature_names_in_)

    if "depense_mensuelle_jeux" in colonnes:
        st.error(
        "Le modèle chargé contient encore la variable "
        "'depense_mensuelle_jeux'. Vérifiez le fichier .pkl."
        )
        st.stop()

    # --------------------------------------------------------
    # MODALITÉS CATÉGORIELLES
    # À ADAPTER AUX DONNÉES D'ENTRAÎNEMENT
    # --------------------------------------------------------
    
    modalites = {

        "smartphone": [
            "Oui", "Non"
        ],

        "multisim": [
            "Oui", "Non"
        ],

        "plusieurs_ligne_telephonie_fixe": [
            "Oui", "Non"
        ],

        "type_internet": [
            "sobox_fibre", "sobox_fixe", "Non"
        ],

        "services_messageries": [
            "Oui", "Non", "service internet indisponible"
        ],

        "sauvegarde_contact_en_ligne": [
            "Oui", "Non", "service internet indisponible"
        ],

        "acces_connexion_4G_mobile": [
            "Oui", "Non", "service internet indisponible"
        ],

        "abonnement_musique": [
            "Oui", "Non", "service internet indisponible"
        ],

        "abonnement_tv": [
            "Oui", "Non", "service internet indisponible"
        ],

        "souscription_sva_jeux": [
            "Oui", "Non", "service internet indisponible"
        ],

        "utlisation_orange_money": [
            "Oui", "Non"
        ]
    }


    # --------------------------------------------------------
    # FORMULAIRE
    # --------------------------------------------------------
    with st.form("formulaire_prediction"):

        st.subheader("1. Profil du client")

        c1, c2 = st.columns(2)

        with c1:

            anciennete = st.number_input(
                "Ancienneté (mois)",
                min_value=0,
                value=12,
                step=1
            )

        with c2:
            smartphone = st.selectbox(
                "Smartphone", modalites["smartphone"]
            )
            multisim = st.selectbox(
                "Multi-SIM", modalites["multisim"]
            )

        st.subheader("2. Services souscrits")

        services = {}

        variables_services = [
            "plusieurs_ligne_telephonie_fixe",
            "type_internet",
            "services_messageries",
            "sauvegarde_contact_en_ligne",
            "acces_connexion_4G_mobile",
            "abonnement_musique",
            "abonnement_tv",
            "souscription_sva_jeux",
            "utlisation_orange_money"
        ]

        for i in range(0, len(variables_services), 2):
            c1, c2 = st.columns(2)

            v1 = variables_services[i]

            with c1:
                services[v1] = st.selectbox(
                    v1.replace("_", " ").capitalize(),
                    modalites[v1],
                    key="pred_" + v1
                )

            if i + 1 < len(variables_services):
                v2 = variables_services[i + 1]

                with c2:
                    services[v2] = st.selectbox(
                        v2.replace("_", " ").capitalize(),
                        modalites[v2],
                        key="pred_" + v2
                    )

        st.subheader("3. Dépenses mensuelles")

        c1, c2 = st.columns(2)

        with c1:
            depense_data = st.number_input(
                "Dépense mensuelle data",
                min_value=0.0,
                value=0.0,
                step=100.0
            )

        with c2:
            depense_credit = st.number_input(
                "Dépense mensuelle crédit",
                min_value=0.0,
                value=0.0,
                step=100.0
            )

        lancer = st.form_submit_button(
            "🔍 Prédire le désabonnement",
            use_container_width=True,
            type="primary"
        )

    # --------------------------------------------------------
    # PRÉDICTION
    # --------------------------------------------------------
    if lancer:

        donnees = {
            "anciennete_telco_mois": anciennete,
            "smartphone": smartphone,
            "multisim": multisim,
            **services,
            "depence_mensuelle_data": depense_data,
            "depense_mensuelle_dredit": depense_credit
        }

        try:
            # Vérifier la présence des 18 variables
            variables_manquantes = [
                col for col in colonnes
                if col not in donnees
            ]

            if variables_manquantes:
                raise ValueError(
                    f"Variables manquantes : {variables_manquantes}"
                )

            # Respecter exactement les noms et l'ordre attendus
            df_client = pd.DataFrame(
                [donnees],
                columns=colonnes
            )

            prediction = modele_prediction.predict(
                df_client
            )[0]

            classes = list(modele_prediction.classes_)
            probabilites = modele_prediction.predict_proba(
                df_client
            )[0]

            # Identifier la classe positive
            classe_positive = next(
                (
                    c for c in classes
                    if str(c).strip().lower()
                    in ["oui", "1", "true"]
                ),
                None
            )

            st.divider()
            st.subheader("📊 Résultat de la prédiction")

            if str(prediction).strip().lower() in [
                "oui", "1", "true"
            ]:
                st.error(
                    "⚠️ Le modèle prédit un désabonnement."
                )
            else:
                st.success(
                    "✅ Le modèle ne prédit pas de désabonnement."
                )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Classe prédite",
                    str(prediction)
                )

            with col2:
                if classe_positive is not None:
                    indice = classes.index(classe_positive)
                    proba = float(probabilites[indice]) * 100

                    st.metric(
                        "Probabilité de désabonnement",
                        f"{proba:.2f} %"
                    )

                    st.progress(
                        min(100, max(0, int(round(proba))))
                    )
                else:
                    st.warning(
                        "La classe positive n'est pas identifiée. "
                        f"Classes disponibles : {classes}. "
                        "Vérifiez le codage de la cible."
                    )

        except Exception as e:
            st.error(
                f"Erreur lors de la prédiction : {e}"
            )


# ============================================================
# PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "Customer Churn Predict • "
    "Master 2 – Statistique et Informatique Décisionnelle"
)