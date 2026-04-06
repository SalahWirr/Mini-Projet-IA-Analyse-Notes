import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="IA Analyse des Notes", layout="wide")

st.title("🎓 Analyse Intelligente des Notes Étudiantes")
st.markdown("Ce projet prédit la note finale (G3) en fonction du temps d'étude et des notes précédentes.")

# --- 1. CHARGEMENT ET NETTOYAGE DES DONNÉES ---
@st.cache_data
def load_data():
    # Remplacer par le chemin de ton fichier CSV
    # On utilise sep=';' car c'est le format standard du dataset Student Performance
    df = pd.read_csv('student-mat.csv', sep=';')
    # Sélection des colonnes pertinentes pour le projet
    cols = ['studytime', 'failures', 'absences', 'G1', 'G2', 'G3']
    return df[cols]

try:
    df = load_data()
    
    # --- 2. PARTIE IA : VISUALISATION ---
    st.sidebar.header("📊 Analyse des Données")
    if st.sidebar.checkbox("Afficher le Dataset"):
        st.subheader("Aperçu des données")
        st.write(df.head())

    if st.sidebar.checkbox("Afficher les Corrélations"):
        st.subheader("Corrélation entre les variables")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    # --- 3. ENTRAÎNEMENT DU MODÈLE ---
    X = df[['studytime', 'failures', 'absences', 'G1', 'G2']]
    y = df['G3']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Évaluation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    # --- 4. INTERFACE UTILISATEUR (PRÉDICTION) ---
    st.divider()
    st.header("🔮 Prédire la Note Finale (G3)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("Informations sur l'étudiant")
        g1 = st.number_input("Note Période 1 (G1) / 20", min_value=0, max_value=20, value=12)
        g2 = st.number_input("Note Période 2 (G2) / 20", min_value=0, max_value=20, value=11)
        study = st.slider("Temps d'étude par semaine (1: <2h, 4: >10h)", 1, 4, 2)

    with col2:
        st.info("Historique")
        failures = st.number_input("Nombre d'échecs passés", min_value=0, max_value=4, value=0)
        absences = st.number_input("Nombre d'absences", min_value=0, max_value=93, value=2)

    if st.button("Calculer la Prédiction"):
        # Préparation des données pour le modèle
        input_data = np.array([[study, failures, absences, g1, g2]])
        prediction = model.predict(input_data)[0]
        
        # Affichage du résultat
        st.success(f"### Note finale prédite : **{prediction:.2f} / 20**")
        
        # Logique métier simple
        if prediction < 10:
            st.error("⚠️ Risque d'échec : Un renforcement scolaire est recommandé.")
        else:
            st.balloons()
            st.write("✅ L'étudiant est sur la bonne voie.")

    # --- 5. PERFORMANCE DU MODÈLE ---
    with st.expander("Voir les détails techniques du modèle (Scikit-Learn)"):
        st.write(f"Précision du modèle (R² Score) : **{r2:.2f}**")
        st.write("Algorithme : Régression Linéaire")

except FileNotFoundError:
    st.error("❌ Erreur : Fichier 'student-mat.csv' introuvable. Place le fichier dans le même dossier que le code.")