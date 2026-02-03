import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
import string
import re

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Survey Analysis - Showcase", layout="wide")

# --- INITIALISATION NLTK ---
@st.cache_resource
def setup_nltk():
    nltk.download('stopwords')
    nltk.download('punkt')

setup_nltk()

# --- FONCTION DE NETTOYAGE TEXTE ---
def nettoyer_texte(texte):
    if pd.isna(texte) or texte == "": return ""
    texte = str(texte).lower()  # Passage en minuscules 
    texte = texte.translate(str.maketrans("", "", string.punctuation))  # Suppression ponctuation 
    
    # Stopwords français + mots spécifiques à exclure 
    stopwords_fr = set(stopwords.words('french'))
    mots_a_exclure = {"jours", "deux", "avoir", "faire", "être", "très", "bien", "peu", "comme", "plus"}
    stopwords_fr.update(mots_a_exclure)
    
    mots = texte.split()  # Tokenisation 
    mots = [mot for mot in mots if mot not in stopwords_fr]  # Filtrage 
    return " ".join(mots)

# --- CHARGEMENT DES DONNÉES ---
@st.cache_data
def load_data():
    try:
        # Lecture du fichier CSV avec séparateur point-virgule [cite: 1]
        df = pd.read_csv("Questionnaire.csv", sep=";")
        
        # Nettoyage automatique des noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Sélection des colonnes utiles (à partir de l'index 5)
        df_utile = df.iloc[:, 5:].copy()
        
        return df_utile
    except Exception as e:
        st.error(f"Erreur de lecture du fichier : {e}")
        return None

df = load_data()

# --- INTERFACE ---
if df is not None:
    st.title("📊 Analyse de l'Enquête Stratégique")
    
    # --- BARRE LATÉRALE (SIDEBAR) ---
    st.sidebar.markdown("### 🛠️ Menu")
    st.sidebar.info("📂 **Mode Vitrine Actif**\n\nDonnées `Questionnaire.csv` pré-chargées avec succès.")
    st.sidebar.metric("Réponses totales", len(df))

    tabs = st.tabs(["📉 Satisfaction Globale", "📋 Synthèse par Question", "🎯 Analyses Spécifiques", "☁️ Text Mining", "🧪 Labo Interactif"])

    # --- TAB 1 : RÉPARTITION GLOBALE ---
    with tabs[0]:
        st.subheader("Analyse globale des réponses")
        reponses_possibles = ["Pas du tout satisfait", "Peu satisfait", "Satisfait", "Très satisfait"]
        
        # Comptage global [cite: 3]
        valeurs_counts = df.apply(pd.Series.value_counts).fillna(0)
        valeurs_existantes = [rep for rep in reponses_possibles if rep in valeurs_counts.index]
        
        if valeurs_existantes:
            satisfaction_globale = valeurs_counts.loc[valeurs_existantes].sum(axis=1)
            satisfaction_globale_percent = (satisfaction_globale / satisfaction_globale.sum() * 100).round(2)
            
            # Taille réduite (7, 3.5 au lieu de 10, 5)
            fig, ax = plt.subplots(figsize=(7, 3.5))
            satisfaction_globale_percent.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title("Répartition globale (%)", fontsize=10)
            ax.set_ylabel("Pourcentage")
            plt.xticks(rotation=45, fontsize=9)
            st.pyplot(fig)
        else:
            st.warning("Données de satisfaction non détectées.")

    # --- TAB 2 : SYNTHÈSE PAR QUESTION ---
    with tabs[1]:
        st.subheader("Détail par indicateur")
        colonnes_a_ignorer = [
            'Quel a été votre moment préféré des deux jours et pourquoi ?',
            'Quels aspects pourraient être améliorés pour une prochaine édition ?'
        ]
        colonnes_questions = [col for col in df.columns if col not in colonnes_a_ignorer]
        
        choix_q = st.selectbox("Choisir une question :", colonnes_questions)
        
        counts = df[choix_q].value_counts()
        pourcentages = (counts / counts.sum() * 100).round(2)
        
        col_left, col_right = st.columns([1, 1.5]) # Ajustement des largeurs
        with col_left:
            st.write(f"**Stats :** {choix_q}")
            st.table(pd.DataFrame({"Nombre": counts, "%": pourcentages}))
        
        with col_right:
            # Taille réduite pour le barplot
            fig_q, ax_q = plt.subplots(figsize=(6, 3.5))
            sns.barplot(x=pourcentages.index, y=pourcentages.values, palette="mako", ax=ax_q)
            ax_q.set_title("Répartition par catégorie", fontsize=10)
            plt.xticks(rotation=45, ha="right", fontsize=8)
            st.pyplot(fig_q)

    # --- TAB 3 : ANALYSES SPÉCIFIQUES ---
    with tabs[2]:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Clarté des Objectifs")
            col_binaire = 'Les objectifs de la journée stratégique étaient-ils clairs pour vous ?'
            if col_binaire in df.columns:
                fig_pie, ax_pie = plt.subplots(figsize=(5, 4))
                df[col_binaire].value_counts(normalize=True).plot.pie(
                    autopct="%1.1f%%", colors=["#66c2a5", "#fc8d62"], 
                    startangle=90, wedgeprops=dict(width=0.5), ax=ax_pie, textprops={'fontsize': 9}
                )
                ax_pie.set_ylabel("")
                st.pyplot(fig_pie)
        
        with col2:
            st.subheader("Satisfaction Durée")
            col_duree = "La durée globale des deux jours vous a-t-elle convenu ?"
            if col_duree in df.columns:
                counts_d = df[col_duree].value_counts(normalize=True) * 100
                fig_d, ax_d = plt.subplots(figsize=(5, 4))
                sns.barplot(x=counts_d.index, y=counts_d.values, palette="viridis", ax=ax_d)
                ax_d.set_ylabel("%")
                plt.xticks(fontsize=8)
                st.pyplot(fig_d)

    # --- TAB 4 : TEXT MINING ---
    with tabs[3]:
        st.subheader("Nuages de mots")
        
        col_pref = "Quel a été votre moment préféré des deux jours et pourquoi ?"
        col_amelio = "Quels aspects pourraient être améliorés pour une prochaine édition ?"
        
        df["Texte_Prefere"] = df[col_pref].fillna("").apply(nettoyer_texte)
        df["Texte_Amelioration"] = df[col_amelio].fillna("").apply(nettoyer_texte)
        
        option_nuage = st.radio("Sélection :", ["Global", "Moments préférés", "Améliorations"], horizontal=True)
        
        if option_nuage == "Moments préférés":
            texte_final = " ".join(df["Texte_Prefere"])
            titre_f = "Moments préférés"
        elif option_nuage == "Améliorations":
            texte_final = " ".join(df["Texte_Amelioration"])
            titre_f = "Aspects à améliorer"
        else:
            texte_final = " ".join(df["Texte_Prefere"]) + " " + " ".join(df["Texte_Amelioration"])
            titre_f = "Vue Globale"

        if texte_final.strip():
            wc = WordCloud(background_color="white", width=800, height=400, colormap="coolwarm").generate(texte_final)
            fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

    # --- TAB 5 : LABO INTERACTIF ---
    with tabs[4]:
        st.subheader("🧪 Démo de l'algorithme")
        exemple_sale = "C'était TRÈS BIEN, mais peut-être un peu trop long pour deux jours. Il faudrait faire PLUS de pauses !!!"
        user_text = st.text_area("Texte brut à traiter (Vous pouvez aussi allez chercher le bout de texte de votre choix et le coller dans le champs ci dessou) :", value=exemple_sale, height=100)
        
        if st.button("Lancer le traitement"):
            clean_res = nettoyer_texte(user_text)
            st.markdown("**Texte après nettoyage :**")
            st.code(clean_res)
            
            if len(clean_res.split()) > 0:
                wc_test = WordCloud(background_color="black", width=400, height=200).generate(clean_res)
                fig_test, ax_test = plt.subplots(figsize=(5, 2.5))
                ax_test.imshow(wc_test)
                ax_test.axis("off")
                st.pyplot(fig_test)

else:
    st.error("Fichier 'Questionnaire.csv' introuvable.")