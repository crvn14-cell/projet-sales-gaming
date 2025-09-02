import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
import unicodedata
import numpy as np
import re

# ────────────────────────────────────────────────
# APP STREAMLIT : DOSSIER UBISOFT
# ────────────────────────────────────────────────
st.set_page_config(page_title="Dossier Ubisoft", page_icon="🎮", layout="wide")

# 🔧 1) Ajoute "Conclusion" à la navigation (mets à jour ta liste existante)
page = st.sidebar.radio(
    "Aller vers :",
    [
        "Introduction",
        "Analyse financière comparative",
        "Analyse des performances des jeux Ubisoft",
        "Perception et critique : la rupture avec les joueurs",
        "Conclusion",  # ← ajoute cette ligne
    ]
)


# ────────────────────────────────────────────────
# Helpers communs
# ────────────────────────────────────────────────
def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFKD', str(s)) if not unicodedata.combining(c))

def norm_col(c: str) -> str:
    c = strip_accents(str(c)).lower().strip()
    c = re.sub(r'[\s\-_\/]+', ' ', c)
    return c

def clean_numeric(x):
    """Convertit vers float en nettoyant espaces/insécables/virgules/texte ; NaN -> 0."""
    if pd.isna(x):
        return 0.0
    s = str(x).strip()
    if s in {"", "-", "NA", "N/A", "na", "n/a", "None", "null"}:
        return 0.0
    s = (s.replace('\u202f', '')
           .replace('\xa0', '')
           .replace(' ', '')
           .replace(',', '.'))
    try:
        return float(s)
    except Exception:
        return 0.0

# Chargement unique du CSV global pour toutes les pages
@st.cache_data
def load_finance_data():
    fname = "Finance_Finale.csv"
    candidates = [
        dict(sep=",", encoding="utf-8"),
        dict(sep=";", encoding="utf-8"),
        dict(sep=",", encoding="utf-8-sig"),
        dict(sep=";", encoding="utf-8-sig"),
        dict(sep=",", encoding="cp1252"),
        dict(sep=";", encoding="cp1252"),
    ]
    last_err = None
    for opt in candidates:
        try:
            df = pd.read_csv(fname, **opt, engine="python")
            if df.shape[1] >= 2:
                return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Impossible de lire {fname} : {last_err}")

# ────────────────────────────────────────────────
# PAGE 1 : INTRODUCTION
# ────────────────────────────────────────────────
if page == "Introduction":
    st.image("imaubi.jpeg", use_column_width=True, caption="Ubisoft — Franchises et univers emblématiques")
    st.title(" 🎮 Ubisoft — Introduction")
    
    introduction = """
    Ubisoft est l’un des plus grands éditeurs de jeux vidéo au monde, reconnu pour ses franchises emblématiques telles que *Assassin's Creed*, *Far Cry*, *Just Dance*, *Rainbow Six* ou encore *The Division*. Fondée en 1986 par les frères Guillemot, l’entreprise a longtemps incarné le savoir-faire vidéoludique français. Introduite en Bourse en 1996, Ubisoft connaît une croissance spectaculaire pendant plus de deux décennies, atteignant un sommet historique en 2018 avec une action valorisée à plus de **100 €**.

    Depuis ce pic, Ubisoft semble enchaîner les difficultés. En **2024**, sa capitalisation boursière a chuté de plus de **6 milliards d’euros**, une dégringolade qui suscite de nombreuses interrogations. Est-elle le reflet d’une crise sectorielle généralisée ? Est-elle symptomatique de difficultés internes à l’entreprise ?

    À travers ce projet de *data analyse*, notre objectif est de comprendre les facteurs internes ayant contribué à ce déclin, en collectant des données financières, critiques et comportementales. Nous chercherons également à identifier les signaux faibles et les ruptures stratégiques pouvant expliquer cette trajectoire descendante, tout en proposant des pistes d’amélioration.
    """
    st.markdown(introduction)
    st.divider()
    
# ────────────────────────────────────────────────
# PAGE 2 : ANALYSE FINANCIÈRE COMPARATIVE
# ────────────────────────────────────────────────
elif page == "Analyse financière comparative":
    st.title(" 📊 Analyse Financière Comparative")
    st.caption("Évolution historique, comparaison avec le secteur et analyse des tendances.")

    try:
        df_finance = load_finance_data()
    except Exception as e:
        st.error(f"⚠️ Chargement CSV échoué : {e}")
        st.stop()

   
    # ── PARTIE 1 : Historique Ubisoft (texte + image locale)
    st.markdown("""
    ## 1. Analyse financière comparative  
    ### Une trajectoire spectaculaire puis un effondrement brutal…

    L’action **Ubisoft** a connu une évolution remarquable depuis son introduction en Bourse le **1er juillet 1996**. Dès le premier jour de cotation, le titre est multiplié par **252**, porté par l’engouement pour l’industrie vidéoludique et une forte levée de fonds.  
    Cette dynamique s’est poursuivie pendant plus d’une décennie, atteignant un **pic historique de plus de 100 € en juillet 2018**. Cette valorisation exceptionnelle reflète alors la solidité des franchises d’Ubisoft, telles que *Assassin’s Creed*, *Far Cry*, *Rainbow Six Siege* et *The Division*, ainsi que la stratégie de l’éditeur axée sur les **jeux à monde ouvert** et à fort contenu **solo/multijoueur**.  
    Entre **2014 et 2018**, les résultats financiers sont en nette progression, avec un chiffre d’affaires passant de **1,4** à **2,2 milliards de dollars** et une amélioration significative des marges. À cette période, **Tencent** entre au capital, consolidant l’image d’Ubisoft comme acteur stratégique à l’international.  
    Pourtant, dès **2019**, les résultats commencent à décevoir : plusieurs jeux ne répondent pas aux attentes, les retards s’accumulent, et la rentabilité s’effrite. Le titre entame alors une **chute prolongée** : en **cinq ans**, l’action perd plus de **80 % de sa valeur**. Depuis 2018, cela représente une **perte de capitalisation boursière d’environ 9 milliards d’euros**.
    """)

        # ── PARTIE 1 : Historique Ubisoft (chargement auto de l'image)
    st.subheader(" Évolution historique du cours de l’action Ubisoft")

    @st.cache_data(show_spinner=False)
    def _find_ubisoft_chart() -> str | None:
        base = Path(__file__).parent
        # chemins les plus probables (mets l'image à la racine ou dans assets/images/static)
        candidates = [
            base / "ubisoft_google_finance.png",
            base / "assets" / "ubisoft_google_finance.png",
            base / "images" / "ubisoft_google_finance.png",
            base / "static" / "ubisoft_google_finance.png",
            base / "Capture d'écran 2025-08-25 141139.png",
            base / "assets" / "Capture d'écran 2025-08-25 141139.png",
            base / "images" / "Capture d'écran 2025-08-25 141139.png",
            base / "static" / "Capture d'écran 2025-08-25 141139.png",
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        # recherche de secours par motif
        for folder in [base, base / "assets", base / "images", base / "static"]:
            for pat in ("ubisoft*finance*.*", "Ubisoft*Finance*.*", "Capture d'écran 2025-08-25 141139.*"):
                for p in folder.glob(pat):
                    return str(p)
        return None

    img_path = _find_ubisoft_chart()

    if img_path:
        st.image(
            img_path,
            caption="Évolution historique du cours Ubisoft — Source : Google Finance (EPA : UBI)",
            use_container_width=True
        )
    else:
        st.error(
            "Image introuvable. Place le fichier **ubisoft_google_finance.png** "
            "ou **Capture d'écran 2025-08-25 141139.png** à la racine du projet "
            "ou dans **./assets/**, **./images/** ou **./static/**."
        )

    st.divider()

    # ── PARTIE 2 : Performance relative au secteur (texte + courbes comparatives)
    st.markdown("""
    ## 2. Une performance financière en retrait

    Pour mieux comprendre le contexte du déclin d’Ubisoft, nous avons comparé l’évolution de son cours de Bourse à celle des deux principaux **ETF sectoriels** dédiés au jeu vidéo : **ESPO** (*VanEck Video Gaming & eSports*) et **HERO** (*Global X Video Games & Esports ETF*). Ces deux indices regroupent les plus grands éditeurs mondiaux du secteur.

    L’analyse sur les **cinq dernières années** met en évidence une **divergence nette**. Si les trois courbes suivent une trajectoire globalement similaire jusqu’en **2022** — marquée par une baisse partagée —, les dynamiques s’opposent par la suite : **ESPO** repart à la hausse dès **2023**, amorçant une phase de croissance continue, tandis qu’**Ubisoft** poursuit son repli, atteignant même un **point bas autour de 10 € en 2024**.

    Cette dissociation entre l’évolution du marché global et celle d’Ubisoft confirme que **le problème semble spécifique à l’entreprise**. La performance boursière d’Ubisoft ne peut pas être attribuée à une crise sectorielle : au contraire, l’industrie du jeu vidéo **continue de progresser dans son ensemble**. Cela renforce l’hypothèse d’une **crise interne** — un axe que nous tenterons d’explorer dans les chapitres suivants.
    """)

    st.subheader(" Comparaison Ubisoft vs ETF ESPO & HERO")
    df_etf = pd.DataFrame({
        "Année":   [2020, 2021, 2022, 2023, 2024],
        "Ubisoft": [85,   75,   50,   25,   10],
        "ESPO":    [100,  110,  90,   120,  140],
        "HERO":    [95,   105,  85,   115,  135],
    })
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(df_etf["Année"], df_etf["Ubisoft"], marker="o", color="red",   label="Ubisoft")
    ax2.plot(df_etf["Année"], df_etf["ESPO"],    marker="o", color="green", label="ESPO")
    ax2.plot(df_etf["Année"], df_etf["HERO"],    marker="o", color="blue",  label="HERO")
    ax2.set_title("Évolution du cours Ubisoft vs ESPO & HERO (5 dernières années)", fontsize=14)
    ax2.set_xlabel("Année"); ax2.set_ylabel("Valeur normalisée (base 100)")
    ax2.grid(True, linestyle="--", alpha=0.6); ax2.legend()
    ax2.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    st.pyplot(fig2)
    st.divider()

    # ── PARTIE 3 : CA cumulé par éditeur (lecture robuste depuis df_finance)
    st.markdown("""
    **Observation complémentaire.**  
    Sur la période étudiée, le **chiffre d’affaires cumulé** d’Ubisoft est **le plus faible parmi les éditeurs majeurs du secteur**. 
    """)
    st.subheader(" Chiffre d’affaires cumulé par éditeur (2018–2024) ")

    raw = df_finance.copy()
    norm_map = {c: norm_col(c) for c in raw.columns}
    df = raw.rename(columns=norm_map)

    cumu_alias = [
        'ca cumule (m€)','ca cumule','chiffre daffaires cumule (m€)',
        'chiffre daffaires cumule','revenue cumule (m€)','revenu cumule (m€)','revenue total (m€)'
    ]
    year_cols_cols = [c for c in df.columns if re.fullmatch(r'(?:fy)?(20(1[8-9]|2[0-4]))', c)]
    if not year_cols_cols:
        year_cols_cols = [c for c in df.columns if re.search(r'20(1[8-9]|2[0-4])', c)]
    year_line_alias = ['annee','year','date']
    editor_alias = ['editeur','éditeur','publisher','societe','entreprise','company','studio','nom','compagnie']
    editor_col = next((c for c in df.columns if c in editor_alias), None)
    if editor_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                editor_col = c; break
    if editor_col is None:
        st.error("Colonne 'Editeur' introuvable dans Finance_Finale.csv"); st.stop()

    cumu_col = next((c for c in df.columns if c in cumu_alias), None)
    if cumu_col:
        out = pd.DataFrame({
            "Éditeur": df[editor_col],
            "CA cumulé (M€)": df[cumu_col].apply(clean_numeric)
        })
    elif year_cols_cols:
        tmp = df[[editor_col] + year_cols_cols].copy()
        for c in year_cols_cols:
            tmp[c] = tmp[c].apply(clean_numeric)
        total = tmp[year_cols_cols].sum(axis=1)
        if pd.notna(total.max()) and total.max() > 1_000_000:
            total = total / 1_000_000.0
        out = pd.DataFrame({"Éditeur": tmp[editor_col], "CA cumulé (M€)": total})
    else:
        annee_col = next((c for c in df.columns if c in year_line_alias or "annee" in c or "year" in c or "date" in c), None)
        ca_candidates = [c for c in df.columns if any(k in c for k in ['chiffre','revenue','revenu','sales','ca '])]
        ca_col = ca_candidates[0] if ca_candidates else None
        if not (annee_col and ca_col):
            st.error("Colonnes nécessaires non trouvées (Année + Chiffre d'affaires)."); st.stop()
        work = df[[editor_col, annee_col, ca_col]].copy()
        work['__year__'] = pd.to_datetime(work[annee_col], errors='coerce').dt.year
        work['__ca__'] = work[ca_col].apply(clean_numeric)
        mask = work['__year__'].between(2018, 2024, inclusive='both')
        grouped = (work[mask].groupby(editor_col, as_index=False)['__ca__'].sum()
                   .rename(columns={editor_col:"Éditeur", '__ca__':"CA cumulé (M€)"}))
        out = grouped
        if pd.notna(out["CA cumulé (M€)"].max()) and out["CA cumulé (M€)"].max() > 1_000_000:
            out["CA cumulé (M€)"] = out["CA cumulé (M€)"] / 1_000_000.0

    out = out.dropna(subset=["Éditeur"]).copy()
    out["CA cumulé (M€)"] = pd.to_numeric(out["CA cumulé (M€)"], errors='coerce').fillna(0)
    out = out.sort_values("CA cumulé (M€)", ascending=False)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    bars = ax3.bar(out["Éditeur"], out["CA cumulé (M€)"])
    ax3.set_title("Chiffre d'affaires cumulé par éditeur de 2018 à 2024", fontsize=14)
    ax3.set_xlabel("Éditeurs"); ax3.set_ylabel("Chiffre d'affaires cumulé (M€)")
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    for b, v in zip(bars, out["CA cumulé (M€)"]):
        ax3.annotate(f"{int(round(v)):,}".replace(",", " "),
                     xy=(b.get_x() + b.get_width()/2, v),
                     xytext=(0, 5), textcoords="offset points",
                     ha="center", va="bottom", fontsize=9)
    st.pyplot(fig3)

    # ────────────────────────────────────────────────
    # Graphiques comparatifs CA, Résultat net, Masse salariale (interactifs)
    # ────────────────────────────────────────────────
    st.divider()
    st.markdown("""
    Plus préoccupant encore, **le chiffre d’affaires d’Ubisoft n’évolue quasiment pas**, alors que la majorité des **concurrents** (*Sony Interactive Entertainment, Electronic Arts, Bandai Namco*, etc.) affichent **une croissance continue**.  
    Cette **stagnation** est un **signal d’alerte fort**, d’autant plus que le **marché global du jeu vidéo** est, lui, **en croissance**.
    """)
    st.subheader("Évolution du chiffre d’affaires (2018–2024) ")

    def _to_long(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.rename(columns={c: unicodedata.normalize("NFKD", str(c)).encode("ascii","ignore").decode().strip().lower()
                                   for c in df_in.columns})
        year_cols = [c for c in df.columns if re.fullmatch(r'(?:fy)?(20(1[8-9]|2[0-4]))', c)]
        if not year_cols:
            year_cols = [c for c in df.columns if re.search(r'20(1[8-9]|2[0-4])', c)]
        ed_col = next((c for c in df.columns if c in
                       ["editeur","publisher","entreprise","societe","company","studio","nom","compagnie"]), None)
        if ed_col is None:
            for c in df.columns:
                if df[c].dtype == object:
                    ed_col = c; break
        if ed_col is None:
            raise ValueError("Colonne éditeur introuvable.")

        if year_cols:
            long = df[[ed_col] + year_cols].copy().melt(id_vars=[ed_col], var_name="annee", value_name="valeur")
            long["annee"] = long["annee"].astype(str).str.extract(r'(20\d{2})').astype(int)
            long["valeur"] = long["valeur"].apply(clean_numeric)
            long = long.rename(columns={ed_col:"Editeur"})
            return long

        an_col = next((c for c in df.columns if c in ["annee","year","date"] or "annee" in c or "year" in c or "date" in c), None)
        val_col = next((c for c in df.columns if any(k in c for k in ["chiffre","revenue","revenu","sales","ca"])), None)
        if an_col is None or val_col is None:
            raise ValueError("Colonnes requises non trouvées (Année + CA).")
        long = df[[ed_col, an_col, val_col]].copy().rename(columns={ed_col:"Editeur", an_col:"annee", val_col:"valeur"})
        long["annee"] = pd.to_datetime(long["annee"], errors="coerce").dt.year
        long["valeur"] = long["valeur"].apply(clean_numeric)
        return long

    df_multi_raw = df_finance.copy()
    data_long = _to_long(df_multi_raw)
    data_long = data_long.dropna(subset=["Editeur","annee"])
    data_long = data_long[(data_long["annee"]>=2018) & (data_long["annee"]<=2024)]
    data_long["valeur"] = data_long["valeur"].apply(clean_numeric)

    editeurs_dispos = sorted(data_long["Editeur"].unique().tolist())
    col_a, col_b = st.columns([2,1])
    with col_a:
        sel_editeurs = st.multiselect("Éditeurs à afficher :", editeurs_dispos, default=editeurs_dispos)
    with col_b:
        years_min, years_max = int(data_long["annee"].min()), int(data_long["annee"].max())
        an_range = st.slider("Plage d’années :", min_value=years_min, max_value=years_max, value=(2018, 2024), step=1)

    dfp = data_long[(data_long["Editeur"].isin(sel_editeurs)) &
                    (data_long["annee"].between(an_range[0], an_range[1]))].copy()
    full_index = pd.MultiIndex.from_product([sorted(set(sel_editeurs)), list(range(an_range[0], an_range[1]+1))],
                                            names=["Editeur", "annee"])
    dfp = (dfp.groupby(["Editeur", "annee"], as_index=False)["valeur"].sum()
              .set_index(["Editeur","annee"])
              .reindex(full_index)
              .fillna(0.0)
              .reset_index())

    if dfp.empty:
        st.warning("Aucune donnée pour la sélection actuelle.")
    else:
        annees = sorted(dfp["annee"].unique().tolist())
        publishers = sel_editeurs
        n_pub = len(publishers)
        total_width = 0.8
        bar_width = total_width / max(n_pub,1)
        x = list(range(len(annees)))
        fig, ax = plt.subplots(figsize=(10,6))
        for i, pub in enumerate(publishers):
            y_vals = [float(dfp[(dfp["Editeur"]==pub) & (dfp["annee"]==a)]["valeur"].sum()) for a in annees]
            offsets = [xx + (i - (n_pub-1)/2)*bar_width for xx in x]
            ax.bar(offsets, y_vals, width=bar_width, label=pub)
        ax.set_xticks(x); ax.set_xticklabels(annees, rotation=0)
        ax.set_title("Évolution du chiffre d’affaires (M€) par éditeur", fontsize=14)
        ax.set_xlabel("Année"); ax.set_ylabel("Chiffre d'affaires (M€)")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(ncol=2, fontsize=9)
        st.pyplot(fig)

    # Résultat net — similaire
    st.divider()
    st.markdown("""
    **Le résultat net cumulé d’Ubisoft est en net retrait par rapport à ses pairs**, alors que la majorité de ses concurrents restent **bénéficiaires** sur la même période.  
    Ce **déficit chronique** montre qu’Ubisoft ne parvient pas à **transformer ses ventes en valeur** pour ses actionnaires, et que sa **structure de coûts** n’est pas suffisamment maîtrisée.
    """)
    st.subheader(" Résultat net (M€) — évolution 2018–2024")

    def to_long_metric(df_in: pd.DataFrame, metric_keywords) -> pd.DataFrame:
        df = df_in.rename(columns={c: norm_col(c) for c in df_in.columns})
        year_cols = [c for c in df.columns if re.fullmatch(r'(?:fy)?(20(1[8-9]|2[0-4]))', c)]
        if not year_cols:
            year_cols = [c for c in df.columns if re.search(r'20(1[8-9]|2[0-4])', c)]
        ed_col = next((c for c in df.columns if c in
                       ["editeur","éditeur","publisher","entreprise","societe","company","studio","nom","compagnie"]), None)
        if ed_col is None:
            for c in df.columns:
                if df[c].dtype == object:
                    ed_col = c; break
        if ed_col is None:
            raise ValueError("Colonne éditeur introuvable.")
        if year_cols:
            long = df[[ed_col] + year_cols].copy().melt(id_vars=[ed_col], var_name="annee", value_name="valeur")
            long["annee"] = long["annee"].astype(str).str.extract(r'(20\d{2})').astype(int)
            long["valeur"] = long["valeur"].apply(clean_numeric)
            long = long.rename(columns={ed_col: "Editeur"})
            return long
        an_col = next((c for c in df.columns if c in ["annee","year","date"] or "annee" in c or "year" in c or "date" in c), None)
        val_col = next((c for c in df.columns if any(k in c for k in metric_keywords)), None)
        if an_col is None or val_col is None:
            raise ValueError("Colonnes requises non trouvées (Année + Résultat net).")
        long = df[[ed_col, an_col, val_col]].copy().rename(columns={ed_col:"Editeur", an_col:"annee", val_col:"valeur"})
        long["annee"] = pd.to_datetime(long["annee"], errors="coerce").dt.year
        long["valeur"] = long["valeur"].apply(clean_numeric)
        return long

    data_profit = to_long_metric(df_finance.copy(), ["resultat","résultat","net income","profit","benefice","bénéfice"])
    data_profit = data_profit.dropna(subset=["Editeur","annee"])
    data_profit = data_profit[(data_profit["annee"]>=2018) & (data_profit["annee"]<=2024)]
    data_profit["valeur"] = data_profit["valeur"].apply(clean_numeric)

    editeurs_p = sorted(data_profit["Editeur"].unique().tolist())
    col1, col2 = st.columns([2,1])
    with col1:
        sel_ed_p = st.multiselect("Éditeurs à afficher :", editeurs_p, default=editeurs_p, key="prof_ed")
    with col2:
        y_min, y_max = int(data_profit["annee"].min()), int(data_profit["annee"].max())
        an_range_p = st.slider("Plage d’années :", min_value=y_min, max_value=y_max, value=(2018, 2024), step=1, key="prof_year")

    dfp_p = data_profit[(data_profit["Editeur"].isin(sel_ed_p)) &
                        (data_profit["annee"].between(an_range_p[0], an_range_p[1]))].copy()
    idx_full = pd.MultiIndex.from_product([sorted(set(sel_ed_p)), list(range(an_range_p[0], an_range_p[1]+1))],
                                          names=["Editeur","annee"])
    dfp_p = (dfp_p.groupby(["Editeur","annee"], as_index=False)["valeur"].sum()
                .set_index(["Editeur","annee"])
                .reindex(idx_full)
                .fillna(0.0)
                .reset_index())

    if dfp_p.empty:
        st.warning("Aucune donnée pour la sélection actuelle.")
    else:
        annees_p = sorted(dfp_p["annee"].unique().tolist())
        pubs_p = sel_ed_p; n_pub_p = len(pubs_p)
        total_w = 0.8; bw = total_w / max(n_pub_p,1); x = list(range(len(annees_p)))
        figp, axp = plt.subplots(figsize=(10,6))
        for i, pub in enumerate(pubs_p):
            yv = [float(dfp_p[(dfp_p["Editeur"]==pub) & (dfp_p["annee"]==a)]["valeur"].sum()) for a in annees_p]
            offs = [xx + (i - (n_pub_p-1)/2)*bw for xx in x]
            axp.bar(offs, yv, width=bw, label=pub)
        axp.axhline(0, color="black", linewidth=1)
        axp.set_xticks(x); axp.set_xticklabels(annees_p, rotation=0)
        axp.set_title("Résultat net (M€) par éditeur", fontsize=14)
        axp.set_xlabel("Année"); axp.set_ylabel("Résultat net (M€)")
        axp.grid(axis="y", linestyle="--", alpha=0.5)
        axp.legend(ncol=2, fontsize=9)
        st.pyplot(figp)

    # Masse salariale
    st.divider()
    st.markdown("""
    L’un des écarts les plus marquants est observé au niveau de la **masse salariale**.  
    **Ubisoft** emploie un volume de salariés **comparable** à celui d’**Activision Blizzard**, mais ses **performances financières** sont nettement **inférieures**.  
    Par exemple, **Electronic Arts** opère avec **environ un tiers de personnel en moins**, tout en générant un **chiffre d’affaires** et un **résultat net** largement supérieurs.
    """)
    st.subheader(" Masse salariale (M€) — évolution 2018–2024")

    def _to_long_payroll(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.rename(columns={c: norm_col(c) for c in df_in.columns})
        year_cols = [c for c in df.columns if re.fullmatch(r'(?:fy)?(20(1[8-9]|2[0-4]))', c)]
        if not year_cols:
            year_cols = [c for c in df.columns if re.search(r'20(1[8-9]|2[0-4])', c)]
        ed_col = next((c for c in df.columns if c in
                      ["editeur","éditeur","publisher","entreprise","societe","company","studio","nom","compagnie"]), None)
        if ed_col is None:
            for c in df.columns:
                if df[c].dtype == object:
                    ed_col = c; break
        if ed_col is None:
            raise ValueError("Colonne éditeur introuvable.")
        if year_cols:
            long = df[[ed_col] + year_cols].copy().melt(id_vars=[ed_col], var_name="annee", value_name="valeur")
            long["annee"] = long["annee"].astype(str).str.extract(r'(20\d{2})').astype(int)
            long["valeur"] = long["valeur"].apply(clean_numeric)
            long = long.rename(columns={ed_col:"Editeur"})
            return long
        an_col = next((c for c in df.columns if c in ["annee","year","date"] or "annee" in c or "year" in c or "date" in c), None)
        val_col = next((c for c in df.columns if any(k in c for k in
                   ["masse salariale","payroll","personnel","staff cost","wages","salaires","salary","coût du personnel","cout du personnel"])), None)
        if an_col is None or val_col is None:
            raise ValueError("Colonnes requises non trouvées (Année + Masse salariale).")
        long = df[[ed_col, an_col, val_col]].copy().rename(columns={ed_col:"Editeur", an_col:"annee", val_col:"valeur"})
        long["annee"] = pd.to_datetime(long["annee"], errors="coerce").dt.year
        long["valeur"] = long["valeur"].apply(clean_numeric)
        return long

    payroll_long = _to_long_payroll(df_finance.copy())
    payroll_long = payroll_long.dropna(subset=["Editeur","annee"])
    payroll_long = payroll_long[(payroll_long["annee"]>=2018) & (payroll_long["annee"]<=2024)]
    payroll_long["valeur"] = payroll_long["valeur"].apply(clean_numeric)

    editeurs_pay = sorted(payroll_long["Editeur"].unique().tolist())
    c1, c2 = st.columns([2,1])
    with c1:
        sel_editeurs_pay = st.multiselect("Éditeurs à afficher :", editeurs_pay, default=editeurs_pay, key="pay_ed")
    with c2:
        y_min_p, y_max_p = int(payroll_long["annee"].min()), int(payroll_long["annee"].max())
        an_range_pay = st.slider("Plage d’années :", min_value=y_min_p, max_value=y_max_p, value=(2018, 2024), step=1, key="pay_year")

    dfp_pay = payroll_long[(payroll_long["Editeur"].isin(sel_editeurs_pay)) &
                           (payroll_long["annee"].between(an_range_pay[0], an_range_pay[1]))].copy()
    full_idx_pay = pd.MultiIndex.from_product([sorted(set(sel_editeurs_pay)),
                                               list(range(an_range_pay[0], an_range_pay[1]+1))],
                                              names=["Editeur","annee"])
    dfp_pay = (dfp_pay.groupby(["Editeur","annee"], as_index=False)["valeur"].sum()
                    .set_index(["Editeur","annee"])
                    .reindex(full_idx_pay)
                    .fillna(0.0)
                    .reset_index())

    if dfp_pay.empty:
        st.warning("Aucune donnée pour la sélection actuelle (masse salariale).")
    else:
        annees_pay = sorted(dfp_pay["annee"].unique().tolist())
        pubs_pay = sel_editeurs_pay
        n_pub_pay = len(pubs_pay)
        total_w = 0.8; bw = total_w / max(n_pub_pay,1); x = list(range(len(annees_pay)))
        figp2, axp2 = plt.subplots(figsize=(10,6))
        for i, pub in enumerate(pubs_pay):
            y_vals = [float(dfp_pay[(dfp_pay["Editeur"]==pub) & (dfp_pay["annee"]==a)]["valeur"].sum()) for a in annees_pay]
            offs = [xx + (i - (n_pub_pay-1)/2)*bw for xx in x]
            axp2.bar(offs, y_vals, width=bw, label=pub)
        axp2.set_xticks(x); axp2.set_xticklabels(annees_pay, rotation=0)
        axp2.set_title("Évolution de la masse salariale (M€) par éditeur", fontsize=14)
        axp2.set_xlabel("Année"); axp2.set_ylabel("Masse salariale (M€)")
        axp2.grid(axis="y", linestyle="--", alpha=0.5)
        axp2.legend(ncol=2, fontsize=9)
        st.pyplot(figp2)

    # Bulles : CA↔Résultat (taille = masse salariale) + Masse salariale ↔ Effectif
    st.divider()
    st.subheader(" Résultat net vs Chiffre d’affaires ")
    st.caption("Les deux graphiques ci-dessous utilisent les mêmes données centralisées.")

    def _normalize_columns_for_panel(df_in: pd.DataFrame) -> pd.DataFrame:
        df = df_in.rename(columns={c: norm_col(c) for c in df_in.columns})
        ed_col = next((c for c in df.columns if c in
                       ["editeur","éditeur","publisher","entreprise","societe","company","studio","nom","compagnie"]), None)
        if ed_col is None:
            for c in df.columns:
                if df[c].dtype == object:
                    ed_col = c; break
        if ed_col is None:
            raise ValueError("Colonne éditeur introuvable.")
        KEYWORDS = {
            "ca": ["chiffre","sales","revenue","revenu","ca"],
            "profit": ["resultat","résultat","net income","profit","benefice","bénéfice"],
            "payroll": ["masse salariale","payroll","personnel","staff cost","wages","salaires","salary","coût du personnel","cout du personnel"],
            "headcount": ["effectif","headcount","employe","employee","staff"]
        }
        def find_col(dfcols, words):
            return next((c for c in dfcols if any(w in c for w in words)), None)

        an_col = next((c for c in df.columns if c in ["annee","year","date"] or "annee" in c or "year" in c or "date" in c), None)
        ca_col      = find_col(df.columns, KEYWORDS["ca"])
        profit_col  = find_col(df.columns, KEYWORDS["profit"])
        payroll_col = find_col(df.columns, KEYWORDS["payroll"])
        headc_col   = find_col(df.columns, KEYWORDS["headcount"])

        panel = pd.DataFrame({"Editeur": df[ed_col]})
        if an_col is not None:
            panel["annee"] = pd.to_datetime(df[an_col], errors="coerce").dt.year
        else:
            panel["annee"] = np.nan

        if ca_col:      panel["ca"]       = df[ca_col].apply(clean_numeric)
        if profit_col:  panel["profit"]   = df[profit_col].apply(clean_numeric)
        if payroll_col: panel["payroll"]  = df[payroll_col].apply(clean_numeric)
        if headc_col:   panel["headcount"]= df[headc_col].apply(clean_numeric)
        for col in ["ca","profit","payroll","headcount"]:
            if col not in panel.columns:
                panel[col] = 0.0
        return panel

    panel = _normalize_columns_for_panel(df_finance.copy())
    panel = panel.dropna(subset=["Editeur"])
    if panel["annee"].notna().any():
        panel = panel[(panel["annee"].between(2018, 2024, inclusive="both")) | panel["annee"].isna()].copy()
    for c in ["ca","profit","payroll","headcount"]:
        panel[c] = panel[c].apply(clean_numeric)

    colsA, colsB, colsC = st.columns([2,1,1])
    with colsA:
        editeurs_sel = st.multiselect("Éditeurs :", sorted(panel["Editeur"].unique()),
                                      default=sorted(panel["Editeur"].unique()))
    with colsB:
        if panel["annee"].notna().any():
            y_min2, y_max2 = int(panel["annee"].min()), int(panel["annee"].max())
            an_range2 = st.slider("Années :", min_value=y_min2, max_value=y_max2, value=(max(2018,y_min2), min(2024,y_max2)), step=1)
        else:
            an_range2 = (2018, 2024)
    with colsC:
        size_scale = st.slider("Échelle des bulles (masse salariale)", 0.1, 2.0, 0.7, 0.1)
        alpha_pts  = st.slider("Transparence", 0.2, 1.0, 0.8, 0.1)

    if panel["annee"].notna().any():
        dfp_panel = panel[(panel["Editeur"].isin(editeurs_sel)) &
                          (panel["annee"].between(an_range2[0], an_range2[1]))].copy()
    else:
        dfp_panel = panel[panel["Editeur"].isin(editeurs_sel)].copy()

    if dfp_panel.empty:
        st.warning("Aucune donnée pour la sélection actuelle.")
    else:
        fig_b, ax_b = plt.subplots(figsize=(9.5, 6.5))
        for ed in sorted(dfp_panel["Editeur"].unique()):
            d = dfp_panel[dfp_panel["Editeur"] == ed]
            ax_b.scatter(d["ca"], d["profit"], s=np.sqrt(d["payroll"].clip(lower=0))*(10*size_scale),
                         alpha=alpha_pts, label=ed)
        ax_b.set_title("Résultat net (M€) en fonction du chiffre d’affaires (M€) — taille = masse salariale", fontsize=13)
        ax_b.set_xlabel("Chiffre d’affaires (M€)")
        ax_b.set_ylabel("Résultat net (M€)")
        ax_b.grid(True, linestyle="--", alpha=0.4)
        ax_b.legend(ncol=2, fontsize=9, frameon=True)
        st.pyplot(fig_b)

        st.subheader(" Masse salariale vs Effectif total (2018–2024)")
        fig_c, ax_c = plt.subplots(figsize=(9.5, 6.0))
        for ed in sorted(dfp_panel["Editeur"].unique()):
            d = dfp_panel[dfp_panel["Editeur"] == ed]
            ax_c.scatter(d["headcount"], d["payroll"], alpha=alpha_pts, label=ed)
        ax_c.set_title("Coût de la masse salariale (M€) en fonction de l’effectif total", fontsize=13)
        ax_c.set_xlabel("Effectif total (personnes)")
        ax_c.set_ylabel("Masse salariale (M€)")
        ax_c.grid(True, linestyle="--", alpha=0.4)
        ax_c.legend(ncol=2, fontsize=9, frameon=True)
        st.pyplot(fig_c)

# ────────────────────────────────────────────────
# PAGE 3 : ANALYSE DES PERFORMANCES DES JEUX UBISOFT
# ────────────────────────────────────────────────
elif page == "Analyse des performances des jeux Ubisoft":
    st.title("🎯 Analyse des performances des jeux Ubisoft")
    st.markdown("""
    Au-delà des indicateurs financiers globaux, l’analyse du **catalogue** d’Ubisoft révèle des éléments structurants.
    En étudiant la fréquence des sorties, les revenus par jeu et le volume total de titres publiés, on observe des tendances claires.
    """)

    st.subheader("2.1. Une stratégie axée sur le volume")
    st.markdown("""
    Ubisoft se distingue de ses concurrents par une production particulièrement **prolifique** :
    le **nombre de jeux publiés** chaque année est largement supérieur à la moyenne du secteur.
    Cette stratégie s’appuie sur une **capacité de développement répartie** sur plusieurs studios dans le monde,
    ainsi que sur des **processus industriels bien rodés**.
    """)

    # ---------- IMPORTS ----------
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import unicodedata, re

    # ---------- Helpers ----------
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"[\s\-_\/]+", " ", s).strip().lower()

    def _to_float(x):
        if pd.isna(x): return 0.0
        s = (str(x).strip()
                .replace("\u202f","").replace("\xa0","")
                .replace(" ", "").replace(",", "."))
        s = re.sub(r"(€|eur|euros|millions?)$", "", s, flags=re.I)
        try: return float(s)
        except: return 0.0

    def apply_light_theme(fig, *, title_text, x_title, y1_title, y2_title=None):
        layout_kwargs = dict(
            title=dict(text=title_text, x=0.5, xanchor="center",
                       font=dict(size=20, color="#000")),
            plot_bgcolor="white",
            paper_bgcolor="white",
            hovermode="x unified",
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#000", borderwidth=1, font=dict(size=13, color="#111")),
            margin=dict(l=70, r=70, t=70, b=70),
            xaxis=dict(title=x_title, tickfont=dict(size=12, color="#000"),
                       showline=True, linecolor="#000", linewidth=2, showgrid=True, gridcolor="rgba(0,0,0,0.25)"),
            yaxis=dict(title=y1_title, tickfont=dict(size=12, color="#000"),
                       showline=True, linecolor="#000", linewidth=2, showgrid=True, gridcolor="rgba(0,0,0,0.25)"),
            height=460
        )
        if y2_title:
            layout_kwargs["yaxis2"] = dict(
                title=y2_title, overlaying="y", side="right",
                showline=True, linecolor="#000", linewidth=2,
                tickfont=dict(size=12, color="#000")
            )
        fig.update_layout(**layout_kwargs)
        return fig

    def _find_col(dfcols, keywords):
        for c in dfcols:
            if any(k in c for k in keywords):
                return c
        return None

    # ---------- Chargement du CSV éditeurs ----------
    try:
        df_ed = pd.read_csv("editeurs_nettoyées.csv")
    except Exception as e:
        st.error(f"⚠️ Impossible de charger le CSV : {e}")
        st.stop()

    # ---------- Vérification/alignement colonnes ----------
    expected_cols = ["Nom", "Jeux publiés", "Revenu total (milliards)"]
    for col in expected_cols:
        if col not in df_ed.columns:
            cands = [c for c in df_ed.columns if _norm(c) == _norm(col)]
            if cands:
                df_ed.rename(columns={cands[0]: col}, inplace=True)
            else:
                st.error(f"⚠️ Colonne manquante : '{col}' — colonnes disponibles : {list(df_ed.columns)}")
                st.stop()

    # ---------- Harmonisation des éditeurs ----------
    mapping_editeurs = {
        "ubisoft": "Ubisoft",
        "electronic arts": "Electronic Arts", "ea": "Electronic Arts",
        "sega": "SEGA",
        "square enix": "Square Enix",
        "bandai": "Bandai Namco", "bandai namco": "Bandai Namco",
        "take two": "Take-Two", "take-two": "Take-Two", "2k": "Take-Two", "2k games": "Take-Two"
    }
    df_ed["Editeur"] = df_ed["Nom"].apply(lambda x: mapping_editeurs.get(_norm(x), None))

    # ---------- Filtrer les 6 éditeurs du projet ----------
    editeurs_cibles = ["Ubisoft", "Electronic Arts", "SEGA", "Square Enix", "Bandai Namco", "Take-Two"]
    dff = df_ed[df_ed["Editeur"].isin(editeurs_cibles)].copy()
    if dff.empty:
        st.error("⚠️ Aucune donnée trouvée pour les 6 éditeurs attendus.")
        st.write("Éditeurs trouvés :", sorted(df_ed["Nom"].unique()))
        st.stop()

    # ---------- Agrégation ----------
    dff = (dff.groupby("Editeur", as_index=False)
              .agg(**{
                  "Jeux publiés": ("Jeux publiés", "sum"),
                  "Revenu total (milliards)": ("Revenu total (milliards)", "sum")
              }))

    # ---------- Graphique 1 : Volume vs Revenu total ----------
    dff["Couleur"] = dff["Editeur"].apply(lambda n: "Ubisoft" if n == "Ubisoft" else "Autres")
    fig1 = px.scatter(
        dff,
        x="Jeux publiés",
        y="Revenu total (milliards)",
        text="Editeur",
        color="Couleur",
        color_discrete_map={"Ubisoft": "#e53935", "Autres": "#6e6e6e"},
        size=[26] * len(dff),
        size_max=28,
        labels={
            "Jeux publiés": "Nombre de jeux publiés (somme 2018–2024)",
            "Revenu total (milliards)": "Revenu total (en milliards d'€)"
        }
    )
    fig1.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=14),
        title=dict(
            text="Relation entre le nombre de jeux publiés et le revenu total",
            font=dict(size=18, color="black"),
            x=0.5, xanchor="center"
        ),
        margin=dict(l=100, r=60, t=60, b=70),
    )
    fig1.update_xaxes(showgrid=True, gridcolor="#CCCCCC", zeroline=False,
                      automargin=True, title_font=dict(size=16, color="black"),
                      tickfont=dict(size=14, color="black"))
    fig1.update_yaxes(showgrid=True, gridcolor="#CCCCCC", zeroline=False,
                      automargin=True, title_standoff=22,
                      title_font=dict(size=16, color="black"),
                      tickfont=dict(size=14, color="black"))

    # Padding dynamique
    x_min, x_max = dff["Jeux publiés"].min(), dff["Jeux publiés"].max()
    y_min, y_max = dff["Revenu total (milliards)"].min(), dff["Revenu total (milliards)"].max()
    dx = max(6, (x_max - x_min) * 0.08)
    dy = max(0.4, (y_max - y_min) * 0.10)
    fig1.update_xaxes(range=[x_min - dx, x_max + dx])
    fig1.update_yaxes(range=[max(0, y_min - dy), y_max + dy])
    fig1.update_traces(textposition="top center", cliponaxis=False)

    st.plotly_chart(fig1, use_container_width=True)

    # ────────────────────────────────────────────────
    # 2.2 — Relation volume de jeux / revenu moyen par jeu
    # + Titre demandé "Dépendance aux Blogs Busters" (sans créer de section)
    # ────────────────────────────────────────────────
    

    st.markdown("""
    Cependant, cette approche atteint ses **limites**. En effet, le **revenu moyen généré par jeu** reste inférieur à celui de concurrents
    comme **Electronic Arts** ou **Take-Two**, qui publient moins de titres mais maximisent la **rentabilité** de chacun.
    """)

    if "Revenu moyen par jeu (M€)" not in dff.columns:
        dff["Revenu moyen par jeu (M€)"] = (dff["Revenu total (milliards)"] * 1000.0) / dff["Jeux publiés"]

    fig2 = px.scatter(
        dff,
        x="Jeux publiés",
        y="Revenu moyen par jeu (M€)",
        text="Editeur",
        color="Couleur",
        color_discrete_map={"Ubisoft": "#e53935", "Autres": "#6e6e6e"},
        size=[26] * len(dff),
        size_max=28,
        labels={"Jeux publiés": "Jeux publiés", "Revenu moyen par jeu (M€)": "Revenu moyen par jeu (M€)"}
    )
    fig2.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black", size=14),
        title=dict(
            text="Relation volume de jeux / Revenu moyen par jeu (Ubisoft vs concurrents)",
            font=dict(size=18, color="black"),
            x=0.5, xanchor="center"
        ),
        margin=dict(l=100, r=60, t=70, b=70),
    )
    fig2.update_xaxes(showgrid=True, gridcolor="#CCCCCC", zeroline=False,
                      automargin=True, title_font=dict(size=16, color="black"),
                      tickfont=dict(size=14, color="black"))
    fig2.update_yaxes(showgrid=True, gridcolor="#CCCCCC", zeroline=False,
                      automargin=True, title_standoff=22,
                      title_font=dict(size=16, color="black"),
                      tickfont=dict(size=14, color="black"))

    # Padding dynamique
    x2_min, x2_max = dff["Jeux publiés"].min(), dff["Jeux publiés"].max()
    y2_min, y2_max = dff["Revenu moyen par jeu (M€)"].min(), dff["Revenu moyen par jeu (M€)"].max()
    dx2 = max(8, (x2_max - x2_min) * 0.10)
    dy2 = max(10, (y2_max - y2_min) * 0.12)
    fig2.update_xaxes(range=[x2_min - dx2, x2_max + dx2])
    fig2.update_yaxes(range=[max(0, y2_min - dy2), y2_max + dy2])
    fig2.update_traces(textposition="top center", cliponaxis=False)

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    Le **choix de miser sur la quantité** plutôt que sur la **rentabilité par titre** semble diluer l'impact de chaque sortie,
    et affaiblit la capacité de l'éditeur à transformer ses lancements en **succès retentissants**.
    """)

    # ────────────────────────────────────────────────
    # Séries annuelles : Revenus & Unités vendues — textes AVANT/APRÈS identiques au doc
    # ────────────────────────────────────────────────
    # Chargement données jeux détaillées
    @st.cache_data
    def _load_jeux_csv():
        fname = "Jeux_final.csv"
        tries = [
            dict(sep=",", encoding="utf-8"),
            dict(sep=";", encoding="utf-8"),
            dict(sep=",", encoding="utf-8-sig"),
            dict(sep=";", encoding="utf-8-sig"),
            dict(sep="\t", encoding="utf-8"),
            dict(sep=",", encoding="cp1252"),
            dict(sep=";", encoding="cp1252"),
        ]
        last = None
        for opt in tries:
            try:
                df = pd.read_csv(fname, engine="python", **opt)
                if df.shape[1] >= 2:
                    return df
            except Exception as e:
                last = e
        raise RuntimeError(f"Impossible de lire {fname} : {last}")

    try:
        dfj_raw = _load_jeux_csv()
    except Exception as e:
        st.error(f"⚠️ Chargement de `Jeux_final.csv` impossible : {e}")
        st.stop()

    dfj = dfj_raw.copy()
    dfj.columns = [_norm(c) for c in dfj.columns]

    col_date   = _find_col(dfj.columns, ["premiere publication", "publication", "release", "date"])
    col_rev    = _find_col(dfj.columns, ["revenus", "sales", "chiffre d", "ca"])
    col_units  = _find_col(dfj.columns, ["unites", "unités", "units", "copies", "ventes"])
    col_medhrs = _find_col(dfj.columns, ["temps median", "temps médian", "median playtime", "temps de jeu"])

    if not (col_date and col_rev and col_units):
        st.warning("Colonnes non reconnues automatiquement. Sélectionne-les ci-dessous.")
        with st.expander("Diagnostic colonnes CSV"):
            st.write(list(dfj_raw.columns))
        cols = list(dfj.columns)
        col_date  = st.selectbox("Colonne date de première publication", cols, index=cols.index(col_date) if col_date else 0)
        col_rev   = st.selectbox("Colonne revenus (millions)", cols, index=cols.index(col_rev) if col_rev else 0)
        col_units = st.selectbox("Colonne unités vendues (millions)", cols, index=cols.index(col_units) if col_units else 0)

    work = dfj[[col_date, col_rev, col_units]].rename(columns={
        col_date:  "date_pub",
        col_rev:   "revenus_m",
        col_units: "unites_m"
    }).copy()

    work["Année"] = pd.to_datetime(work["date_pub"], errors="coerce").dt.year
    mask_na = work["Année"].isna()
    if mask_na.any():
        work.loc[mask_na, "Année"] = work.loc[mask_na, "date_pub"].astype(str).str.extract(r"(20\d{2})", expand=False)
    work["Année"] = pd.to_numeric(work["Année"], errors="coerce").astype("Int64")

    work["Revenus (millions)"] = work["revenus_m"].apply(_to_float)
    work["Unités vendues (millions)"] = work["unites_m"].apply(_to_float)

    annual = (work.dropna(subset=["Année"])
                   .groupby("Année", as_index=False)
                   .agg({"Revenus (millions)": "sum", "Unités vendues (millions)": "sum"}))

    if annual.empty:
        st.error("Aucune donnée exploitable après agrégation. Vérifie le mapping des colonnes.")
        st.stop()

    # Texte AVANT (identique au doc)
    st.markdown("""
    Dans un marché de plus en plus **concurrentiel** où **l’attention des joueurs est limitée**, ce positionnement nuit à la 
    **visibilité** des titres d’Ubisoft et limite leur capacité à s’imposer comme des **références durables**.
    """)
    st.divider()

    # ——— Titre de section (même niveau que 2.1) ———
    st.subheader("2.2. Une dépendance à quelques blockbusters")



    # Graphique séries annuelles
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=annual["Année"], y=annual["Revenus (millions)"],
                              mode="lines+markers", name="Revenus (millions)", line=dict(width=3)))
    fig3.add_trace(go.Scatter(x=annual["Année"], y=annual["Unités vendues (millions)"],
                              mode="lines+markers", name="Unités vendues (millions)", yaxis="y2", line=dict(width=3)))
    apply_light_theme(fig3, title_text="Évolution des revenus et unités vendues par année",
                      x_title="Année", y1_title="Revenus (millions)", y2_title="Unités vendues (millions)")
    st.plotly_chart(fig3, use_container_width=True)

    # Texte APRÈS (identique au doc)
    st.markdown("""
    Les données révèlent une **forte concentration des revenus** sur quelques titres phares, notamment entre **2014** et **2015**, 
    période marquée par le lancement d’épisodes majeurs d’*Assassin’s Creed* et de *Far Cry*.  
    Cette dynamique s’est progressivement **estompée**.

    On voit que **chaque jeu contribue fortement** à la volatilité des revenus, confirmant que le **succès d’Ubisoft repose davantage 
    sur quelques blockbusters** que sur l’ensemble de son catalogue.

    Depuis **2019**, Ubisoft peine à reproduire de tels succès, probablement impactée par la **crise Covid-19**.  
    Le recul de ses revenus annuels s’explique en partie par l’absence de **nouveaux hits d’ampleur**, capables de porter à eux seuls 
    l’exercice financier. Ce phénomène met en lumière une **dépendance excessive** à des **franchises anciennes**, sans réelle relève.

    Ainsi, malgré un **catalogue étendu**, la **majorité des titres publiés** génèrent **peu de valeur individuellement**.  
    Ce **déséquilibre fragilise la résilience** du modèle économique, qui repose de fait sur une **minorité de succès critiques et commerciaux**.

    Cette observation se **confirme après analyse croisée** du **temps de jeu médian** et des **revenus générés par année**, 
    qui révèle la même dépendance aux quelques titres à fort impact.
    """)

    # ---------- Temps médian vs revenus (si dispo) ----------
    if not col_medhrs:
        col_medhrs = _find_col(dfj.columns, ["temps median", "temps médian", "median playtime", "temps de jeu"])
    if col_medhrs:
        df_tm = dfj_raw.copy()
        rename_map = {}
        for c in df_tm.columns:
            nc = _norm(c)
            if nc == _norm(col_date):   rename_map[c] = "Première publication"
            if nc == _norm(col_rev):    rename_map[c] = "Revenus (millions)"
            if nc == _norm(col_medhrs): rename_map[c] = "Temps médian de jeu (heures)"
        df_tm.rename(columns=rename_map, inplace=True)

        needed = {"Première publication","Revenus (millions)","Temps médian de jeu (heures)"}
        if needed.issubset(df_tm.columns):
            df_tm["Première publication"] = pd.to_datetime(df_tm["Première publication"], errors="coerce")
            df_tm["Année"] = df_tm["Première publication"].dt.year
            df_tm["Revenus (millions)"] = df_tm["Revenus (millions)"].apply(_to_float)
            df_tm["Temps médian de jeu (heures)"] = df_tm["Temps médian de jeu (heures)"].apply(_to_float)

            df_year = (df_tm.dropna(subset=["Année"])
                            .groupby("Année", as_index=False)
                            .agg({"Revenus (millions)": "sum",
                                  "Temps médian de jeu (heures)": "median"}))

            if not df_year.empty:
                fig4 = go.Figure()
                fig4.add_trace(go.Scatter(x=df_year["Année"], y=df_year["Temps médian de jeu (heures)"],
                                          mode="lines+markers", name="Temps médian de jeu (h)", line=dict(width=3)))
                fig4.add_trace(go.Scatter(x=df_year["Année"], y=df_year["Revenus (millions)"],
                                          mode="lines+markers", name="Revenus (millions)", yaxis="y2", line=dict(width=3)))
                apply_light_theme(fig4, title_text="Temps médian de jeu vs Revenus (annuel)",
                                  x_title="Année", y1_title="Temps médian de jeu (h)", y2_title="Revenus (millions)")
                st.plotly_chart(fig4, use_container_width=True)
            # ---------- Bloc texte à insérer entre les deux graphiques ----------
    st.markdown("""
En effet, **entre 2005 et 2014**, Ubisoft enregistre une **croissance continue** de ces deux indicateurs,
avec un **pic autour de 2014–2015**. Comme expliqué précédemment, cette période correspond à la sortie de
**titres majeurs**, souvent bien accueillis par la **critique** comme par les **joueurs**, et jouant un
**rôle structurant** dans les revenus de l’entreprise.

**Cependant, après 2018**, les **revenus chutent significativement**, tandis que le **temps de jeu médian
reste élevé**. Ce décalage indique que, malgré une baisse de performance économique, Ubisoft **conserve une
base de joueurs fidèles**, probablement attachés à ses **licences historiques**.

Ce phénomène illustre un **problème de renouvellement d’offre** : Ubisoft **capitalise sur ses anciens succès**,
mais **ne parvient plus à recréer l’élan** des précédentes générations de **blockbusters**.
""")
    st.divider()
    st.subheader("2.3. Des modèles économiques mal exploités")
    st.markdown(
    "Entre 2013 et 2015, Ubisoft parvient à capter l’attention du marché avec plusieurs initiatives Free-to-Play "
    "et des titres à fort potentiel multijoueur (*The Mighty Quest for Epic Loot, Trackmania, Brawlhalla*, etc.)."
)

    # ---------- Modèles économiques (gratuits vs payants) ----------
    col_model = _find_col(dfj.columns, ["modele", "modèle", "business", "monet", "model", "pricing", "f2p", "free", "gratuit"])
    col_price = _find_col(dfj.columns, ["prix", "price"])

    work2 = dfj[[col_date, col_units] + ([col_model] if col_model else []) + ([col_price] if col_price else [])].copy()
    work2.rename(columns={col_date:"date_pub", col_units:"unites"}, inplace=True)
    work2["Année"] = pd.to_datetime(work2["date_pub"], errors="coerce").dt.year
    work2["Unités vendues (millions)"] = work2["unites"].apply(_to_float)

    def _is_free(row) -> bool:
        if col_model:
            s = str(row[col_model]).lower()
            if any(k in s for k in ["free", "gratuit", "f2p", "free-to-play", "free to play"]):
                return True
        if col_price:
            try:
                p = _to_float(row[col_price])
                if p == 0:
                    return True
            except:
                pass
        return False

    work2["Type"] = work2.apply(lambda r: "Jeux gratuits" if _is_free(r) else "Jeux payants", axis=1)

    g = (work2.dropna(subset=["Année"])
              .groupby(["Année","Type"], as_index=False)
              .agg({"Unités vendues (millions)":"sum"}))

    if not g.empty:
        import plotly.graph_objects as go
        wide = g.pivot(index="Année", columns="Type", values="Unités vendues (millions)").fillna(0.0).sort_index()
        fig5 = go.Figure()
        if "Jeux payants" in wide.columns:
            fig5.add_trace(go.Scatter(x=wide.index, y=wide["Jeux payants"],
                                      mode="lines+markers", name="Jeux payants", line=dict(width=3)))
        if "Jeux gratuits" in wide.columns:
            fig5.add_trace(go.Scatter(x=wide.index, y=wide["Jeux gratuits"],
                                      mode="lines+markers", name="Jeux gratuits", line=dict(width=3, dash="dash")))
        apply_light_theme(fig5, title_text="Évolution des unités vendues : Jeux gratuits vs payants",
                          x_title="Année", y1_title="Unités vendues (millions)")
        st.plotly_chart(fig5, use_container_width=True)

        # ---------- Bloc de conclusion (à insérer à la fin du chapitre) ----------
    
    st.markdown("""
    Pourtant, cette **dynamique prometteuse** n’a pas été pérennisée. Le **modèle freemium**, pourtant porteur sur le long terme
    pour d’autres éditeurs (comme *Epic Games* avec *Fortnite*), n’a **jamais été solidement ancré** dans la stratégie
    produit d’Ubisoft.

    Cette **incapacité à renouveler les formats**, à proposer des **expériences économiques innovantes** ou à **s’adapter aux tendances**
    (*abonnement*, *cross-platform*, *multijoueur compétitif*, etc.) **risque d’isoler progressivement Ubisoft** d’une partie de la communauté,
    notamment les **joueurs plus jeunes** ou **plus actifs sur mobile et PC**.
    """)
    st.divider()

# ────────────────────────────────────────────────
# PAGE 4 : PERCEPTION ET CRITIQUE — RUPTURE AVEC LES JOUEURS
# ────────────────────────────────────────────────
elif page == "Perception et critique : la rupture avec les joueurs":
    import pandas as pd
    import numpy as np
    import re
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("🧩 Perception et critique : la rupture avec les joueurs")
    st.markdown("""
    Au-delà des performances financières et des stratégies de développement,  
    l’analyse de la **réception critique** des jeux Ubisoft apporte un éclairage essentiel.  

    En observant les notes attribuées par la **presse spécialisée** et les **joueurs** sur des plateformes comme **Metacritic**,  
    on met en évidence une **communauté de joueurs** qui semble **légèrement plus polarisée**  
    et parfois **prête à noter des jeux à 0**.
    """)


    # ───────── Lecture directe du CSV local
    @st.cache_data
    def load_scores():
        tries = [
            dict(sep=",", encoding="utf-8"),
            dict(sep=";", encoding="utf-8"),
            dict(sep=",", encoding="utf-8-sig"),
            dict(sep=";", encoding="utf-8-sig"),
            dict(sep=",", encoding="cp1252"),
            dict(sep=";", encoding="cp1252"),
        ]
        last_err = None
        for opt in tries:
            try:
                return pd.read_csv("ubisoft_scores.csv", engine="python", **opt)
            except Exception as e:
                last_err = e
        st.error(f"⚠️ Impossible de lire `ubisoft_scores.csv`. Vérifie qu'il est bien placé à côté de `app.py`. Détails : {last_err}")
        st.stop()

    raw = load_scores()

    # ───────── Vérification des colonnes attendues
    expected_cols = {"Press_Score", "Users_Score"}
    if not expected_cols.issubset(raw.columns):
        st.error("⚠️ Le fichier CSV doit contenir les colonnes **Press_Score** et **Users_Score**.")
        st.stop()

    # ───────── Nettoyage des données
    def _to_num(x):
        if pd.isna(x): return np.nan
        s = str(x).strip().lower()
        if s in {"tbd", "na", "n/a", "none", "null", "-", ""}:
            return np.nan
        s = s.replace("\u202f", "").replace("\xa0", "").replace(",", ".")
        s = re.sub(r"[^0-9\.\-]", "", s)
        try:
            return float(s)
        except:
            return np.nan

    df_notes = pd.DataFrame({
        "Press_Score": raw["Press_Score"].apply(_to_num),
        "Users_Score": raw["Users_Score"].apply(_to_num),
    }).dropna()

    # Conversion auto 0–100 → 0–10
    if df_notes["Press_Score"].max() > 10:
        df_notes["Press_Score"] /= 10.0
    if df_notes["Users_Score"].max() > 10:
        df_notes["Users_Score"] /= 10.0

    # Filtre borne [0,10]
    df_notes = df_notes[
        (df_notes["Press_Score"].between(0, 10)) &
        (df_notes["Users_Score"].between(0, 10))
    ]

    if df_notes.empty:
        st.error("⚠️ Aucune donnée exploitable après nettoyage (scores attendus entre 0 et 10).")
        st.stop()

    # ───────── Stats descriptives (COUNT + MEAN uniquement)
    st.subheader(" Statistiques descriptives")
    stats = df_notes.describe().loc[["count", "mean"]].round(3)
    st.dataframe(stats, use_container_width=True)

    # ───────── Graphiques : Presse vs Joueurs
    st.subheader(" Comparaison des distributions")
    x_min, x_max = 0, 10
    sns.set_style("whitegrid")

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Presse
    sns.histplot(df_notes["Press_Score"], bins=20, kde=True, color="#4CAF50", ax=axs[0])
    axs[0].set_title("Distribution des notes de la Presse (sur 10)", fontsize=12)
    axs[0].set_ylabel("Nombre de jeux")
    axs[0].set_xlim(x_min, x_max)

    # Joueurs
    sns.histplot(df_notes["Users_Score"], bins=20, kde=True, color="#87CEEB", ax=axs[1])
    axs[1].set_title("Distribution des notes utilisateurs (sur 10)", fontsize=12)
    axs[1].set_xlabel("Notes")
    axs[1].set_ylabel("Nombre de jeux")
    axs[1].set_xlim(x_min, x_max)

    plt.tight_layout()
    st.pyplot(fig)

    # ───────── Analyse rapide
    st.subheader(" Analyse")
    st.markdown("""
    - **Presse** : notes majoritairement concentrées entre **6 et 8**, reflétant une évaluation globalement positive.
    - **Joueurs** : distribution plus **étalée**, avec davantage de notes très basses → signe d'une **polarisation**.
    - Cet écart révèle une différence de perception : Ubisoft convainc la presse mais divise parfois sa communauté.
    """)
    # ——— Détection Year + agrégations annuelles
    import unicodedata, re
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"[\s\-_\/]+", " ", s).strip().lower()

    def _extract_year_column(df: pd.DataFrame) -> pd.Series | None:
        # 1) colonnes de date
        for c in df.columns:
            n = _norm(c)
            if any(k in n for k in ["release", "premiere", "publication", "date", "year", "annee", "année"]):
                y = pd.to_datetime(df[c], errors="coerce").dt.year
                if y.notna().sum() > 0:
                    return y
        # 2) colonnes numériques déjà en années
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                y = pd.to_numeric(df[c], errors="coerce")
                if ((y >= 1990) & (y <= 2035)).sum() > 0:
                    return y
        return None

    year_series = _extract_year_column(raw)
    if year_series is None:
        st.warning("Aucune colonne de date/année reconnue : les graphiques temporels ne peuvent pas être tracés.")
    else:
        work = pd.DataFrame({
            "Year": year_series,
            "Press_Score": df_notes["Press_Score"].values,  # scores déjà nettoyés/ramenés sur 10
            "Users_Score": df_notes["Users_Score"].values,
        }).dropna()
        work = work[(work["Year"] >= 1995) & (work["Year"] <= 2035)]

        yearly = (work.groupby("Year", as_index=False)
                        .agg(Press=("Press_Score","mean"),
                             Users=("Users_Score","mean"))
                        .sort_values("Year"))

        # ——— Graphique 1 : courbes annuelles
        st.subheader(" Notes moyennes par année — Presse vs Joueurs")
        fig_line, axl = plt.subplots(figsize=(10, 5))
        axl.plot(yearly["Year"], yearly["Press"], marker="o", linewidth=2.2, label="Presse", color="#2E7D32")
        axl.plot(yearly["Year"], yearly["Users"], marker="o", linewidth=2.2, label="Joueurs", color="#FB8C00")
        axl.set_xlabel("Année de sortie"); axl.set_ylabel("Note moyenne (sur 10)")
        axl.grid(True, linestyle="--", alpha=0.35)
        axl.legend(title="Source", frameon=True)

        # repère « décrochage » (si l'année est dans la série)
        if (yearly["Year"] >= 2014).any() and (yearly["Year"] <= 2014).any():
            axl.axvline(2014, color="#757575", linestyle="--", alpha=0.6)
            ymin, ymax = axl.get_ylim()
            axl.text(2014 + 0.2, ymin + 0.05*(ymax-ymin),
                     "Décrochage des notes des joueurs", fontsize=9, color="#616161")

        st.pyplot(fig_line)
        # ——— Graphique 2 : écart moyen annuel (Users − Press)
        st.subheader(" Écart moyen entre notes utilisateurs et presse ")
        delta = yearly.copy()
        delta["Diff"] = delta["Users"] - delta["Press"]

        # couleurs: bleu si positif, dégradé de rouge si négatif
        import matplotlib as mpl
        reds = mpl.cm.get_cmap("Reds")
        neg_vals = delta["Diff"].clip(upper=0).abs()
        if neg_vals.max() == 0:
            neg_norm = np.zeros_like(neg_vals)
        else:
            neg_norm = neg_vals / neg_vals.max()

        colors = []
        for d, nn in zip(delta["Diff"], neg_norm):
            if d >= 0:
                colors.append("#1f77b4")          # bleu (joueurs plus généreux)
            else:
                colors.append(reds(0.35 + 0.55*nn))  # rouge plus sombre si l’écart est grand

        fig_bar, axb = plt.subplots(figsize=(10, 5))
        axb.bar(delta["Year"], delta["Diff"], color=colors, width=0.8, edgecolor="none")
        axb.axhline(0, color="black", linewidth=1)
        axb.set_xlabel("Année de sortie"); axb.set_ylabel("Score delta (Users − Press)")
        axb.grid(axis="y", linestyle="--", alpha=0.35)

        # petite légende manuelle
        from matplotlib.patches import Patch
        legend_elems = [
            Patch(facecolor="#1f77b4", label="Joueurs plus généreux que la presse"),
            Patch(facecolor=reds(0.8), label="Joueurs plus critiques que la presse"),
        ]
        axb.legend(handles=legend_elems, title="Interprétation des couleurs", frameon=True)

        st.pyplot(fig_bar)
    st.markdown("""


Historiquement, les jeux Ubisoft ont reçu des évaluations relativement proches entre la **presse** et les **joueurs**. 
Jusqu’en **2014**, la moyenne des notes utilisateurs est stable autour de **7/10**, tandis que la presse affiche 
généralement des scores entre **7 et 8/10**. Les écarts sont modérés, et les critiques convergent globalement.

À partir de **2015**, une **fracture de perception** commence à se dessiner : les joueurs deviennent plus critiques, 
attribuant des notes **significativement inférieures** à celles de la presse. Cette tendance s’accentue au fil des années, 
jusqu’à atteindre un **écart moyen de –2,3 points** entre les deux types d’évaluateurs en **2022**. Dans certains cas, 
les utilisateurs attribuent des notes **très basses (0 à 4/10)**, souvent motivées par une frustration liée à la 
**qualité technique** ou à la **déception** vis-à-vis des promesses initiales.

La presse, quant à elle, reste globalement **modérée** dans ses notations, avec peu d’évolutions à la baisse. 
Ce décalage persistant entre **qualité perçue par les joueurs** et **reconnaissance critique** devient un marqueur 
structurel de la **crise** que traverse Ubisoft. Il témoigne d’un **désalignement** entre l’expérience réelle des 
utilisateurs et le produit livré, alimenté par des éléments récurrents dans les critiques : **manque d’innovation**, 
**gameplay répétitif**, **bugs techniques**, ou encore **promesses non tenues**.
""")

    # ────────────────────────────────────────────────
    # 3) Top & Flop Ubisoft – Score moyen global (presse + utilisateurs)
    # ────────────────────────────────────────────────
    import re, unicodedata
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.subheader(" Top & Flop Ubisoft – Score moyen global (presse + utilisateurs)")

    # --- Helpers pour retrouver les colonnes "Name", "Platform" et "Year" si elles ne sont pas déjà dans df_notes
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"[\s\-_\/]+", " ", s).strip().lower()

    def _extract_year_column(df: pd.DataFrame) -> pd.Series | None:
        # 1) colonnes de date
        for c in df.columns:
            n = _norm(c)
            if any(k in n for k in ["release", "premiere", "publication", "date", "year", "annee", "année"]):
                y = pd.to_datetime(df[c], errors="coerce").dt.year
                if y.notna().sum() > 0:
                    return y
        # 2) colonnes numériques déjà en années
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                y = pd.to_numeric(df[c], errors="coerce")
                if ((y >= 1990) & (y <= 2035)).sum() > 0:
                    return y
        return None

    # On part de df_notes (déjà nettoyé + ramené sur 10) et du DataFrame brut 'raw' lu en début de page 4
    df_plot = df_notes.copy()

    # Ajoute Year si manquant
    if "Year" not in df_plot.columns:
        y = _extract_year_column(raw)
        if y is not None:
            df_plot["Year"] = y
        else:
            st.error("Impossible d’identifier la colonne Année. Ajoute une colonne 'Year' ou une date de sortie dans le CSV.")
            st.stop()

    # Ajoute Name si manquant
    if "Name" not in df_plot.columns:
        name_col = next((c for c in raw.columns if _norm(c) in {"name","titre","title","game","jeu"} 
                         or any(k in _norm(c) for k in ["name","title","game","jeu","titre"])), None)
        if name_col:
            df_plot["Name"] = raw[name_col]
        else:
            df_plot["Name"] = [f"Jeu {i}" for i in range(len(df_plot))]

    # Ajoute Platform si manquant
    if "Platform" not in df_plot.columns:
        plat_col = next((c for c in raw.columns if any(k in _norm(c) for k in ["platform","console","system"])), None)
        df_plot["Platform"] = raw[plat_col] if plat_col else "N/A"

    # Ajoute Score_Avg si manquant
    if "Score_Avg" not in df_plot.columns:
        df_plot["Score_Avg"] = df_plot[["Press_Score","Users_Score"]].mean(axis=1)

    # --- Ton code adapté à Streamlit ---
    # Filtrer les jeux sortis depuis 2015
    df_notes_recent = df_plot[df_plot["Year"] >= 2015].copy()

    # Top 10 des jeux Ubisoft selon score_avg
    top_avg = (df_notes_recent.sort_values(by="Score_Avg", ascending=False)
               .drop_duplicates(subset=["Name","Platform"])
               .head(10))

    # Flop 10 des jeux Ubisoft selon score_avg
    flop_avg = (df_notes_recent.sort_values(by="Score_Avg", ascending=True)
                .drop_duplicates(subset=["Name","Platform"])
                .head(10))

    # Fusion pour plot
    topflop_avg = pd.concat([top_avg.assign(cat="Top"), flop_avg.assign(cat="Flop")], ignore_index=True)

    if topflop_avg.empty:
        st.warning("Aucune donnée après filtrage (Year ≥ 2015). Vérifie les colonnes Year/Name/Platform.")
    else:
        # Tri de l’affichage (du plus faible au plus fort, puis inversion de l’axe Y pour avoir les meilleurs en haut)
        order_names = topflop_avg.sort_values("Score_Avg", ascending=True)["Name"]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(
            data=topflop_avg,
            y="Name",
            x="Score_Avg",
            hue="cat",
            dodge=False,
            order=order_names,
            palette={"Top": "green", "Flop": "red"},
            ax=ax
        )
        ax.set_title("Top & Flop Ubisoft – Score moyen global (presse + utilisateurs)", fontsize=13)
        ax.set_xlabel("Score moyen (/10)")
        ax.set_ylabel("Jeu")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.legend(title="Catégorie", frameon=True)
        ax.invert_yaxis()  # meilleurs en haut

        # Ajout des valeurs au bout des barres
        for p in ax.patches:
            width = p.get_width()
            y = p.get_y() + p.get_height() / 2
            ax.text(width + 0.05, y, f"{width:.1f}", va="center", fontsize=9)

        plt.tight_layout()
        st.pyplot(fig)
    # ——— Texte d'analyse : Top & Flop Ubisoft 2015–2025
    st.markdown("""
Sur la période **2015–2025**, l’étude des **notes moyennes globales** (*presse + utilisateurs*) met en évidence une
**tendance préoccupante** : les meilleurs jeux Ubisoft récents ne sont pas ceux qui bénéficient du plus fort
**soutien marketing**, ni ceux issus des **franchises historiques**.

Parmi les titres les mieux reçus, on retrouve notamment **Beyond Good & Evil 20th Anniversary Edition** ou
**Prince of Persia: The Lost Crown** – des jeux moins exposés médiatiquement.  
À l’inverse, plusieurs **blockbusters très attendus**, à **gros budget**, échouent à convaincre :
**Ghost Recon Breakpoint**, **The Settlers: New Allies**, ou encore **Just Dance 2024 Edition**
reçoivent des notes particulièrement basses, en décalage avec leurs ambitions.

Ce phénomène appuie davantage sur la **diminution de la confiance des joueurs** envers les grands lancements Ubisoft.
L’un des exemples les plus emblématiques de cette rupture est le cas de **Skull & Bones**, que nous allons analyser
dans la dernière partie.
""")

    # ——— Barre de séparation avant la section 3.3
    st.markdown("---")

    # ——— Partie 3.3 : Le cas Skull & Bones
    st.subheader("3.3. Le cas Skull & Bones : un échec emblématique")

    st.markdown("""
L’épisode le plus marquant de cette rupture entre Ubisoft et sa communauté est incarné par **Skull & Bones**,
considéré comme l’un des plus gros échecs récents de l’éditeur.  
Ce jeu, censé capitaliser sur le succès de *Assassin’s Creed IV: Black Flag* et sur l’engouement pour les
**thématiques pirates**, a connu un **développement chaotique** étalé sur près de **10 ans**.  
À sa sortie, il recueille une **note utilisateur catastrophique de 3/10**, tandis que la presse reste
**modérément indulgente**.

Un **nuage de mots** généré à partir des critiques utilisateurs sur *Metacritic* permet de mettre en lumière
cette perception.  
Les termes les plus fréquents parlent d’eux-mêmes :  
*“boring”*, *“repetitive”*, *“money”*, *“combat”*, *“waste”*, *“gameplay”*, *“disappointing”*, *“Black Flag”*, etc.

Ils illustrent une combinaison de **déception**, **d’ennui** et de **frustration économique**.  
Beaucoup de joueurs font explicitement référence à *Black Flag*, renforçant la comparaison avec un jeu sur une
**thématique proche**, perçu comme **bien mieux réussi**, pourtant sorti **dix ans plus tôt**.
""")
    # ────────────────────────────────────────────────
    # Nuage de mots négatifs — Skull & Bones (stopwords fournis + suppression "skull" et "bones")
    # ────────────────────────────────────────────────
    import sys
    import subprocess
    import string
    import unicodedata
    import re
    import pandas as pd
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    from textblob import TextBlob

    st.subheader(" Nuage de mots des critiques négatives — Skull & Bones")

    # --- Installer automatiquement wordcloud & textblob si manquants
    def _ensure_package(mod_name, pip_name=None):
        try:
            __import__(mod_name)
        except ModuleNotFoundError:
            with st.spinner(f"Installation de `{pip_name or mod_name}`…"):
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name or mod_name])
            __import__(mod_name)

    _ensure_package("wordcloud")
    _ensure_package("textblob")

    # --- Lecture CSV critiques
    @st.cache_data
    def _read_critiques():
        tries = [
            dict(sep=",", encoding="utf-8"),
            dict(sep=";", encoding="utf-8"),
            dict(sep=",", encoding="utf-8-sig"),
            dict(sep=";", encoding="utf-8-sig"),
            dict(sep=",", encoding="cp1252"),
            dict(sep=";", encoding="cp1252"),
            dict(sep="\t", encoding="utf-8"),
        ]
        for path in ["ubisoft_critiques.csv", "data/ubisoft_critiques.csv"]:
            for opt in tries:
                try:
                    return pd.read_csv(path, engine="python", **opt)
                except Exception:
                    continue
        raise RuntimeError("⚠️ Fichier `ubisoft_critiques.csv` introuvable.")

    try:
        dfc_raw = _read_critiques()
    except Exception as e:
        st.error(f"⚠️ {e}")
        st.stop()

    # --- Détection colonnes Jeu / Critique
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", str(s))
        s = "".join(c for c in s if not unicodedata.combining(c))
        return re.sub(r"[\s\-_\/]+", " ", s).strip().lower()

    cols_map = {_norm(c): c for c in dfc_raw.columns}
    col_game = next((cols_map[k] for k in ["jeu","name","title","game","nom"] if k in cols_map), None)
    col_text = next((cols_map[k] for k in ["critique","review","user review","user_review","comment","texte","text"] if k in cols_map), None)
    if not (col_game and col_text):
        st.error("⚠️ Le CSV des critiques doit contenir une colonne **Jeu/Name** et **Critique/Review**.")
        st.stop()

    dfc = dfc_raw[[col_game, col_text]].rename(columns={col_game:"Jeu", col_text:"Critique"}).dropna()

    # --- Filtrer uniquement Skull & Bones
    jeu_cible = "Skull and Bones"
    df_skull = dfc[dfc["Jeu"].astype(str).str.lower().str.contains("skull")]

    if df_skull.empty:
        st.warning("⚠️ Aucune critique trouvée pour 'Skull & Bones'. Vérifie ton CSV.")
    else:
        # --- Garder uniquement les critiques NEGATIVES
        critiques_neg = df_skull["Critique"].dropna().astype(str)
        critiques_neg = critiques_neg[critiques_neg.apply(lambda x: TextBlob(x).sentiment.polarity < 0)]

        # --- Nettoyage texte
        texte = " ".join(critiques_neg.str.lower())
        texte = texte.translate(str.maketrans("", "", string.punctuation))

        # --- Stopwords EXACTS : uniquement ta liste + suppression des noms "skull" et "bones"
        custom_stopwords = set(STOPWORDS)
        custom_stopwords.update([
            'game', 'games', 'ubisoft', 'play', 'player', 'playing',
            'dlc', 'edition', 'content', 'series', 'experience',
            'version', 'new', 'like', 'one', 'ship', 'pirate','fun','combat',
            'skull', 'bones'  # ← Ajoutés ici pour ne PAS les afficher
        ])

        # --- Liste des mots clés à mettre en ROUGE vif
        highlight_words = {
            "boring": "#d62728",
            "repetitive": "#d62728",
            "money": "#d62728",
            "combat": "#d62728",
            "waste": "#d62728",
            "gameplay": "#d62728",
            "disappointing": "#d62728",
            "black": "#d62728",
            "flag": "#d62728"
        }

        # --- Fonction de coloration dynamique des mots
        def color_function(word, **kwargs):
            return highlight_words.get(word.lower(), "lightcoral")

        # --- Génération du WordCloud (style projet)
        wordcloud = WordCloud(
            width=1200,
            height=700,
            background_color="white",
            stopwords=custom_stopwords,
            color_func=color_function
        ).generate(texte)

        # --- Affichage graphique
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig)
    # ────────────────────────────────────────────────
    # 📊 Section 3.4 : Un désalignement total entre budget, durée et résultat
    # ────────────────────────────────────────────────

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # --- Titre de la section
    st.markdown("## 3.4. Un désalignement total entre budget, durée et résultat")

    # ——— Texte introductif (au-dessus du graphique budget)
    st.markdown("""
Ce qui rend **Skull & Bones** encore plus problématique, c’est la **disproportion**
entre les **moyens engagés** et la **qualité perçue**.  
Avec un **budget estimé à plus de 200 millions de dollars** *(voire **500 M$** selon certaines sources,
notamment d’anciens employés d’Ubisoft)*, le jeu se classe parmi les **plus ambitieux de l’industrie**,  
aux côtés de productions ayant connu un **succès énorme** comme **GTA V** ou **Call of Duty: Modern Warfare**.
""")

    # --- Données fictives de l'étude AAA (à adapter selon tes fichiers CSV si nécessaire)
    data_budget = {
        "Jeu": [
            "Assassin's Creed II", "Far Cry 3", "The Last of Us Part II",
            "FIFA 23", "Elden Ring", "The Legend of Zelda: BOTW",
            "Skull and Bones", "Call of Duty: Modern Warfare (2019)",
            "GTA V", "Red Dead Redemption 2"
        ],
        "Budget": [80, 90, 110, 120, 130, 150, 200, 220, 265, 350]
    }

    df_budget = pd.DataFrame(data_budget)

    # --- Création du graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=df_budget,
        y="Jeu",
        x="Budget",
        palette=["#6baed6" if jeu != "Skull and Bones" else "#fdae6b" for jeu in df_budget["Jeu"]],
        ax=ax
    )

    # --- Ligne rouge verticale sur le budget de Skull & Bones
    ax.axvline(200, color="red", linestyle="--", linewidth=2, label="Budget Skull & Bones")

    # --- Personnalisation graphique
    ax.set_title("Budgets de production des jeux AAA", fontsize=14, fontweight="bold")
    ax.set_xlabel("Budget (en millions $)")
    ax.set_ylabel("Jeu")
    ax.legend()

    # --- Affichage
    st.pyplot(fig)
    # ——— Texte d’interprétation (après le graphique budget)
    st.markdown("""
En comparant la **durée de développement**, le **budget** et la **note Metacritic** de ces jeux,
on observe que **Skull & Bones** se positionne à l’**extrême** : **coûteux**, **le plus long à produire**,
avec **le score critique le plus bas**.
""")
    # ────────────────────────────────────────────────
    # Durée de Développement vs Note Metacritic (bulles = budget) — version Seaborn/Matplotlib pour Streamlit
    # ────────────────────────────────────────────────
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import streamlit as st

    st.subheader(" Durée de Développement vs Note Metacritic — 💰 Taille des bulles = Budget de développement")

    @st.cache_data
    def load_aaa():
        # Charge ton fichier tel quel (mêmes noms de colonnes que dans ton code)
        tries = [
            dict(sep=",", encoding="utf-8"),
            dict(sep=";", encoding="utf-8"),
            dict(sep=",", encoding="utf-8-sig"),
            dict(sep=";", encoding="utf-8-sig"),
            dict(sep=",", encoding="cp1252"),
            dict(sep=";", encoding="cp1252"),
            dict(sep="\t", encoding="utf-8"),
        ]
        last_err = None
        for path in ["jeux_AAA_metacritic_only.csv", "data/jeux_AAA_metacritic_only.csv"]:
            for opt in tries:
                try:
                    df = pd.read_csv(path, engine="python", **opt)
                    return df
                except Exception as e:
                    last_err = e
        raise RuntimeError(f"Impossible de lire le CSV : {last_err}")

    # ➜ On suppose que ton CSV a bien les colonnes : title, publisher, budget_musd, development_years, metacritic_score
    df_full = load_aaa()
    required = {"title", "publisher", "budget_musd", "development_years", "metacritic_score"}
    if not required.issubset(df_full.columns):
        st.error(f"Colonnes attendues manquantes. Il faut au minimum : {sorted(required)}")
        st.stop()

    # — Graphique identique à ton code
    plt.figure(figsize=(14, 8))
    sns.set(style="whitegrid", font_scale=1.1)

    # Palette dynamique basée sur le nombre d'éditeurs
    n_publishers = df_full["publisher"].nunique()
    palette = sns.color_palette("tab10", n_colors=n_publishers)

    plot = sns.scatterplot(
        data=df_full,
        x="development_years",
        y="metacritic_score",
        hue="publisher",
        size="budget_musd",
        sizes=(200, 2000),
        alpha=0.9,
        edgecolor="black",
        linewidth=0.5,
        palette=palette,
        legend="brief"
    )

    # Titres des jeux au centre des bulles (comme ton code)
    for _, row in df_full.iterrows():
        plt.text(
            row["development_years"],
            row["metacritic_score"],
            row["title"],
            fontsize=10,
            ha="center",
            va="center",
            color="black",
            weight="bold",
        )

    # Mise en page (identique à l’esprit de ton snippet)
    plt.title("🎮 Durée de Développement vs Note Metacritic\n💰 Taille des bulles = Budget de développement",
              fontsize=16, weight="bold")
    plt.xlabel("Durée de développement (années)", fontsize=12)
    plt.ylabel("Note Metacritic", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)

    # Légendes sur la droite
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left",
               title="Éditeur / Budget", borderaxespad=1)

    # Petites marges pour éviter de couper les bulles/labels
    plt.margins(x=0.08, y=0.06)

    # Optionnel : forcer un peu de marge en X pour que tout soit bien “entier”
    xmin, xmax = df_full["development_years"].min(), df_full["development_years"].max()
    plt.xlim(xmin - 0.5, xmax + 1.0)

    plt.tight_layout()

    # ➜ Affichage Streamlit
    st.pyplot(plt.gcf(), clear_figure=True)
    st.markdown("""
Ce graphique met en lumière un décalage profond entre effort, coût et valeur livrée. Il constitue un cas d’école d’échec produit, qui interroge autant la stratégie d’Ubisoft que sa capacité à piloter efficacement des projets à long terme.
""")

    st.divider()

# ────────────────────────────────────────────────
# PAGE 5 : CONCLUSION — texte identique au screenshot
# ────────────────────────────────────────────────
elif page == "Conclusion":
    st.title("Conclusion")

    st.markdown("""
L’ensemble des analyses menées dans ce projet met en lumière un constat clair : la chute d’Ubisoft ne s’explique pas par une crise généralisée du secteur du jeu vidéo, mais bien par des faiblesses propres à l’entreprise. Si l’éditeur continue de produire un volume élevé de titres chaque année, cette stratégie ne se traduit plus par une rentabilité suffisante. Depuis près d’une décennie, les signaux d’alerte sont visibles, notamment à travers une rupture croissante entre la promesse marketing des jeux et leur réception par les joueurs.


                
Plus préoccupant encore, les projets les plus ambitieux — tant en durée de développement qu’en investissement financier — figurent aujourd’hui parmi les plus sévèrement critiqués. Cette anomalie entre les moyens déployés et la qualité perçue traduit une perte de cohérence entre la stratégie de production et les attentes du marché.

Dans ce contexte, Ubisoft doit impérativement repenser deux dimensions centrales de son organisation : sa capacité à piloter des projets sur le long terme de manière agile et réaliste, et sa faculté à intégrer de manière proactive les retours et attentes de ses communautés de joueurs.

Pour inverser cette tendance, plusieurs pistes peuvent être envisagées. Il apparaît essentiel d’améliorer l’efficience opérationnelle en ré-alignant la masse salariale avec les ambitions réelles et la performance attendue, tout en identifiant les studios ou projets structurellement sous-performants. Il est également crucial de mieux valoriser l’engagement des joueurs en exploitant les signaux faibles issus des critiques, des forums ou des données d’usage afin d’orienter les décisions produit de manière plus pertinente.

Par ailleurs, Ubisoft gagnerait à repenser ses modèles économiques, en redonnant une place au freemium lorsque cela est pertinent, ou en explorant des formats hybrides comme les abonnements ou les contenus additionnels. Enfin, la dépendance à des blockbusters à très long développement devrait être réévaluée, au profit de cycles plus courts, plus agiles, et potentiellement plus en phase avec les évolutions du marché                
                """)
    st.divider()
    st.divider()
    

# Annexe méthodologique — petite police + italique
    st.markdown(
    """
    <style>
      .annexe-ux { font-size: 0.92rem; font-style: italic; line-height: 1.55; }
      .annexe-ux h3 { font-size: 1.08rem; font-style: italic; margin: 0 0 0.6rem 0; }
      .annexe-ux p { margin: 0 0 0.6rem 0; }
      .annexe-ux strong { font-style: normal; } /* garder les sous-titres en gras lisibles */
    </style>
    <div class="annexe-ux">
      <h3>Annexe méthodologique</h3>

      <p>Pour mieux comprendre le déclin d’Ubisoft, nous avons entrepris une approche en trois volets, mêlant analyse financière, exploration des performances des jeux, et étude de leur réception critique.</p>

      <p><strong>I. Une plongée dans les chiffres : l’analyse financière comparative</strong><br/>
      Notre première étape fut de comparer l’évolution boursière d’Ubisoft à celle de deux ETF emblématiques du secteur : ESPO et HERO. Grâce à la plateforme TradingView, nous avons recueilli les données couvrant la période 2018 à 2024.<br/>
      Pour les analyser, nous avons mobilisé les capacités de LLM couplées à une recherche documentaire rigoureuse.<br/>
      Résultat : un jeu de données financier complet, traçant les courbes de valeur de l’action Ubisoft et de nos deux ETF de référence.<br/>
      <em>Source : Site TradingView</em><br/>
      <a href="https://fr.tradingview.com/chart/OHym2NDq/?symbol=NASDAQ%3AESPO">ESPO</a> ;
      <a href="https://fr.tradingview.com/chart/OHym2NDq/?symbol=NASDAQ%3AHERO">HERO</a> ;
      <a href="https://fr.tradingview.com/chart/OHym2NDq/?symbol=EUROTLX%3A4UBI">4UBI</a>
      </p>

      <p><strong>II. Explorer les performances concrètes : les jeux Ubisoft à la loupe</strong><br/>
      Nous nous sommes ensuite intéressés aux performances des jeux publiés par Ubisoft entre 1995 et 2025.<br/>
      Pour ce faire, nous avons utilisé le site VG Insights, une base de données spécialisée, afin d’extraire les informations-clés : éditeurs, volumes de jeux, etc.<br/>
      Cette extraction a été automatisée grâce à des scripts basés sur BeautifulSoup et Selenium, ce qui nous a permis de constituer deux jeux de données – l’un dédié aux éditeurs , l’autre aux jeux eux-mêmes.<br/>
      <em>Source : Site VG insights :</em><br/>
      <a href="https://vginsights.com/publishers-database">https://vginsights.com/publishers-database</a> &amp;
      <a href="https://vginsights.com/publisher/8/ubisoft">https://vginsights.com/publisher/8/ubisoft</a>
      </p>

      <p><strong>III. Ce que le public en pense : critiques et perception</strong><br/>
      Enfin, pour comprendre la réception des jeux Ubisoft par la critique et les joueurs, nous avons scruté Metacritic, plateforme de référence dans l’agrégation de critiques.<br/>
      Nous avons collecté les notes et critiques des jeux Ubisoft sortis entre 1995 et 2025, encore une fois à l’aide de scraping automatisé via BeautifulSoup et Selenium.<br/>
      Cette phase a abouti à la création de deux jeux de données complémentaires, l’un centré sur les notes, l’autre sur les commentaires qualitatifs.<br/>
      <em>Source : Site Metacritic :</em><br/>
      <a href="https://www.metacritic.com/browse/game/?releaseYearMin=1995&releaseYearMax=2025&page=1">https://www.metacritic.com/browse/game/?releaseYearMin=1995&amp;releaseYearMax=2025&amp;page=1</a>
      </p>
    </div>
    """,
    unsafe_allow_html=True
)








