# opti_poids.py
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.exceptions import OptimizationError
import streamlit as st


# =====================================================
# === 1️⃣ Téléchargement et nettoyage des données ===
# =====================================================
def get_data(tickers, period="5y"):
    """
    Télécharge les données de clôture via yfinance et nettoie le DataFrame.
    Gère le cas d'un seul ticker (Series -> DataFrame) et supprime les valeurs manquantes.
    """
    try:
        if not tickers:
            st.error("❌ Aucun ticker spécifié.")
            return pd.DataFrame()

        data = yf.download(tickers, period=period, progress=False)["Close"]

        # Si un seul ticker -> conversion en DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=tickers if isinstance(tickers, str) else tickers[0])

        if data.empty:
            st.error("❌ Aucune donnée récupérée pour les tickers spécifiés.")
            return pd.DataFrame()

        # Nettoyage : suppression colonnes ou lignes vides
        data = data.dropna(axis=1, how="all").dropna(how="any")

        if data.empty:
            st.error("❌ Données invalides ou trop incomplètes après nettoyage.")
        return data

    except Exception as e:
        st.error(f"❌ Erreur lors du téléchargement des données : {e}")
        return pd.DataFrame()


# =====================================================
# === 2️⃣ Optimisation du portefeuille (Markowitz) ===
# =====================================================
def portefeuille_opti(tickers, period="5y", allow_short=False, objective="max_sharpe"):
    """
    Calcule le portefeuille optimal selon Markowitz.
    Retourne :
        - poids optimaux (dict)
        - données brutes (DataFrame)
        - indicateurs du portefeuille (dict)
        - rendements espérés individuels (Series)
        - matrice de covariance (DataFrame)
    """
    if not tickers:
        return {}, pd.DataFrame(), {}, None, None

    data = get_data(tickers, period)
    if data.empty:
        return {}, pd.DataFrame(), {}, None, None

    if len(data.columns) < 2:
        st.error("❌ Au moins 2 actifs avec des données sont nécessaires pour l'optimisation.")
        return {}, data, {}, None, None

    # Estimation des rendements espérés et de la covariance annualisés
    try:
        mu = expected_returns.mean_historical_return(data, frequency=252)
        S = risk_models.sample_cov(data, frequency=252)
    except Exception as e:
        st.error(f"❌ Erreur lors du calcul des rendements/covariances : {e}")
        return {}, data, {}, None, None

    # Création du modèle d’optimisation
    bounds = (-1, 1) if allow_short else (0, 1)
    ef = EfficientFrontier(mu, S, weight_bounds=bounds)

    try:
        if objective == "max_sharpe":
            ef.max_sharpe()
        elif objective == "min_volatility":
            ef.min_volatility()
        else:
            ef.max_sharpe()
    except OptimizationError as e:
        st.error(f"❌ Erreur d'optimisation : {e}")
        return {}, data, {}, mu, S
    except Exception as e:
        st.error(f"❌ Erreur inattendue d'optimisation : {e}")
        return {}, data, {}, mu, S

    poids = ef.clean_weights()

    # Calcul des performances du portefeuille
    try:
        rendement_annuel, volatilite_annuelle, ratio_sharpe = ef.portfolio_performance(verbose=False)
    except Exception:
        rendement_annuel, volatilite_annuelle, ratio_sharpe = None, None, None

    # --- Méthode A : conversion du rendement annualisé du portefeuille en rendement mensuel
    rendement_mensuel = (1 + rendement_annuel) ** (1 / 12) - 1 if rendement_annuel is not None else None

    metrics = {
        "Rendement annuel attendu": rendement_annuel,
        "Rendement mensuel attendu": rendement_mensuel,
        "Volatilité annuelle": volatilite_annuelle,
        "Ratio de Sharpe": ratio_sharpe,
    }

    return poids, data, metrics, mu, S


# =====================================================
# === 3️⃣ Allocation en euros selon les poids ===
# =====================================================
def calculer_allocation(poids, montant_total, seuil_pourcentage=0.1):
    """
    Calcule l'allocation monétaire selon les poids optimaux.
    Retourne un dict {ticker: {poids%, montant€}} filtré selon un seuil minimal de pourcentage.
    """
    if not poids:
        return {}

    allocation = {}
    for ticker, w in poids.items():
        if w and w * 100 >= seuil_pourcentage:
            allocation[ticker] = {
                "poids_pourcentage": w * 100,
                "montant_euros": w * montant_total,
            }
    return allocation


# =====================================================
# === 4️⃣ Simulation de performance future simple ===
# =====================================================
def simuler_performance_historique(data, poids, montant_total, mois_projection=12):
    """
    Simule la performance future du portefeuille à partir de rendements historiques moyens.
    Basé sur la moyenne journalière historique du portefeuille (pondération dot product).
    """
    if data.empty or not poids:
        return {}

    try:
        rendements_journaliers = data.pct_change().dropna()
        poids_series = pd.Series(poids).reindex(rendements_journaliers.columns).fillna(0).astype(float)

        # Rendement journalier du portefeuille (produit scalaire)
        port_daily = rendements_journaliers.dot(poids_series)
        if port_daily.empty:
            return {}

        mean_daily = port_daily.mean()
        std_daily = port_daily.std()

        # Conversion en rendement mensuel moyen (21 jours de trading/mois)
        rendement_mensuel_moyen = (1 + mean_daily) ** 21 - 1

        # Projection géométrique sur n mois
        valeur_projetee = montant_total * ((1 + rendement_mensuel_moyen) ** mois_projection)
        plus_value = valeur_projetee - montant_total

        return {
            "investissement_initial": montant_total,
            "valeur_projetee": valeur_projetee,
            "plus_value": plus_value,
            "rendement_mensuel_moyen": rendement_mensuel_moyen,
            "rendement_total": (valeur_projetee / montant_total - 1) * 100,
            "mois_simulation": mois_projection,
            "mean_daily": mean_daily,
            "std_daily": std_daily,
        }

    except Exception as e:
        st.error(f"❌ Erreur dans la simulation : {e}")
        return {}


# =====================================================
# === 5️⃣ Backtesting du portefeuille optimisé ===
# =====================================================
def backtest_portefeuille(data, poids, montant_initial=10000):
    """
    Effectue un backtest historique du portefeuille.
    Retourne :
      - DataFrame de la valeur cumulée
      - statistiques : rendement cumulé, volatilité, drawdown, ratio de Sharpe empirique
    """
    if data.empty or not poids:
        st.warning("⚠️ Pas de données disponibles pour le backtest.")
        return pd.DataFrame(), {}

    try:
        rendements = data.pct_change().dropna()
        poids_series = pd.Series(poids).reindex(rendements.columns).fillna(0).astype(float)

        # Rendement quotidien du portefeuille
        port_daily = rendements.dot(poids_series)
        cumulative = (1 + port_daily).cumprod() * montant_initial

        # Statistiques du backtest
        rendement_total = cumulative.iloc[-1] / montant_initial - 1
        volatilite = port_daily.std() * np.sqrt(252)
        sharpe = (port_daily.mean() / port_daily.std()) * np.sqrt(252) if port_daily.std() > 0 else np.nan
        max_drawdown = ((cumulative / cumulative.cummax()) - 1).min()

        stats = {
            "Rendement total": rendement_total,
            "Volatilité annualisée": volatilite,
            "Ratio de Sharpe empirique": sharpe,
            "Max Drawdown": max_drawdown,
            "Valeur finale": cumulative.iloc[-1],
        }

        return cumulative, stats

    except Exception as e:
        st.error(f"❌ Erreur lors du backtest : {e}")
        return pd.DataFrame(), {}
