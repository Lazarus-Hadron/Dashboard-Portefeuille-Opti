# app.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from opti_poids import (
    portefeuille_opti,
    calculer_allocation,
    simuler_performance_historique,
    backtest_portefeuille,
)
from pypfopt import EfficientFrontier, expected_returns, risk_models


# ==========================================================
# === CONFIG GÉNÉRALE ET THÈME ===
# ==========================================================
st.set_page_config(page_title="Optimisation de Portefeuille", layout="wide")


st.title("😼 Optimisation de Portefeuille - Modèle de Markowitz 😼")

# ==========================================================
# === TICKERS FIXES ===
# ==========================================================
st.sidebar.subheader("📈 Actifs inclus (fixes)")
default_tickers = [
    "0P0001BL65.F", "0P0001L260.F", "0P0000SYIL.F", "0P0001IL66.F", "0P0001RA8B.F", "0P00009VYF.F",
    "0P00013PKD.F", "0P0001BM4D.F", "0P0001NCVJ.F", "0P0001IFVT.F", "0P0001BM5K.F", "0P0001LIPM.F",
    "0P0001BL4U.F", "0P0001BP2U.F", "0P00011RBT.F", "0P0001NCUR.F", "0P0000NDZK.F", "0P0000HLKB.F",
    "0P0001EXJ9.F", "0P0001BOR1.F", "0P00013PIS.F", "0P0000U52L.F", "0P0000HLRW.F", "0P0001NCEU.F",
    "0P0000HLK5.F", "0P00008W3L.F", "0P0001BM48.F", "0P0001IKPM.F", "0P0000HLKD.F", "0P0001LGV9.F",
    "0P0001NCFU.F", "0P0001NCHI.F", "0P000177JS.F", "0P0001PGO0.F", "0P0000HLKA.F", "0P0001PHB5.F",
    "0P0001NYI4.F", "0P00018OAL.F", "0P0000HLK8.F", "0P0001IJVO.F"
]
st.sidebar.dataframe(pd.DataFrame(default_tickers, columns=["Tickers"]))

# ==========================================================
# === PARAMÈTRES ===
# ==========================================================
st.sidebar.subheader("⚙️ Paramètres")
period = st.sidebar.selectbox("Période d'analyse", ["1y", "3y", "5y", "10y"], index=2)
montant_total = st.sidebar.number_input("Montant à investir (€)", min_value=1000, value=10000, step=100)
allow_short = st.sidebar.checkbox("Autoriser la vente à découvert ?", value=False)
objectif = st.sidebar.selectbox("Objectif d'optimisation", ["max_sharpe", "min_volatility"], index=0)
mois_estimation = st.sidebar.slider("Durée d'investissement (mois)", 6, 120, 12, step=6)

# ==========================================================
# === OPTIMISATION AUTOMATIQUE ===
# ==========================================================
poids, data, metrics, mu, S = portefeuille_opti(default_tickers, period=period, allow_short=allow_short, objective=objectif)

if poids and not data.empty:
    allocation = calculer_allocation(poids, montant_total)
    simulation = simuler_performance_historique(data, poids, montant_total, mois_projection=mois_estimation)
    cumulative, stats = backtest_portefeuille(data, poids, montant_total)

    valeur_finale = simulation["valeur_projetee"]
    plus_value = simulation["plus_value"]

    # ==========================================================
    # === DASHBOARD INDICATEURS ===
    # ==========================================================
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📈 Rendement annuel attendu", f"{metrics['Rendement annuel attendu']:.2%}",
                help="Performance moyenne annualisée calculée à partir des rendements historiques ajustés.")
    col2.metric("📉 Volatilité annuelle", f"{metrics['Volatilité annuelle']:.2%}",
                help="Écart-type des rendements annualisé (indique le risque du portefeuille).")
    col3.metric("💵 Investissement initial", f"{montant_total:,.0f} €", help="Montant total investi au départ.")
    col4.metric("💰 Plus-value estimée", f"{plus_value:,.0f} €", f"{simulation['rendement_total']:.2f}%",
                help="Gain projeté basé sur la moyenne historique du portefeuille.")
    col5.metric("🏦 Valeur totale estimée", f"{valeur_finale:,.0f} €", help="Montant total estimé après la plus-value.")
    st.divider()

    # ==========================================================
    # === RÉPARTITION OPTIMALE ===
    # ==========================================================
    st.subheader("🥧 Répartition optimale du portefeuille")

    col_pie, col_table = st.columns([0.55, 0.45])

    df_poids = pd.DataFrame.from_dict(poids, orient='index', columns=['Poids']).sort_values(by='Poids', ascending=False)
    df_alloc = pd.DataFrame(allocation).T
    df_alloc.columns = ["Poids (%)", "Montant (€)"]

    # --- Palette de base (Plotly officielle) ---
    base_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
                '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    with col_pie:
        df_top = df_poids[df_poids['Poids'] > 0.001]

        # --- Création d’un dictionnaire couleur par actif ---
        color_map = {ticker: base_colors[i % len(base_colors)] for i, ticker in enumerate(df_top.index)}

        fig_pie = go.Figure(data=[go.Pie(
            labels=df_top.index,
            values=df_top["Poids"],
            hole=0.4,
            textinfo='label+percent',
            marker=dict(colors=[color_map[t] for t in df_top.index])
        )])
        fig_pie.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_pie, use_container_width=True)


    with col_table:
        st.write("### 💸 Détail des montants à investir")
        df_display = df_alloc.copy()
        df_display["Poids (%)"] = df_display["Poids (%)"].map("{:.2f}%".format)
        df_display["Montant (€)"] = df_display["Montant (€)"].map("{:,.0f} €".format)
        st.dataframe(df_display, use_container_width=True)

    st.divider()

    # ==========================================================
    # === ONGLET DES GRAPHIQUES ===
    # ==========================================================
    tab1, tab2, tab3, tab4 = st.tabs(["💹 Évolution des prix des actifs","📉 Frontière efficiente", "🎲 Simulation Monte Carlo", "📊 Backtest historique"])

    # === Évolution des prix des actifs ===
    with tab1:
        try:
            st.write(f"### 💹 Évolution des prix sur la période : {period}")

            if not data.empty and poids:
                # --- On ne garde que les actifs qui ont un poids positif ---
                actifs_selectionnes = [t for t, w in poids.items() if w > 0]
                data_filtrée = data[actifs_selectionnes].copy()

                # --- Rechargement des données selon la période choisie ---
                from opti_poids import get_data
                data_filtrée = get_data(actifs_selectionnes, period=period)

                if not data_filtrée.empty:
                    fig_prices = go.Figure()

                    for i, ticker in enumerate(data_filtrée.columns):
                        couleur = color_map.get(ticker, base_colors[i % len(base_colors)])
                        fig_prices.add_trace(
                            go.Scatter(
                                x=data_filtrée.index,
                                y=data_filtrée[ticker],
                                mode="lines",
                                name=ticker,
                                line=dict(width=2, color=couleur)
                            )
                        )

                    fig_prices.update_layout(
                        title=f"Évolution des prix des {len(actifs_selectionnes)} actifs sélectionnés",
                        xaxis_title="Date",
                        yaxis_title="Prix de clôture",
                        template="plotly_dark",
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.3,
                            xanchor="center",
                            x=0.5
                        ),
                        height=600
                    )
                    st.plotly_chart(fig_prices, use_container_width=True)
                else:
                    st.warning("⚠️ Impossible de charger les données pour cette période.")
            else:
                st.warning("⚠️ Aucun actif sélectionné dans le portefeuille optimal.")
        except Exception as e:
            st.error(f"Erreur affichage évolution des prix des actifs : {e}")

    # === FRONTIÈRE EFFICIENTE ===
    with tab2:
        try:
            bounds = (-1, 1) if allow_short else (0, 1)
            ef_tmp = EfficientFrontier(mu, S, weight_bounds=bounds)
            ef_tmp.min_volatility()
            min_port_return = ef_tmp.portfolio_performance(verbose=False)[0]
            max_port_return = float(mu.max())

            ret_range = np.linspace(min_port_return, max_port_return, 30)
            risks, returns = [], []
            for r in ret_range:
                try:
                    ef_local = EfficientFrontier(mu, S, weight_bounds=bounds)
                    ef_local.efficient_return(r)
                    perf = ef_local.portfolio_performance(verbose=False)
                    returns.append(perf[0])
                    risks.append(perf[1])
                except Exception:
                    continue

            fig_front = go.Figure()
            fig_front.add_trace(go.Scatter(
                x=risks, y=returns, mode='lines',
                line=dict(color='#52b788', width=3), name='Frontière efficiente'
            ))
            fig_front.add_trace(go.Scatter(
                x=[metrics["Volatilité annuelle"]],
                y=[metrics["Rendement annuel attendu"]],
                mode='markers+text',
                name='Portefeuille optimal',
                text=["Portefeuille optimal"],
                textposition="top center",
                marker=dict(size=10, color='#ffd700')
            ))
            fig_front.update_layout(
                xaxis_title='Volatilité annuelle',
                yaxis_title='Rendement annuel attendu',
                template='plotly_dark',
            )
            st.plotly_chart(fig_front, use_container_width=True)
        except Exception as e:
            st.error(f"Erreur affichage frontière efficiente : {e}")

    # === MONTE CARLO ===
    with tab3:
        n_sim = st.slider("Nombre de simulations", 500, 5000, 1000, step=500)
        horizon = st.slider("Horizon (mois)", 6, 60, 12, step=6)
        mu_month = metrics["Rendement mensuel attendu"]
        sigma_month = metrics["Volatilité annuelle"] / np.sqrt(12)

        sim_paths = np.zeros((horizon, n_sim))
        for i in range(n_sim):
            rand_returns = np.random.normal(mu_month, sigma_month, horizon)
            sim_paths[:, i] = (1 + rand_returns).cumprod()
        mean_path = sim_paths.mean(axis=1)

        fig_mc = go.Figure()
        for i in range(min(80, n_sim)):
            fig_mc.add_trace(go.Scatter(
                y=sim_paths[:, i],
                mode='lines',
                line=dict(width=1, color='#2d6a4f'),
                opacity=0.2,
                showlegend=False
            ))
        fig_mc.add_trace(go.Scatter(
            y=mean_path,
            mode='lines',
            line=dict(width=3, color='#52b788'),
            name='Moyenne'
        ))
        fig_mc.update_layout(
            xaxis_title="Mois",
            yaxis_title="Croissance (base 1)",
            template='plotly_dark',
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    # === BACKTEST ===
    with tab4:
        if not cumulative.empty:
            st.line_chart(cumulative, use_container_width=True)
            st.write(f"**Rendement total :** {stats['Rendement total']:.2%}")
            st.write(f"**Volatilité annualisée :** {stats['Volatilité annualisée']:.2%}")
            st.write(f"**Ratio de Sharpe empirique :** {stats['Ratio de Sharpe empirique']:.2f}")
            st.write(f"**Max Drawdown :** {stats['Max Drawdown']:.2%}")
            st.write(f"**Valeur finale :** {stats['Valeur finale']:.0f} €")

else:
    st.warning("⚠️ Impossible de calculer le portefeuille : vérifiez la connexion ou les données.")
