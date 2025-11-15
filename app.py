# streamlit_portfolio_pro_fr.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import plotly.graph_objects as go

st.set_page_config(page_title="Portefeuille Optimisé", layout="wide")
st.title("💼 Portefeuille Optimisé")

# ---------------------------
# Sidebar - paramètres globaux
# ---------------------------
st.sidebar.header("Paramètres globaux")

default_tickers = [
    "0P0001BL65.F","0P0001L260.F","0P0000SYIL.F","0P0001IL66.F",
    "0P0001RA8B.F","0P00009VYF.F","0P00013PKD.F","0P0001BM4D.F",
    "0P0001NCVJ.F","0P0001IFVT.F","0P0001BM5K.F","0P0001LIPM.F",
    "0P0001BL4U.F","0P0001BP2U.F","0P00011RBT.F","0P0001NCUR.F",
    "0P0000NDZK.F","0P0000HLKB.F","0P0001EXJ9.F","0P0001BOR1.F",
    "0P00013PIS.F","0P0000U52L.F","0P0000HLRW.F","0P0001NCEU.F",
    "0P0000HLK5.F","0P00008W3L.F","0P0001BM48.F","0P0001IKPM.F",
    "0P0000HLKD.F","0P0001LGV9.F","0P0001NCFU.F","0P0001NCHI.F",
    "0P000177JS.F","0P0001PGO0.F","0P0000HLKA.F","0P0001PHB5.F",
    "0P0001NYI4.F","0P00018OAL.F","0P0000HLK8.F","0P0001IJVO.F"
]

tickers = st.sidebar.multiselect(
    "Titres (groupe 'Chat')",
    options=default_tickers,
    default=default_tickers,
    help="Sélectionnez les fonds ou actions à inclure dans le portefeuille."
)
start_date = st.sidebar.date_input(
    "Date de début",
    pd.to_datetime("2015-01-01"),
    help="Date de début pour les historiques de prix."
)
end_date = st.sidebar.date_input(
    "Date de fin",
    pd.to_datetime("today"),
    help="Date de fin pour les historiques de prix."
)
initial_investment = st.sidebar.number_input(
    "Capital initial (€)",
    min_value=1000,
    value=10000,
    step=1000,
    help="Montant total que vous souhaitez investir."
)
period_months = st.sidebar.slider(
    "Horizon d'investissement (mois)",
    1, 240, 36,
    help="Durée prévue de l'investissement pour calculer la valeur future."
)
risk_free_rate = st.sidebar.number_input(
    "Taux sans risque (%)",
    value=1.5,
    help="Taux sans risque annuel utilisé pour le calcul du ratio de Sharpe."
)/100

# ---------------------------
# Sidebar - Méthode d'optimisation
# ---------------------------
optimization_method = st.sidebar.selectbox(
    "Méthode d'optimisation",
    ["Max Sharpe", "Min Vol", "Rendement cible"],
    help="Choisissez la méthode d'optimisation du portefeuille."
)

from pypfopt import objective_functions

# --- Sidebar: Diversification L2 ---
l2_reg = st.sidebar.slider(
    "Diversification (L2)",
    min_value=0.0, max_value=0.5, value=0.01, step=0.01,
    help="Pénalisation L2 pour favoriser la diversification du portefeuille. 0 = pas de régularisation."
)


# ---------------------------
# Internal defaults for data quality (no sliders)
# ---------------------------
max_missing = 0.60  # allow up to 60% missing (weekly VL for funds)
low_vol_threshold = 0.05  # 5% annual vol threshold for warnings

# ---------------------------
# Robust data loader (improved for OPCVM)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data_clean(tickers_list, start, end, max_missing_allowed=0.60):
    if not tickers_list:
        return pd.DataFrame(), pd.Series(dtype=float), []

    raw_daily = yf.download(
        tickers_list,
        start=start,
        end=end,
        auto_adjust=True,
        interval="1d",
        progress=False
    )

    df_daily = raw_daily["Close"] if "Close" in raw_daily.columns else raw_daily
    missing_daily = df_daily.isna().mean()

    opcvm_threshold = 0.70
    need_weekly = missing_daily > opcvm_threshold

    if need_weekly.any():
        tickers_weekly = need_weekly[need_weekly].index.tolist()
        tickers_daily_ok = need_weekly[~need_weekly].index.tolist()

        raw_weekly = yf.download(
            tickers_weekly,
            start=start,
            end=end,
            auto_adjust=True,
            interval="1wk",
            progress=False
        )
        df_weekly = raw_weekly["Close"] if "Close" in raw_weekly.columns else raw_weekly

        df_combined = pd.concat([df_daily[tickers_daily_ok], df_weekly], axis=1)
    else:
        df_combined = df_daily

    df_filled = df_combined.ffill()
    missing_after = df_filled.isna().mean()
    good = missing_after[missing_after <= max_missing_allowed].index.tolist()
    bad = missing_after[missing_after > max_missing_allowed].index.tolist()

    df_final = df_filled[good].copy()
    return df_final, missing_after, bad

# ---------------------------
# Load data
# ---------------------------
data, missing_pct_all, removed_tickers = load_data_clean(tickers, start_date, end_date, max_missing_allowed=max_missing)

if data.empty:
    st.warning("Aucune donnée valide après nettoyage. Vérifiez les tickers/dates ou augmentez le seuil de valeurs manquantes.")
    st.stop()

st.sidebar.markdown(f"Données chargées : **{data.shape[1]}** titres, période {data.index.min().date()} à {data.index.max().date()}")
if removed_tickers:
    st.sidebar.warning(f"{len(removed_tickers)} titres retirés (> {max_missing*100:.0f}% valeurs manquantes) : {removed_tickers}")

# ---------------------------
# Compute returns, mu, covariance
# ---------------------------
@st.cache_data(show_spinner=False)
def compute_mu_S(price_df):
    rets = price_df.pct_change().dropna()
    mu = expected_returns.mean_historical_return(price_df, frequency=252)
    S = risk_models.CovarianceShrinkage(price_df, frequency=252).ledoit_wolf()
    return rets, mu, S

rets, mu, S = compute_mu_S(data)

# ---------------------------
# Compute feasible target return range safely
# ---------------------------
def compute_target_bounds(mu, S):
    ef_min = EfficientFrontier(mu, S)
    ef_min.min_volatility()
    min_ret = ef_min.portfolio_performance(verbose=False)[0]
    ef_max = EfficientFrontier(mu, S)
    try:
        max_ret = ef_max._max_return()
    except Exception:
        max_ret = max(mu)
    return min_ret, max_ret

min_target, max_target = compute_target_bounds(mu, S)

# ---------------------------
# Sidebar - Target Return
# ---------------------------
if optimization_method == "Rendement cible":
    st.sidebar.markdown(f"**Plage de rendement réalisable:** {min_target*100:.2f}% — {max_target*100:.2f}%")
    default_target = float((min_target + max_target) / 2 * 100)
    target_return_input = st.sidebar.number_input(
        "Rendement cible (%)",
        min_value=float(min_target*100),
        max_value=float(max_target*100),
        value=default_target,
        step=0.1,
        help="Rendement annuel cible pour le portefeuille."
    ) / 100.0
else:
    target_return_input = (min_target + max_target) / 2

# ---------------------------
# Optimization function
# ---------------------------
def compute_weights(method, mu, S, rf, target_ret=None):
    ef_local = EfficientFrontier(mu, S)

    if l2_reg > 0:
        ef_local.add_objective(objective_functions.L2_reg, gamma=l2_reg)


    if method == "Max Sharpe":
        ef_local.max_sharpe(risk_free_rate=rf)
    elif method == "Min Vol":
        ef_local.min_volatility()
    elif method == "Rendement cible":
        if target_ret is None:
            raise ValueError("Rendement cible demandé mais target_ret est None")
        target_ret = max(min_target, min(max_target, target_ret))
        ef_local.efficient_return(target_ret)
    else:
        ef_local.max_sharpe(risk_free_rate=rf)
    w = ef_local.clean_weights()
    ann_r, ann_v, shar = ef_local.portfolio_performance(risk_free_rate=rf, verbose=False)
    return w, ann_r, ann_v, shar

weights_dict, ann_ret, ann_vol, sharpe = compute_weights(optimization_method, mu, S, risk_free_rate, target_return_input)
weights_series = pd.Series(weights_dict).reindex(data.columns).fillna(0.0)
optimized_tickers = [k for k,v in weights_dict.items() if v > 0]

# ---------------------------
# Discrete allocation
# ---------------------------
latest_prices = get_latest_prices(data)
try:
    da = DiscreteAllocation(weights_dict, latest_prices, total_portfolio_value=initial_investment)
    allocation, leftover = da.lp_portfolio()
except Exception:
    positive_weights = {k: v for k, v in weights_dict.items() if v > 0}
    latest_prices_pos = latest_prices.reindex(list(positive_weights.keys()))
    da = DiscreteAllocation(positive_weights, latest_prices_pos, total_portfolio_value=initial_investment)
    allocation, leftover = da.lp_portfolio()

# ---------------------------
# Backtest metrics
# ---------------------------
port_ret_daily = (rets * weights_series).sum(axis=1)
cum_perf = (1 + port_ret_daily).cumprod()
monthly_returns = port_ret_daily.resample('M').apply(lambda x: (1 + x).prod() - 1)
monthly_mean = monthly_returns.mean()
years = period_months / 12.0
capital_final = initial_investment * (1 + ann_ret) ** years
plus_value = capital_final - initial_investment

# ---------------------------
# Plausibility checks
# ---------------------------
plausible_return_range = (0.0, 0.40)
plausible_vol_range = (0.01, 0.60)
return_ok = plausible_return_range[0] <= ann_ret <= plausible_return_range[1]
vol_ok = plausible_vol_range[0] <= ann_vol <= plausible_vol_range[1]
asset_vols = rets.std() * np.sqrt(252)
low_vol_assets = asset_vols[asset_vols < low_vol_threshold].sort_values()

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["Portefeuille", "Cours historiques", "Comparer méthodes"]) 

data_end = data.index.max()
data_start_dynamic = data_end - pd.DateOffset(months=period_months) 

# --- TAB 1: Portfolio ---
with tab1:
    st.subheader("Résumé du portefeuille")
    if not return_ok or not vol_ok:
        msg = "⚠️ Vérification de plausibilité : "
        if not return_ok:
            msg += f"Rendement annualisé {ann_ret*100:.2f}% hors plage plausible ({plausible_return_range[0]*100:.0f}%–{plausible_return_range[1]*100:.0f}%). "
        if not vol_ok:
            msg += f"Volatilité annualisée {ann_vol*100:.2f}% hors plage plausible ({plausible_vol_range[0]*100:.0f}%–{plausible_vol_range[1]*100:.0f}%)."
        st.warning(msg)

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r1c1.metric("📊 Rendement annualisé", f"{ann_ret*100:.2f}%", help="Rendement moyen par an du portefeuille.")
    r1c2.metric("📅 Moyenne mensuelle", f"{monthly_mean*100:.2f}%", help="Rendement moyen par mois.")
    r1c3.metric("⚡ Volatilité annualisée", f"{ann_vol*100:.2f}%", help="Écart type annualisé des rendements.")
    r1c4.metric("⭐ Ratio de Sharpe", f"{sharpe:.2f}", help="Rendement excédentaire par unité de risque.")

    c1, c2 = st.columns([1,2])
    c1.metric("💰 Capital initial", f"{initial_investment:,.0f} €")
    c2.metric("💵 Capital final (horizon)", f"{capital_final:,.0f} €", delta=f"{plus_value:,.0f} €")

    st.markdown("### Répartition (interactive)")
    weights_pos = {k: v for k, v in weights_dict.items() if v > 0}
    if weights_pos:
        labels = list(weights_pos.keys())
        vals = [weights_pos[k] for k in labels]
        amounts = [v * capital_final for v in vals]
        pie = go.Figure(data=[go.Pie(
            labels=labels, values=vals, hole=0.4,
            hovertemplate="<b>%{label}</b><br>%{percent:.2%}<br>Montant: %{customdata} €",
            customdata=[f"{a:,.2f}" for a in amounts]
        )])
        pie.update_layout(template="plotly_dark", margin=dict(t=10))
        st.plotly_chart(pie, use_container_width=True)

    st.markdown("### Allocation discrète (actions entières)")
    if allocation:
        df_alloc = pd.DataFrame([
            {"Titre": t, "Actions": s, "Prix": latest_prices.get(t, np.nan), "Valeur": s * latest_prices.get(t, np.nan)}
            for t, s in allocation.items()
        ]).sort_values("Valeur", ascending=False)
        st.dataframe(df_alloc.style.format({"Prix":"{:.2f}", "Valeur":"{:.2f}"}))
    else:
        st.write("Aucune allocation discrète disponible.")
    st.write(f"**Liquidités restantes :** {leftover:.2f} €")

    st.markdown("### Performance cumulative du portefeuille")
    mask = (cum_perf.index >= data_start_dynamic) & (cum_perf.index <= data_end) 
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_perf.index[mask], y=cum_perf.values[mask],
        mode="lines", name="Portefeuille", line=dict(width=3, color="#1f77b4")
    ))
    fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Performance cumulative")
    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: Compare Methods ---
with tab3:
    st.subheader("Comparaison des méthodes d'optimisation")
    methods = ["Max Sharpe", "Min Vol", "Rendement cible"]
    comp_fig = go.Figure()
    comp_metrics = []

    for m, color in zip(methods, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        if m == "Rendement cible":
            w_m, r_m, v_m, s_m = compute_weights(m, mu, S, risk_free_rate, target_return_input)
        else:
            w_m, r_m, v_m, s_m = compute_weights(m, mu, S, risk_free_rate)

        w_m_ser = pd.Series(w_m).reindex(data.columns).fillna(0.0)
        port_ret_m = (rets * w_m_ser).sum(axis=1)
        cum_m = (1 + port_ret_m).cumprod()
        mask_m = (cum_m.index >= data_start_dynamic) & (cum_m.index <= data_end) 

        comp_fig.add_trace(go.Scatter(
            x=cum_m.index[mask_m], y=cum_m.values[mask_m],
            mode="lines", name=m, line=dict(color=color, width=3)
        ))

        cap_final_m = initial_investment * (1 + r_m) ** years
        plus_m = cap_final_m - initial_investment
        comp_metrics.append({
            "Méthode": m, "RendAnn": r_m, "VolAnn": v_m,
            "Sharpe": s_m, "CapFinal": cap_final_m, "PlusValue": plus_m
        })

    comp_fig.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Cumulé", margin=dict(t=30))
    st.plotly_chart(comp_fig, use_container_width=True)

    df_comp = pd.DataFrame(comp_metrics).set_index("Méthode")
    st.dataframe(df_comp.style.format({
        "RendAnn":"{:.2%}",
        "VolAnn":"{:.2%}",
        "Sharpe":"{:.2f}",
        "CapFinal":"{:.2f}",
        "PlusValue":"{:.2f}"
    }))

# --- TAB 2: Historical Prices ---
with tab2:
    st.subheader("Cours historiques — sélection d'actifs (derniers 2 ans)")
    all_assets = optimized_tickers
    chosen = st.multiselect("Choisir actif(s) à afficher", options=all_assets, default=all_assets) 

    if chosen:
        fig_hist = go.Figure()
        mask_hist = (data.index >= data_start_dynamic) & (data.index <= data_end) 
        for asset in chosen:
            fig_hist.add_trace(go.Scatter(
                x=data.index[mask_hist],
                y=data[asset].values[mask_hist],
                mode="lines", name=asset
            ))
        fig_hist.update_layout(template="plotly_dark", xaxis_title="Date", yaxis_title="Prix")
        st.plotly_chart(fig_hist, use_container_width=True)
