import os
import sys
import argparse
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# 匯入 GHSOM 層級工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
import get_ghsom_dim


# ============================================================
# ⭐ 單獨版 Feature Map
# ============================================================
def run_feature_map_test(job_id, tau1, tau2, feature="mean"):

    folder = f"{job_id}-{tau1}-{tau2}"
    df_path = f"./applications/{folder}/data/{job_id}_with_clustered_label-{tau1}-{tau2}.csv"

    if not os.path.exists(df_path):
        raise FileNotFoundError(f"[ERROR] Cannot find: {df_path}")

    print(f"[LOAD] {df_path}")
    df = pd.read_csv(df_path)

    # ---- Label ----
    label_path = f"./label/{job_id}_label.csv"
    has_label = os.path.exists(label_path)
    if has_label:
        df_label = pd.read_csv(label_path)
        df["label"] = df_label[df_label.columns[0]].values
    else:
        df["label"] = None

    # ---- GHSOM 層級 ----
    layers, max_layer, _ = get_ghsom_dim.layers(folder)

    pathlist = []
    for i in range(1, max_layer + 1):
        col = f"clusterL{i}"
        if col in df.columns and df[col].nunique() > 1:
            pathlist.append(col)

    df = df.fillna("")

    # ---- Treemap ----
    fig_main = px.treemap(
        df,
        path=pathlist,
        color=feature,
        color_continuous_scale='RdBu',
        branchvalues='remainder'
    )

    # ============================================================
    # ⭐ 啟動 Dash（固定在 port 8050）
    # ============================================================
    app = dash.Dash(__name__)
    app.title = f"Feature Map — {job_id}"

    app.layout = html.Div([
        html.H2(f"Feature Map — Job {job_id}"),

        html.Div([
            dcc.Graph(id="treemap", figure=fig_main, style={"height": "85vh"})
        ], style={"width": "60%", "display": "inline-block", "padding": "0 20"}),

        html.Div([
            dcc.Graph(id="feature-bar"),
            dcc.Graph(id="pie-chart", style={"display": "none"})
        ], style={"width": "35%", "display": "inline-block", "verticalAlign": "top"})
    ])

    # ============================================================
    # ⭐ 點擊 Treemap Callback
    # ============================================================
    @app.callback(
        [Output("feature-bar", "figure"),
         Output("pie-chart", "figure"),
         Output("pie-chart", "style")],
        [Input("treemap", "clickData")]
    )
    def update_features(clickData):

        if clickData is None:
            raise dash.exceptions.PreventUpdate

        clicked_id = clickData["points"][0]["id"].rstrip("/")
        levels = clicked_id.split("/")
        depth = len(levels)
        cluster_name = levels[-1]

        mask = df[f"clusterL{depth}"] == cluster_name
        sub_df = df[mask]

        # ---- Features ----
        exclude = set(pathlist + [
            "Event", "label", "clustered_label", "x_y_label",
            "point_x", "point_y", "mean", "median"
        ])
        feature_cols = [c for c in df.columns if c not in exclude]

        cluster_means = df.groupby(f"clusterL{depth}")[feature_cols].mean()
        all_clusters = cluster_means.index.tolist()

        sig_scores = {}

        # ---- Significant Feature 計算 ----
        for col in feature_cols:
            cluster_mean = sub_df[col].mean()
            sigma_I = np.sqrt(((sub_df[col] - cluster_mean) ** 2).sum() / len(sub_df))

            others = [c for c in all_clusters if c != cluster_name]
            m_c = cluster_means.loc[cluster_name, col]
            m_c_primes = cluster_means.loc[others, col]
            sigma_B = np.sqrt(((m_c - m_c_primes) ** 2).sum() / len(others))

            sig_scores[col] = sigma_B - sigma_I

        top5 = sorted(sig_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        names = [x[0] for x in top5]
        values = [sub_df[x[0]].mean() for x in top5]

        fig_bar = go.Figure([go.Bar(x=values, y=names, orientation="h")])
        fig_bar.update_layout(
            title="Top 5 Significant Features",
            yaxis={"autorange": "reversed"}
        )

        # ---- Pie Chart ----
        if not has_label:
            return fig_bar, go.Figure(), {"display": "none"}

        counts = sub_df["label"].value_counts()
        total = counts.sum()
        counts = counts[counts / total >= 0.05]

        if len(counts) > 5:
            counts = counts[:5]

        others = total - counts.sum()
        if others > 0:
            counts["Others"] = others

        fig_pie = go.Figure(go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.5
        ))
        fig_pie.update_layout(title="Cell Type Distribution")

        return fig_bar, fig_pie, {"display": "block"}

    # ---- Run Dash（新版 Dash 必須用 app.run）----
    print("[DASH] Running at http://127.0.0.1:8050")
    app.run(debug=True, port=8050)


# ============================================================
# ⭐ CLI 入口點
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--tau1", type=float, required=True)
    parser.add_argument("--tau2", type=float, required=True)
    parser.add_argument("--feature", default="mean")

    args = parser.parse_args()

    run_feature_map_test(
        job_id=args.data,
        tau1=args.tau1,
        tau2=args.tau2,
        feature=args.feature
    )

# python3 programs/Visualize/cluster_feature_map.py --data=Samusik_01_cleaned --tau1=1.0 --tau2=1.0






