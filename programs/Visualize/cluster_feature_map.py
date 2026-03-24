import os
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_processing')))
import get_ghsom_dim


# ======================================================================
# ⭐ 全域 Cache（每個 job_id 只載入一次）
# ======================================================================
JOB_CACHE = {}     # { job_id : {df, has_label, cluster_means, pathlist, feature_cols} }


def load_job_into_cache(job_id):
    """
    若 job_id 不在 cache → 讀取資料並預先計算所有 expensive 元件
    """

    if job_id in JOB_CACHE:
        return JOB_CACHE[job_id]

    # ---- 找資料夾 ----
    app_root = "./applications"
    folders = [f for f in os.listdir(app_root) if f.startswith(job_id + "-")]
    if not folders:
        raise FileNotFoundError(f"No application folder for {job_id}")

    folder = folders[0]
    tau1, tau2 = map(float, folder.split("-")[1:3])

    # ---- 讀 CSV ----
    df_path = f"./applications/{folder}/data/{job_id}_with_clustered_label-{tau1}-{tau2}.csv"
    df = pd.read_csv(df_path)

    # ---- label ----
    label_path = f'./label/{job_id}_label.csv'
    has_label = os.path.exists(label_path)
    if has_label:
        df_label = pd.read_csv(label_path)
        df["label"] = df_label[df_label.columns[0]].values
    else:
        df["label"] = None

    df = df.fillna("")

    # ---- GHSOM 層級 ----
    layers, max_layer, _ = get_ghsom_dim.layers(f"{job_id}-{tau1}-{tau2}")

    pathlist = []
    for i in range(1, max_layer + 1):
        col = f"clusterL{i}"
        if col in df.columns and df[col].nunique() > 1:
            pathlist.append(col)

    # ---- 找 feature columns（一次性）----
    exclude_cols = set(pathlist + [
        "Event", "label", "clustered_label",
        "x_y_label", "point_x", "point_y",
        "mean", "median"
    ])
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # ---- 預先計算 cluster means 避免每次 callback 重算 ----
    cluster_means_cache = {}

    for depth, col in enumerate(pathlist, start=1):
        feature_means = df.groupby(col)[feature_cols].mean()
        cluster_means_cache[depth] = feature_means

    # ---- 存 cache ----
    info = {
        "df": df,
        "has_label": has_label,
        "pathlist": pathlist,
        "feature_cols": feature_cols,
        "cluster_means": cluster_means_cache,
        "folder": folder,
        "tau1": tau1,
        "tau2": tau2,
    }

    JOB_CACHE[job_id] = info
    return info


# ======================================================================
# ⭐ 建立 Dash app（只做一次）
# ======================================================================
def init_feature_map_dash(flask_app):

    dash_app = dash.Dash(
        __name__,
        server=flask_app,
        routes_pathname_prefix="/feature-map/",
        suppress_callback_exceptions=True
    )

    dash_app.title = "Feature Map"

    dash_app.layout = html.Div([
        dcc.Location(id='url', refresh=False),

        html.H2(id='title'),

        html.Div([
            dcc.Graph(id='treemap', style={'height': '90vh'})
        ], style={'width': '60%', 'display': 'inline-block', 'padding': '0 20'}),

        html.Div([
            dcc.Graph(id='feature-bar'),
            dcc.Graph(id='pie-chart', style={'display': 'none'})
        ], style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])

    # ==================================================================
    # ⭐ Callback：載入 Treemap（第一次）
    # ==================================================================
    @dash_app.callback(
        [Output('title', 'children'),
         Output('treemap', 'figure')],
        [Input('url', 'pathname')]
    )
    def load_treemap(pathname):

        if pathname is None:
            raise dash.exceptions.PreventUpdate

        job_id = pathname.replace("/feature-map/", "").strip("/")

        # ---- 使用 Cache ----
        try:
            info = load_job_into_cache(job_id)
        except:
            return f"Feature Map — Job {job_id} NOT FOUND", go.Figure()

        df = info["df"]
        pathlist = info["pathlist"]

        # ---- Treemap ----
        fig = px.treemap(
            df,
            path=pathlist,
            color='mean',
            color_continuous_scale='RdBu',
            branchvalues='remainder',
        )

        return f"Feature Map — Job {job_id}", fig

    # ==================================================================
    # ⭐ Callback：點擊 Treemap → 顯示 Top5 + Pie（極速版本）
    # ==================================================================
    @dash_app.callback(
        [Output('feature-bar', 'figure'),
         Output('pie-chart', 'figure'),
         Output('pie-chart', 'style')],
        [Input('treemap', 'clickData'),
         Input('url', 'pathname')]
    )
    def update_features(clickData, pathname):

        if pathname is None or clickData is None:
            raise dash.exceptions.PreventUpdate

        job_id = pathname.replace("/feature-map/", "").strip("/")
        info = load_job_into_cache(job_id)

        df = info["df"]
        has_label = info["has_label"]
        pathlist = info["pathlist"]
        feature_cols = info["feature_cols"]
        cluster_means_cache = info["cluster_means"]

        # ---- 找點到的 cluster ----
        clicked_id = clickData['points'][0]['id'].rstrip('/')
        levels = clicked_id.split('/')
        depth = len(levels)
        cluster_name = levels[-1]

        mask = df[f"clusterL{depth}"] == cluster_name
        sub_df = df[mask]

        # ---- Significant Feature 計算（極速）----
        cluster_means = cluster_means_cache[depth]
        all_clusters = cluster_means.index.tolist()

        sig_scores = {}

        for col in feature_cols:
            cluster_mean = sub_df[col].mean()
            sigma_I = np.sqrt(((sub_df[col] - cluster_mean) ** 2).sum() / len(sub_df))

            others = [c for c in all_clusters if c != cluster_name]
            m_c = cluster_means.loc[cluster_name, col]
            m_c_primes = cluster_means.loc[others, col]

            sigma_B = np.sqrt(((m_c - m_c_primes) ** 2).sum() / len(others))

            sig_scores[col] = sigma_B - sigma_I

        # Top 5
        top5 = sorted(sig_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        names = [x[0] for x in top5]
        values = [sub_df[x[0]].mean() for x in top5]

        fig_bar = go.Figure([
            go.Bar(x=values, y=names, orientation='h')
        ])
        fig_bar.update_layout(
            title="Top 5 Significant Features",
            yaxis={'autorange': 'reversed'}
        )

        # ---- Pie chart ----
        if not has_label:
            return fig_bar, go.Figure(), {'display': 'none'}

        counts = sub_df["label"].value_counts()
        total = counts.sum()
        counts = counts[counts / total >= 0.05]

        if len(counts) > 5:
            counts = counts[:5]

        others = total - counts.sum()
        if others > 0:
            counts["Others"] = others

        blue_colors = [
            'rgb(198,219,239)', 'rgb(158,202,225)', 'rgb(107,174,214)',
            'rgb(49,130,189)', 'rgb(8,81,156)', 'rgb(200,200,200)'
        ]

        fig_pie = go.Figure(
            go.Pie(
                labels=counts.index,
                values=counts.values,
                hole=0.5,
                marker_colors=blue_colors[:len(counts)],
                textinfo='percent+label'
            )
        )
        fig_pie.update_layout(title='Cell Type Distribution')

        return fig_bar, fig_pie, {'display': 'block'}

    return dash_app




















