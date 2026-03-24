import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from dash import Dash, dcc, html, dash_table, Input, Output, State, no_update


# ============================================================
# Utilities
# ============================================================

def fmt_percent(x):
    if pd.isna(x):
        return "-"
    return f"{x * 100:.2f}%"


def fmt_cells(x):
    if pd.isna(x):
        return "-"
    return f"{int(round(x))} cells"


def get_project_root():
    return Path(__file__).resolve().parents[2]


def build_sibling_maps(hierarchy_df):
    """
    hierarchy csv 格式：
    cluster_number,layer,sibling_group
    """
    df = hierarchy_df.copy()
    df["cluster_number"] = df["cluster_number"].astype(int)
    df["sibling_group"] = df["sibling_group"].astype(str)

    group_map = {}
    siblings_map = {}

    for _, row in df.iterrows():
        c = int(row["cluster_number"])
        g = str(row["sibling_group"])
        group_map[c] = g

    valid_group_df = df[df["sibling_group"].str.upper() != "NONE"].copy()
    group_to_clusters = (
        valid_group_df.groupby("sibling_group")["cluster_number"]
        .apply(lambda x: sorted([int(v) for v in x.tolist()]))
        .to_dict()
    )

    for _, row in df.iterrows():
        c = int(row["cluster_number"])
        g = str(row["sibling_group"])

        if g.upper() == "NONE":
            siblings_map[c] = []
        else:
            sibs = group_to_clusters.get(g, [])
            siblings_map[c] = [v for v in sibs if v != c]

    return group_map, siblings_map


def get_cluster_feature_table(cluster_number, center_df, weight_df):
    """
    回傳欄位：
    feature | importance | mean
    並依 importance 由大到小排序
    """
    center_row = center_df[center_df["cluster_number"] == cluster_number]
    weight_row = weight_df[weight_df["cluster_number"] == cluster_number]

    if center_row.empty or weight_row.empty:
        return pd.DataFrame(columns=["feature", "importance", "mean"])

    center_row = center_row.iloc[0]
    weight_row = weight_row.iloc[0]

    feature_cols = [c for c in center_df.columns if c != "cluster_number"]

    rows = []
    for feat in feature_cols:
        rows.append({
            "feature": feat,
            "importance": float(weight_row[feat]),
            "mean": float(center_row[feat])
        })

    feat_df = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    return feat_df


def get_distribution_slice(distribution_df, time_name, patient_name):
    sub = distribution_df[
        (distribution_df["time"].astype(str) == str(time_name)) &
        (distribution_df["patient"].astype(str) == str(patient_name))
    ].copy()

    if sub.empty:
        return pd.DataFrame(columns=["patient", "time", "cluster_number", "count", "prob"])

    sub["cluster_number"] = sub["cluster_number"].astype(int)
    return sub


def build_snapshot_df(coord_df, dist_df):
    merged = coord_df.merge(
        dist_df[["cluster_number", "count", "prob"]],
        on="cluster_number",
        how="left"
    )
    merged["count"] = merged["count"].fillna(0)
    merged["prob"] = merged["prob"].fillna(0.0)
    return merged


def get_top_k_targets(prob_matrix, source_cluster, k=3):
    if source_cluster not in prob_matrix.index:
        return []

    row = prob_matrix.loc[source_cluster].copy()
    row = row[row > 0].sort_values(ascending=False)
    return row.head(k).index.astype(int).tolist()


def build_flow_table(source_cluster, prob_matrix, count_matrix, source_dist_df):
    if source_cluster not in prob_matrix.index or source_cluster not in count_matrix.index:
        return pd.DataFrame(columns=[
            "target_cluster",
            "transport_cells_raw",
            "transport_prob_raw",
            "cluster_level_prob_raw",
            "transport_cells",
            "transport_prob",
            "cluster_level_prob"
        ])

    source_count_row = source_dist_df[source_dist_df["cluster_number"] == source_cluster]
    source_count = float(source_count_row["count"].iloc[0]) if not source_count_row.empty else 0.0

    rows = []
    for target_cluster in prob_matrix.columns:
        joint_prob = float(prob_matrix.loc[source_cluster, target_cluster])
        flow_cells = float(count_matrix.loc[source_cluster, target_cluster])
        cluster_level_prob = flow_cells / source_count if source_count > 0 else 0.0

        rows.append({
            "target_cluster": int(target_cluster),
            "transport_cells_raw": flow_cells,
            "transport_prob_raw": joint_prob,
            "cluster_level_prob_raw": cluster_level_prob,
            "transport_cells": fmt_cells(flow_cells),
            "transport_prob": fmt_percent(joint_prob),
            "cluster_level_prob": fmt_percent(cluster_level_prob),
        })

    table_df = pd.DataFrame(rows).sort_values(
        by=["transport_prob_raw", "transport_cells_raw"],
        ascending=False
    ).reset_index(drop=True)

    return table_df


def make_node_customdata(df, side_name):
    return np.array([
        [
            side_name,
            int(r["cluster_number"]),
            float(r["point_x"]),
            float(r["point_y"]),
            float(r["count"]),
            float(r["prob"])
        ]
        for _, r in df.iterrows()
    ], dtype=object)


def get_cluster_info_dict(cluster_number, time_name, snapshot_df, siblings_map,
                          center_df, weight_df, side_name):
    if cluster_number is None:
        return {
            "side": side_name,
            "cluster_number": None,
            "time_name": time_name,
            "point_x": None,
            "point_y": None,
            "count": None,
            "prob": None,
            "siblings": [],
            "top5": pd.DataFrame(columns=["feature", "importance", "mean"])
        }

    row = snapshot_df[snapshot_df["cluster_number"] == cluster_number]
    if row.empty:
        return {
            "side": side_name,
            "cluster_number": cluster_number,
            "time_name": time_name,
            "point_x": None,
            "point_y": None,
            "count": None,
            "prob": None,
            "siblings": siblings_map.get(cluster_number, []),
            "top5": pd.DataFrame(columns=["feature", "importance", "mean"])
        }

    row = row.iloc[0]
    feat_df = get_cluster_feature_table(cluster_number, center_df, weight_df)
    top5 = feat_df.head(5)

    return {
        "side": side_name,
        "cluster_number": cluster_number,
        "time_name": time_name,
        "point_x": float(row["point_x"]),
        "point_y": float(row["point_y"]),
        "count": float(row["count"]),
        "prob": float(row["prob"]),
        "siblings": siblings_map.get(cluster_number, []),
        "top5": top5
    }


def render_cluster_info_from_dict(info):
    if info is None or info.get("cluster_number") is None:
        return html.Div("請先 hover 任一個 cluster", style={"padding": "10px", "color": "#666"})

    feature_rows = []
    top5 = info["top5"]
    for _, r in top5.iterrows():
        feature_rows.append(
            html.Tr([
                html.Td(r["feature"], style={"padding": "6px", "borderBottom": "1px solid #eee"}),
                html.Td(f'{r["importance"]:.4f}', style={"padding": "6px", "borderBottom": "1px solid #eee"}),
                html.Td(f'{r["mean"]:.4f}', style={"padding": "6px", "borderBottom": "1px solid #eee"})
            ])
        )

    siblings = info.get("siblings", [])
    siblings_text = ", ".join(str(x) for x in siblings) if len(siblings) > 0 else "None"

    return html.Div([
        html.H4(f"{info['side'].capitalize()} Cluster {info['cluster_number']}", style={"marginBottom": "8px"}),
        html.P(f"Time: {info['time_name']}", style={"margin": "4px 0"}),
        html.P(f"Coordinate: ({info['point_x']:.4f}, {info['point_y']:.4f})", style={"margin": "4px 0"}),
        html.P(f"Cell count: {int(round(info['count']))}", style={"margin": "4px 0"}),
        html.P(f"Cluster proportion: {fmt_percent(info['prob'])}", style={"margin": "4px 0"}),
        html.P(f"Siblings: {siblings_text}", style={"margin": "4px 0 12px 0"}),
        html.H5("Top 5 important features", style={"marginBottom": "8px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Feature", style={"textAlign": "left", "padding": "6px", "borderBottom": "2px solid #ddd"}),
                html.Th("Importance", style={"textAlign": "left", "padding": "6px", "borderBottom": "2px solid #ddd"}),
                html.Th("Mean", style={"textAlign": "left", "padding": "6px", "borderBottom": "2px solid #ddd"})
            ])),
            html.Tbody(feature_rows)
        ], style={"width": "100%", "borderCollapse": "collapse"})
    ])


def build_all_features_table(cluster_number, center_df, weight_df):
    if cluster_number is None:
        return []

    feat_df = get_cluster_feature_table(cluster_number, center_df, weight_df)
    if feat_df.empty:
        return []

    feat_df["importance"] = feat_df["importance"].map(lambda x: f"{x:.4f}")
    feat_df["mean"] = feat_df["mean"].map(lambda x: f"{x:.4f}")
    return feat_df.to_dict("records")


def scan_patient_list_from_result_dir(result_dir):
    """
    從 result 資料夾倒著抓 patient 名單
    檔名格式：
        {patient}_prob.npy
        {patient}_count.npy
    只保留 prob/count 都存在的 patient
    """
    if not result_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 patient result folder：{result_dir}")

    prob_patients = set()
    count_patients = set()

    for f in result_dir.iterdir():
        if not f.is_file():
            continue

        name = f.name
        if name.endswith("_prob.npy"):
            prob_patients.add(name[:-len("_prob.npy")])
        elif name.endswith("_count.npy"):
            count_patients.add(name[:-len("_count.npy")])

    patient_list = sorted(prob_patients & count_patients)

    if len(patient_list) == 0:
        raise ValueError(f"❌ {result_dir} 中找不到任何成對的 patient_prob.npy / patient_count.npy")

    return patient_list


def make_base_figure(source_snapshot, target_snapshot, source_time, target_time,
                     active_info=None, prob_matrix=None, top_targets=None):
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.12,
        subplot_titles=(f"Source: {source_time}", f"Target: {target_time}")
    )

    all_counts = pd.concat([source_snapshot["count"], target_snapshot["count"]], axis=0)
    cmin, cmax = all_counts.min(), all_counts.max()

    def scale_size(v):
        if cmax == cmin:
            return 20
        return 14 + (v - cmin) / (cmax - cmin) * 32

    src_sizes = source_snapshot["count"].apply(scale_size)
    tgt_sizes = target_snapshot["count"].apply(scale_size)

    src_line_width = [2] * len(source_snapshot)
    src_line_color = ["white"] * len(source_snapshot)
    tgt_line_width = [2] * len(target_snapshot)
    tgt_line_color = ["white"] * len(target_snapshot)

    if active_info is not None and active_info.get("cluster_number") is not None:
        active_side = active_info.get("side")
        active_cluster = int(active_info.get("cluster_number"))

        if active_side == "source":
            for i, c in enumerate(source_snapshot["cluster_number"].tolist()):
                if int(c) == active_cluster:
                    src_line_width[i] = 4
                    src_line_color[i] = "#c62828"

            if top_targets is not None:
                top_targets_set = set(int(x) for x in top_targets)
                for i, c in enumerate(target_snapshot["cluster_number"].tolist()):
                    if int(c) in top_targets_set:
                        tgt_line_width[i] = 4
                        tgt_line_color[i] = "#c62828"

        elif active_side == "target":
            for i, c in enumerate(target_snapshot["cluster_number"].tolist()):
                if int(c) == active_cluster:
                    tgt_line_width[i] = 4
                    tgt_line_color[i] = "#c62828"

    fig.add_trace(go.Scatter(
        x=source_snapshot["point_x"],
        y=source_snapshot["point_y"],
        mode="markers+text",
        text=source_snapshot["cluster_number"].astype(str),
        textposition="middle center",
        marker=dict(
            size=src_sizes,
            color=source_snapshot["count"],
            colorscale="Blues",
            line=dict(width=src_line_width, color=src_line_color),
            opacity=0.92,
            showscale=False,
        ),
        customdata=make_node_customdata(source_snapshot, "source"),
        hovertemplate=(
            "<b>Source Cluster %{customdata[1]}</b><br>"
            "Coordinate: (%{customdata[2]:.4f}, %{customdata[3]:.4f})<br>"
            "Count: %{customdata[4]:.0f}<br>"
            "Proportion: %{customdata[5]:.2%}<extra></extra>"
        ),
        showlegend=False
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=target_snapshot["point_x"],
        y=target_snapshot["point_y"],
        mode="markers+text",
        text=target_snapshot["cluster_number"].astype(str),
        textposition="middle center",
        marker=dict(
            size=tgt_sizes,
            color=target_snapshot["count"],
            colorscale="Oranges",
            line=dict(width=tgt_line_width, color=tgt_line_color),
            opacity=0.92,
            showscale=False,
        ),
        customdata=make_node_customdata(target_snapshot, "target"),
        hovertemplate=(
            "<b>Target Cluster %{customdata[1]}</b><br>"
            "Coordinate: (%{customdata[2]:.4f}, %{customdata[3]:.4f})<br>"
            "Count: %{customdata[4]:.0f}<br>"
            "Proportion: %{customdata[5]:.2%}<extra></extra>"
        ),
        showlegend=False
    ), row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=760,
        margin=dict(l=40, r=40, t=80, b=40),
        showlegend=False,
        hovermode="closest",
    )

    fig.update_xaxes(range=[0, 1], title="point_x", showgrid=True, zeroline=False, row=1, col=1)
    fig.update_yaxes(range=[0, 1], title="point_y", showgrid=True, zeroline=False, row=1, col=1)
    fig.update_xaxes(range=[0, 1], title="point_x", showgrid=True, zeroline=False, row=1, col=2)
    fig.update_yaxes(range=[0, 1], title="point_y", showgrid=True, zeroline=False, row=1, col=2)

    for ann in fig.layout.annotations:
        if ann.text and ("Source:" in ann.text or "Target:" in ann.text):
            ann.font = dict(size=18)

    return fig


# ============================================================
# Main loader
# ============================================================

def load_transport_matrix_for_patient(result_dir, patient_name, cluster_order):
    prob_path = result_dir / f"{patient_name}_prob.npy"
    count_path = result_dir / f"{patient_name}_count.npy"

    if not prob_path.exists():
        raise FileNotFoundError(f"❌ 找不到 patient prob npy：{prob_path}")
    if not count_path.exists():
        raise FileNotFoundError(f"❌ 找不到 patient count npy：{count_path}")

    prob_arr = np.load(prob_path)
    count_arr = np.load(count_path)

    if prob_arr.ndim != 2 or count_arr.ndim != 2:
        raise ValueError(f"❌ patient transport matrix 維度錯誤：{patient_name}")

    n = len(cluster_order)
    if prob_arr.shape != (n, n):
        raise ValueError(
            f"❌ prob matrix shape mismatch for {patient_name}: expected {(n, n)}, got {prob_arr.shape}"
        )
    if count_arr.shape != (n, n):
        raise ValueError(
            f"❌ count matrix shape mismatch for {patient_name}: expected {(n, n)}, got {count_arr.shape}"
        )

    prob_matrix = pd.DataFrame(prob_arr, index=cluster_order, columns=cluster_order)
    count_matrix = pd.DataFrame(count_arr, index=cluster_order, columns=cluster_order)

    prob_matrix.index = prob_matrix.index.astype(int)
    prob_matrix.columns = prob_matrix.columns.astype(int)
    count_matrix.index = count_matrix.index.astype(int)
    count_matrix.columns = count_matrix.columns.astype(int)

    return prob_matrix, count_matrix


def load_all_data(data, tau1, tau2, source, target):
    root = get_project_root()
    data_root = root / "optimal_transport" / "data"

    coord_path = data_root / f"{data}-{tau1}-{tau2}_cluster_coordinate.csv"
    center_path = data_root / f"{data}-{tau1}-{tau2}_cluster_center.csv"
    weight_path = data_root / f"{data}-{tau1}-{tau2}_cluster_weight.csv"
    distribution_path = data_root / f"{data}-{tau1}-{tau2}_distribution.csv"
    hierarchy_path = root / "optimal_transport" / "hierarchy" / f"{data}_cluster_hierarchy-{tau1}-{tau2}.csv"
    result_dir = root / "optimal_transport" / "result" / f"{data}-{tau1}-{tau2}_{source}_to_{target}"

    if not coord_path.exists():
        raise FileNotFoundError(f"❌ 找不到 cluster coordinate：{coord_path}")
    if not center_path.exists():
        raise FileNotFoundError(f"❌ 找不到 cluster center：{center_path}")
    if not weight_path.exists():
        raise FileNotFoundError(f"❌ 找不到 cluster weight：{weight_path}")
    if not distribution_path.exists():
        raise FileNotFoundError(f"❌ 找不到 distribution：{distribution_path}")
    if not hierarchy_path.exists():
        raise FileNotFoundError(f"❌ 找不到 hierarchy csv：{hierarchy_path}")
    if not result_dir.exists():
        raise FileNotFoundError(f"❌ 找不到 patient result folder：{result_dir}")

    coord_df = pd.read_csv(coord_path)
    center_df = pd.read_csv(center_path)
    weight_df = pd.read_csv(weight_path)
    distribution_df = pd.read_csv(distribution_path)
    hierarchy_df = pd.read_csv(hierarchy_path)

    coord_df["cluster_number"] = coord_df["cluster_number"].astype(int)
    center_df["cluster_number"] = center_df["cluster_number"].astype(int)
    weight_df["cluster_number"] = weight_df["cluster_number"].astype(int)
    distribution_df["cluster_number"] = distribution_df["cluster_number"].astype(int)
    hierarchy_df["cluster_number"] = hierarchy_df["cluster_number"].astype(int)

    coord_df = coord_df.sort_values("cluster_number").reset_index(drop=True)
    center_df = center_df.sort_values("cluster_number").reset_index(drop=True)
    weight_df = weight_df.sort_values("cluster_number").reset_index(drop=True)
    hierarchy_df = hierarchy_df.sort_values("cluster_number").reset_index(drop=True)

    cluster_order = coord_df["cluster_number"].astype(int).tolist()

    if "patient" not in distribution_df.columns:
        raise ValueError("❌ distribution.csv 缺少 patient 欄位")

    patient_list = scan_patient_list_from_result_dir(result_dir)
    group_map, siblings_map = build_sibling_maps(hierarchy_df)

    return {
        "root": root,
        "coord_df": coord_df,
        "center_df": center_df,
        "weight_df": weight_df,
        "distribution_df": distribution_df,
        "hierarchy_df": hierarchy_df,
        "group_map": group_map,
        "siblings_map": siblings_map,
        "cluster_order": cluster_order,
        "result_dir": result_dir,
        "patient_list": patient_list,
    }


# ============================================================
# Dash App
# ============================================================

def create_app(data, tau1, tau2, source, target):
    loaded = load_all_data(data, tau1, tau2, source, target)
    patient_cache = {}

    def get_patient_context(patient_name):
        patient_name = str(patient_name)

        if patient_name in patient_cache:
            return patient_cache[patient_name]

        source_dist_df = get_distribution_slice(loaded["distribution_df"], source, patient_name)
        target_dist_df = get_distribution_slice(loaded["distribution_df"], target, patient_name)

        source_snapshot = build_snapshot_df(loaded["coord_df"], source_dist_df)
        target_snapshot = build_snapshot_df(loaded["coord_df"], target_dist_df)

        prob_matrix, count_matrix = load_transport_matrix_for_patient(
            loaded["result_dir"],
            patient_name,
            loaded["cluster_order"]
        )

        ctx = {
            "source_dist_df": source_dist_df,
            "target_dist_df": target_dist_df,
            "source_snapshot": source_snapshot,
            "target_snapshot": target_snapshot,
            "prob_matrix": prob_matrix,
            "count_matrix": count_matrix,
        }

        patient_cache[patient_name] = ctx
        return ctx

    default_patient = loaded["patient_list"][0]
    init_ctx = get_patient_context(default_patient)

    source_total = int(round(init_ctx["source_snapshot"]["count"].sum()))
    target_total = int(round(init_ctx["target_snapshot"]["count"].sum()))

    app = Dash(__name__)

    app.layout = html.Div([
        html.H2(f"OT Map: {data} | τ1={tau1}, τ2={tau2} | {source} → {target}"),

        html.Div([
            html.Label("Patient", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="patient-dropdown",
                options=[{"label": p, "value": p} for p in loaded["patient_list"]],
                value=default_patient,
                clearable=False,
                style={"width": "360px"}
            ),
        ], style={"marginBottom": "15px"}),

        html.Div([
            html.Div(
                id="source-total-text",
                children=f"Source total cells: {source_total}",
                style={"fontSize": "18px", "fontWeight": "bold"}
            ),
            html.Div(
                id="target-total-text",
                children=f"Target total cells: {target_total}",
                style={"fontSize": "18px", "fontWeight": "bold", "marginLeft": "30px"}
            ),
        ], style={"display": "flex", "gap": "20px", "marginBottom": "15px"}),

        dcc.Store(id="hovered-cluster-info"),

        html.Div([
            html.Div([
                dcc.Graph(
                    id="ot-map-graph",
                    figure=make_base_figure(
                        init_ctx["source_snapshot"],
                        init_ctx["target_snapshot"],
                        source,
                        target,
                        active_info=None
                    ),
                    style={"height": "780px"},
                    clear_on_unhover=False
                )
            ], style={"width": "68%", "display": "inline-block", "verticalAlign": "top"}),

            html.Div([
                html.Div(id="cluster-info-panel", style={
                    "border": "1px solid #ddd",
                    "borderRadius": "10px",
                    "padding": "12px",
                    "marginBottom": "16px",
                    "backgroundColor": "#fafafa"
                }),
            ], style={"width": "30%", "display": "inline-block", "verticalAlign": "top", "marginLeft": "2%"})
        ]),

        html.Hr(),

        html.H3("Flow summary table"),
        dash_table.DataTable(
            id="flow-table",
            columns=[
                {"name": "Target Cluster", "id": "target_cluster"},
                {"name": "Transport Cells", "id": "transport_cells"},
                {"name": "Population-level OT Proportion", "id": "transport_prob"},
                {"name": "Cluster-level OT Proportion", "id": "cluster_level_prob"},
            ],
            data=[],
            page_size=12,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"fontWeight": "bold"}
        ),

        html.Hr(),

        html.H3("All feature list"),
        dash_table.DataTable(
            id="all-features-table",
            columns=[
                {"name": "Feature", "id": "feature"},
                {"name": "Importance", "id": "importance"},
                {"name": "Mean", "id": "mean"},
            ],
            data=[],
            page_size=15,
            style_table={"overflowX": "auto"},
            style_cell={"textAlign": "left", "padding": "8px"},
            style_header={"fontWeight": "bold"}
        )
    ], style={"padding": "20px"})

    @app.callback(
        Output("source-total-text", "children"),
        Output("target-total-text", "children"),
        Input("patient-dropdown", "value")
    )
    def update_total_text(selected_patient):
        ctx = get_patient_context(selected_patient)
        source_total_now = int(round(ctx["source_snapshot"]["count"].sum()))
        target_total_now = int(round(ctx["target_snapshot"]["count"].sum()))
        return (
            f"Source total cells: {source_total_now}",
            f"Target total cells: {target_total_now}"
        )

    @app.callback(
        Output("hovered-cluster-info", "data"),
        Input("ot-map-graph", "hoverData"),
        Input("patient-dropdown", "value"),
        State("hovered-cluster-info", "data")
    )
    def update_hovered_cluster_info(hoverData, selected_patient, old_info):
        ctx = get_patient_context(selected_patient)

        if hoverData is not None and "points" in hoverData and len(hoverData["points"]) > 0:
            point = hoverData["points"][0]
            customdata = point.get("customdata", None)

            if customdata is None:
                return no_update

            side = customdata[0]
            if side not in ("source", "target"):
                return no_update

            cluster_number = int(customdata[1])

            if side == "source":
                info = get_cluster_info_dict(
                    cluster_number,
                    source,
                    ctx["source_snapshot"],
                    loaded["siblings_map"],
                    loaded["center_df"],
                    loaded["weight_df"],
                    "source"
                )
            else:
                info = get_cluster_info_dict(
                    cluster_number,
                    target,
                    ctx["target_snapshot"],
                    loaded["siblings_map"],
                    loaded["center_df"],
                    loaded["weight_df"],
                    "target"
                )

            return {
                "side": info["side"],
                "cluster_number": info["cluster_number"],
                "time_name": info["time_name"],
                "point_x": info["point_x"],
                "point_y": info["point_y"],
                "count": info["count"],
                "prob": info["prob"],
                "siblings": info["siblings"],
                "top5": info["top5"].to_dict("records")
            }

        if old_info is None:
            return None

        cluster_number = old_info.get("cluster_number")
        side = old_info.get("side")

        if cluster_number is None or side not in ("source", "target"):
            return old_info

        if side == "source":
            info = get_cluster_info_dict(
                int(cluster_number),
                source,
                ctx["source_snapshot"],
                loaded["siblings_map"],
                loaded["center_df"],
                loaded["weight_df"],
                "source"
            )
        else:
            info = get_cluster_info_dict(
                int(cluster_number),
                target,
                ctx["target_snapshot"],
                loaded["siblings_map"],
                loaded["center_df"],
                loaded["weight_df"],
                "target"
            )

        return {
            "side": info["side"],
            "cluster_number": info["cluster_number"],
            "time_name": info["time_name"],
            "point_x": info["point_x"],
            "point_y": info["point_y"],
            "count": info["count"],
            "prob": info["prob"],
            "siblings": info["siblings"],
            "top5": info["top5"].to_dict("records")
        }

    @app.callback(
        Output("ot-map-graph", "figure"),
        Input("patient-dropdown", "value"),
        Input("hovered-cluster-info", "data")
    )
    def update_graph(selected_patient, hovered_info):
        ctx = get_patient_context(selected_patient)

        top_targets = None
        if hovered_info is not None and hovered_info.get("side") == "source" and hovered_info.get("cluster_number") is not None:
            active_source = int(hovered_info["cluster_number"])
            top_targets = get_top_k_targets(ctx["prob_matrix"], active_source, k=3)

        return make_base_figure(
            ctx["source_snapshot"],
            ctx["target_snapshot"],
            source,
            target,
            active_info=hovered_info,
            prob_matrix=ctx["prob_matrix"],
            top_targets=top_targets
        )

    @app.callback(
        Output("cluster-info-panel", "children"),
        Input("hovered-cluster-info", "data")
    )
    def update_cluster_panel(hovered_info):
        if hovered_info is None or hovered_info.get("cluster_number") is None:
            return html.Div("請先 hover 任一個 cluster", style={"padding": "10px", "color": "#666"})

        info = hovered_info.copy()
        info["top5"] = pd.DataFrame(info.get("top5", []))
        return render_cluster_info_from_dict(info)

    @app.callback(
        Output("flow-table", "data"),
        Input("patient-dropdown", "value"),
        Input("hovered-cluster-info", "data")
    )
    def update_flow_table(selected_patient, hovered_info):
        ctx = get_patient_context(selected_patient)

        if hovered_info is None:
            return []

        if hovered_info.get("side") != "source":
            return []

        active_source = hovered_info.get("cluster_number", None)
        if active_source is None:
            return []

        table_df = build_flow_table(
            int(active_source),
            ctx["prob_matrix"],
            ctx["count_matrix"],
            ctx["source_dist_df"]
        )

        return table_df[[
            "target_cluster",
            "transport_cells",
            "transport_prob",
            "cluster_level_prob"
        ]].to_dict("records")

    @app.callback(
        Output("all-features-table", "data"),
        Input("hovered-cluster-info", "data")
    )
    def update_all_features_table(hovered_info):
        if hovered_info is None:
            return []

        cluster_number = hovered_info.get("cluster_number", None)
        if cluster_number is None:
            return []

        return build_all_features_table(
            int(cluster_number),
            loaded["center_df"],
            loaded["weight_df"]
        )

    return app


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive OT map visualization")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--tau1", type=float, required=True)
    parser.add_argument("--tau2", type=float, required=True)
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--port", type=int, default=8050)

    args = parser.parse_args()

    app = create_app(
        data=args.data,
        tau1=args.tau1,
        tau2=args.tau2,
        source=args.source,
        target=args.target
    )

    app.run(debug=True, host="0.0.0.0", port=args.port)

# python optimal_transport/visualize/ot_map.py --data=CART_0320 --tau1=0.5 --tau2=0.5 --source=day0 --target=day7