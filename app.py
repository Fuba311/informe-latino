# -*- coding: utf-8 -*-
"""
LATAM Territorial Poverty Map — Render-ready Dash app.

Boot behavior:
- On startup, downloads (if missing) the two data files from
  https://github.com/Fuba311/informe-latino (branch MAIN) and saves
  them next to this app.py. Override via env vars if needed.

Env overrides (optional):
  DATA_REPO=Owner/Repo         (default: Fuba311/informe-latino)
  DATA_BRANCH=branch_or_tag    (default: main)
  PANEL_URL=...                (full URL to parquet; overrides repo/branch)
  GEOJSON_URL=...              (full URL to geojson; overrides repo/branch)
  DASH_DEBUG=0/1               (default 1 locally; Render sets to 0 typically)

Run locally:
  pip install -r requirements.txt
  python app.py
On Render:
  start command: gunicorn app:server --workers 2 --threads 8 --timeout 120
"""

import os
import re
import json
import warnings
from pathlib import Path
from functools import lru_cache

# --- network fetch (Raw GitHub) ------------------------------------------------
from typing import Tuple, Dict, List
import requests

def _raw_url(owner_repo: str, branch: str, filename: str) -> str:
    return f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{filename}"

def download_if_needed(url: str, dest: Path, timeout: int = 60, chunk: int = 1 << 20) -> Path:
    """Idempotent, streaming download with atomic write."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
    os.replace(tmp, dest)
    return dest

# --- scientific stack ----------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- Dash ----------------------------------------------------------------------
import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, MATCH
try:
    # Dash >=2.9 has Patch (partial property updates)
    from dash import Patch
    HAVE_PATCH = True
except Exception:
    HAVE_PATCH = False

print("--- STARTING LATAM POVERTY DASHBOARD ---")

# =============================================================================
# 1) PATHS & FETCH REMOTE DATA
# =============================================================================
BASE_DIR = Path(__file__).parent

# Remote repo defaults (can be overridden with env vars)
DATA_REPO = os.getenv("DATA_REPO", "Fuba311/informe-latino")
DATA_BRANCH = os.getenv("DATA_BRANCH", "main")

# Filenames we expect in that repo (at root)
PANEL_FILE = "panel_territorial.parquet"
GEOJSON_FILE = "latam_regiones_simplified.geojson"

# Allow full-URL overrides
PANEL_URL = os.getenv("PANEL_URL") or _raw_url(DATA_REPO, DATA_BRANCH, PANEL_FILE)
GEOJSON_URL = os.getenv("GEOJSON_URL") or _raw_url(DATA_REPO, DATA_BRANCH, GEOJSON_FILE)

PANEL_PATH = BASE_DIR / PANEL_FILE
GEOJSON_PATH = BASE_DIR / GEOJSON_FILE

# Fetch if missing (Render’s disk is ephemeral, so we download at boot)
print(f"→ Ensuring data files exist:\n  {PANEL_FILE}\n  {GEOJSON_FILE}")
download_if_needed(PANEL_URL, PANEL_PATH)
download_if_needed(GEOJSON_URL, GEOJSON_PATH)

# =============================================================================
# 2) LOADING & PRECOMPUTATIONS
# =============================================================================
required_cols = {
    "country_code", "country_name", "periodo",
    "region_code", "region_name", "feature_id",
    "subset_key", "agri_def",
    "indicator_label", "indicator_col",
    "value_pct", "se_pp", "ci95_lo", "ci95_hi", "sample_n"
}

# ---- Load panel
panel = pd.read_parquet(PANEL_PATH)
panel["feature_id"] = panel["feature_id"].astype(str)

missing = required_cols - set(panel.columns)
if missing:
    raise ValueError(f"Faltan columnas en el Parquet: {missing}")

# Keep only what we need (smaller DataFrame → faster serialization)
panel = panel[list(required_cols)].copy()

# Dtypes for speed/memory
for c in ["value_pct", "se_pp", "ci95_lo", "ci95_hi"]:
    panel[c] = pd.to_numeric(panel[c], errors="coerce").astype("float32")
panel["sample_n"] = pd.to_numeric(panel["sample_n"], errors="coerce").astype("float32")

cat_cols = [
    "country_code", "country_name", "subset_key", "agri_def",
    "indicator_label", "indicator_col", "region_name", "periodo"
]
for c in cat_cols:
    if panel[c].dtype != "category":
        panel[c] = panel[c].astype("category")

# ---- Load GeoJSON
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    GEOJSON_ALL = json.load(f)

FEATURES: List[dict] = GEOJSON_ALL.get("features", [])
FEATURE_BY_ID: Dict[str, dict] = {feat.get("properties", {}).get("id"): feat for feat in FEATURES}

# Map ISO3 -> set(feature_ids) for quick per-country subset
COUNTRY_FEATURE_IDS: Dict[str, set] = {}
for feat in FEATURES:
    fid = feat.get("properties", {}).get("id")
    if not fid or "|" not in fid:
        continue
    iso3 = fid.split("|", 1)[0]
    COUNTRY_FEATURE_IDS.setdefault(iso3, set()).add(fid)

# ---- Precompute bounding boxes (massive perf gain on scope switches)
def _compute_bbox_from_geometry(geom) -> Tuple[float, float, float, float]:
    """Return (lonW, latS, lonE, latN)."""
    lons, lats = [], []

    def crawl(arr):
        if not isinstance(arr, (list, tuple)):
            return
        if arr and isinstance(arr[0], (float, int)) and len(arr) == 2:
            lons.append(float(arr[0])); lats.append(float(arr[1]))
        else:
            for a in arr:
                crawl(a)

    coords = (geom or {}).get("coordinates", [])
    crawl(coords)
    if not lons or not lats:
        return (-90.0, -56.0, -30.0, 15.0)  # broad LATAM default
    return (min(lons), min(lats), max(lons), max(lats))

BBOX_BY_ID: Dict[str, Tuple[float, float, float, float]] = {}
for fid, feat in FEATURE_BY_ID.items():
    bbox = feat.get("bbox")
    if bbox and len(bbox) == 4:
        lonW, latS, lonE, latN = bbox
    else:
        lonW, latS, lonE, latN = _compute_bbox_from_geometry(feat.get("geometry"))
    BBOX_BY_ID[fid] = (float(lonW), float(latS), float(lonE), float(latN))

# Aggregate per country
COUNTRY_BBOX: Dict[str, Tuple[float, float, float, float]] = {}
for iso3, fids in COUNTRY_FEATURE_IDS.items():
    boxes = [BBOX_BY_ID[f] for f in fids if f in BBOX_BY_ID]
    if boxes:
        lonW = min(b[0] for b in boxes)
        latS = min(b[1] for b in boxes)
        lonE = max(b[2] for b in boxes)
        latN = max(b[3] for b in boxes)
        COUNTRY_BBOX[iso3] = (lonW, latS, lonE, latN)

# =============================================================================
# 3) UI OPTIONS
# =============================================================================
# Indicators
ind_map = (panel[["indicator_label", "indicator_col"]]
           .drop_duplicates()
           .sort_values("indicator_label"))
INDICATOR_OPTIONS = [{"label": lbl, "value": lbl} for lbl in ind_map["indicator_label"].tolist()]
INDICATOR_TO_COL = dict(zip(ind_map["indicator_label"], ind_map["indicator_col"]))

# Periods (ordered if categorical)
if pd.api.types.is_categorical_dtype(panel["periodo"]) and panel["periodo"].cat.ordered:
    PERIODOS = panel["periodo"].cat.categories.tolist()
else:
    PERIODOS = sorted(panel["periodo"].astype(str).unique().tolist())
PERIOD_TO_IDX = {p: i for i, p in enumerate(PERIODOS, start=1)}
IDX_TO_PERIOD = {i: p for p, i in PERIOD_TO_IDX.items()}

# Countries
countries = (panel[["country_code", "country_name"]]
             .drop_duplicates()
             .sort_values("country_name"))
COUNTRY_OPTIONS = [{"label": n, "value": c} for c, n in zip(countries["country_code"], countries["country_name"])]
COUNTRY_NAME_BY_CODE = dict(zip(countries["country_code"], countries["country_name"]))

# Agricultural definitions present (clean labels; remove variable codes in parentheses)
def _clean_agri_label(s: str) -> str:
    if not isinstance(s, str):
        return s
    s = re.sub(r"\s*\([^)]+\)\s*$", "", s).strip()
    s = s.replace("Algún", "Algún").replace("agricultura", "agricultura").replace("autoempleado", "autoempleado")
    return s

agri_defs_raw = (panel.loc[panel["agri_def"].ne("N/A"), "agri_def"]
                 .dropna().drop_duplicates().sort_values().astype(str).tolist())
AGRI_OPTIONS = [{"label": _clean_agri_label(x), "value": x} for x in agri_defs_raw] if agri_defs_raw else []

# =============================================================================
# 4) SMALL HELPERS
# =============================================================================
def _subset_ui_text(subset_key: str, agri_def: str) -> str:
    if subset_key == "Todos":
        return ""
    if "Agrícola" in subset_key and agri_def and agri_def != "N/A":
        return f"{subset_key} — {_clean_agri_label(agri_def)}"
    return subset_key

def resolve_subset(filtros, agri_def_label):
    filtros = filtros or []
    rural = "RURAL" in filtros
    indig = "INDIGENA" in filtros
    agri  = "AGRI" in filtros
    if not agri:
        if rural and not indig:   return "Rural", "N/A"
        if indig and not rural:   return "Indígena", "N/A"
        if rural and indig:       return "Rural & Indígena", "N/A"
        return "Todos", "N/A"
    else:
        agdef = agri_def_label if agri_def_label else "N/A"
        if rural and indig:       return "Rural & Indígena & Agrícola", agdef
        if rural:                 return "Rural & Agrícola", agdef
        if indig:                 return "Indígena & Agrícola", agdef
        return "Agrícola", agdef

def _robust_min_max(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return 0.0, 1.0
    lo = float(np.nanquantile(s, 0.02))
    hi = float(np.nanquantile(s, 0.98))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = float(np.nanmin(s)), float(np.nanmax(s))
    if lo == hi:
        lo, hi = max(0.0, lo - 0.1), min(100.0, hi + 0.1)
    return lo, hi

def _center_zoom_from_bbox(lonW, latS, lonE, latN):
    center = {"lat": (latS + latN) / 2.0, "lon": (lonW + lonE) / 2.0}
    span = max(latN - latS, lonE - lonW)
    zoom = 3.0 if span > 60 else 4.0 if span > 30 else 5.0 if span > 15 else 6.0
    return center, zoom

def _bbox_from_features(selected_feature_ids):
    if not selected_feature_ids:
        return {"lat": -15.0, "lon": -60.0}, 3.0
    boxes = [BBOX_BY_ID.get(fid) for fid in selected_feature_ids if fid in BBOX_BY_ID]
    if not boxes:
        return {"lat": -15.0, "lon": -60.0}, 3.0
    lonW = min(b[0] for b in boxes); latS = min(b[1] for b in boxes)
    lonE = max(b[2] for b in boxes); latN = max(b[3] for b in boxes)
    return _center_zoom_from_bbox(lonW, latS, lonE, latN)

@lru_cache(maxsize=256)
def _subset_geojson_cached(ids_tuple: tuple):
    feats = [FEATURE_BY_ID[i] for i in ids_tuple if i in FEATURE_BY_ID]
    return {"type": "FeatureCollection", "features": feats}

def _feature_ids_for_scope(df_map: pd.DataFrame, escala: str, pais_iso3: str | None):
    if escala == "PAIS" and pais_iso3:
        return sorted(list(COUNTRY_FEATURE_IDS.get(pais_iso3, set())))
    return sorted(df_map["feature_id"].dropna().astype(str).unique().tolist())

# =============================================================================
# 4.1) Preindexed slices & cached national layers
# =============================================================================
_panel_reg = panel[panel["region_code"].astype("int64") > 0]
REG_INDEX: Dict[tuple, np.ndarray] = {}
for key, df_g in _panel_reg.groupby(["periodo", "subset_key", "agri_def", "indicator_col"], observed=True):
    REG_INDEX[key] = df_g.index.values.astype(np.int64)

REGIONS_BY_PERIOD: Dict[str, pd.DataFrame] = {}
for per, df_g in _panel_reg.groupby("periodo", observed=True):
    REGIONS_BY_PERIOD[str(per)] = df_g[[
        "country_code", "country_name", "region_code", "region_name", "feature_id"
    ]].drop_duplicates().copy()

@lru_cache(maxsize=2048)
def _regional_slice(periodo_sel: str, subset_key: str, agri_def: str, indicator_col: str):
    key = (periodo_sel, subset_key, agri_def, indicator_col)
    idx = REG_INDEX.get(key)
    if idx is None or len(idx) == 0:
        return pd.DataFrame(columns=list(panel.columns))
    return panel.loc[idx].copy()

@lru_cache(maxsize=2048)
def _national_layer_all_countries(periodo_sel: str, subset_key: str, agri_def: str, indicator_col: str):
    nat = panel[
        (panel["periodo"] == periodo_sel) &
        (panel["subset_key"] == subset_key) &
        (panel["agri_def"] == agri_def) &
        (panel["indicator_col"] == indicator_col) &
        (panel["region_code"].astype("int64") == 0)
    ][["country_code", "indicator_label", "value_pct", "se_pp", "ci95_lo", "ci95_hi", "sample_n"]].copy()

    if nat.empty:
        return nat

    regions = REGIONS_BY_PERIOD.get(str(periodo_sel))
    if regions is None or regions.empty:
        return pd.DataFrame(columns=list(panel.columns))

    regions = regions.copy()
    cols = ["value_pct", "se_pp", "ci95_lo", "ci95_hi", "sample_n"]
    for c in cols:
        mapping = dict(zip(nat["country_code"], nat[c]))
        regions[c] = regions["country_code"].map(mapping).astype("float32")

    regions["indicator_col"] = indicator_col
    regions["indicator_label"] = nat["indicator_label"].iloc[0]
    regions["subset_key"] = subset_key
    regions["agri_def"] = agri_def
    regions["periodo"] = periodo_sel
    return regions

def build_national_layer(periodo_sel, subset_key, agri_def, indicator_col, country_filter=None):
    df = _national_layer_all_countries(periodo_sel, subset_key, agri_def, indicator_col)
    if df is None or df.empty:
        return df
    if country_filter:
        df = df[df["country_code"] == country_filter]
    return df

# =============================================================================
# 5) APP INIT & CSS
# =============================================================================
app = dash.Dash(__name__, title="Mapa de Pobreza Territorial LATAM")

from flask import send_from_directory

@app.server.route("/geo/latam_regiones_simplified.geojson")
def _serve_geojson():
    return send_from_directory(str(BASE_DIR), GEOJSON_FILE)

GRAPH_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d", "toggleSpikelines",
        "hoverClosestCartesian", "hoverCompareCartesian"
    ],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}

MAP_STYLE = "carto-positron"  # works with MapLibre and Mapbox fallback

# Minimal embedded CSS
app.index_string = """
<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      :root{
        --brand:#004085; --brand-soft:#e7f3ff; --ink:#333; --muted:#6c757d;
        --card:#f0f8ff; --border:#cce5ff; --shadow:0 4px 18px rgba(0,0,0,0.08);
        --panel-bg:#fff; --panel-header:#e9f5ff;
      }
      html, body { height: 100%; }
      body{
        font-family:"Segoe UI","Roboto",Arial,sans-serif;
        background:#f8f9fa; margin:0;
      }
      .muted{color:var(--muted);}
      .section-card{
        background:var(--card); border:1px solid var(--border); border-radius:12px;
        padding:12px 12px; box-shadow:var(--shadow); margin:12px auto;
        max-width: 100%; overflow: visible;
      }
      .floating-controls{
        position:absolute; top:16px; left:8px; width:360px; background:var(--panel-bg);
        border:1px solid #e9ecef; border-radius:12px; box-shadow:var(--shadow);
        overflow:visible !important; z-index:2000; transition:width .2s ease;
      }
      .floating-controls.icon-only{ width:46px; overflow:hidden !important; }
      .control-panel-header{
        display:flex; align-items:center; gap:10px; padding:12px 14px;
        background:var(--panel-header); border-bottom:1px solid #d6ebff; cursor:pointer;
      }
      .control-panel-header .header-icon{font-size:18px;}
      .control-panel-header .header-text{font-weight:700;color:var(--brand);font-size:15px;}
      .toggle-btn{
        margin-left:auto; border:1px solid var(--brand); background:var(--brand-soft);
        color:var(--brand); border-radius:8px; font-size:12px; padding:4px 10px; cursor:pointer;
      }
      .controls-content{
        padding:14px; max-height:74vh; overflow-y:auto;
      }
      .controls-content.hidden{display:none;}
      .controls-content > div { margin-bottom: 16px; }
      .controls-content label { display:block; font-weight:700; font-size:14px; margin-bottom:6px; }
      .lifted-dropdown .Select-menu-outer, .lifted-dropdown .Select__menu{
        z-index:3000 !important;
      }
      .title-wrap{
        position:absolute; top:14px; left:50%; transform:translateX(-50%);
        z-index:1200; text-align:center; pointer-events:none;
      }
      .title-chip{
        background:rgba(255,255,255,.95); padding:8px 20px; border-radius:20px;
        font-size:14px; font-weight:700; color:#004085; box-shadow:0 2px 8px rgba(0,0,0,.12);
        border:1px solid rgba(0,64,133,.2); display:inline-block;
      }
      .title-spinner{ margin-top: 10px; }
      .title-spinner .dash-spinner{ transform:scale(.8); opacity:.85; }
      @media (max-width:900px){
        .floating-controls{ top:56px; left:6px; width:86vw; }
        .floating-controls.icon-only{ width:44px; }
      }
      .hero{ margin-bottom:28px; }
      .hero h1{ text-align:center; color:var(--brand); margin:8px 0 6px; font-size:28px; line-height:1.15;}
      .hero p { text-align:center; margin:0; font-size:16px; line-height:1.35;}
      /* Fullscreen graph height */
      #map-container { height: calc(100vh - 150px); width: 100%; }
      .controls-content{ padding-bottom: 280px; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>
      {%config%}
      {%scripts%}
      {%renderer%}
    </footer>
  </body>
</html>
"""

server = app.server

# =============================================================================
# 6) LAYOUT
# =============================================================================
def _period_marks():
    return {PERIOD_TO_IDX[p]: p for p in PERIODOS}

app.layout = html.Div(
    style={"padding": "10px"},
    children=[
        html.Div(
            className="hero",
            children=[
                html.H1("Mapa de Pobreza Territorial en América Latina"),
                html.P("Explora indicadores por región, periodo y subpoblaciones."),
                html.P("Activa 'Escala por país' para identificar hotspots dentro de un país.", className="muted"),
            ],
        ),
        html.Div(
            className="section-card",
            children=[
                html.Div(
                    id="map-container",
                    style={"position": "relative", "width": "100%", "overflow": "visible"},
                    children=[
                        html.Div(
                            id={"type": "floating-panel-wrapper", "index": "map"},
                            className="floating-controls",
                            children=[
                                html.Div(
                                    id={"type": "panel-header", "index": "map"},
                                    className="control-panel-header",
                                    n_clicks=0,
                                    children=[
                                        html.Span("⚙️", className="header-icon"),
                                        html.Span("Controles", className="header-text"),
                                        html.Button("−", id="toggle-controls-btn", className="toggle-btn"),
                                    ],
                                ),
                                html.Div(
                                    id={"type": "panel-content", "index": "map"},
                                    className="controls-content",
                                    children=[
                                        html.Div([
                                            html.Label("📊 Variable de pobreza"),
                                            dcc.Dropdown(
                                                id="dd-indicador",
                                                className="lifted-dropdown",
                                                options=INDICATOR_OPTIONS,
                                                value=INDICATOR_OPTIONS[0]["value"] if INDICATOR_OPTIONS else None,
                                                clearable=False,
                                            ),
                                        ]),
                                        html.Div([
                                            html.Label("📅 Periodo"),
                                            dcc.Slider(
                                                id="sl-periodo",
                                                min=1, max=len(PERIODOS), step=None,
                                                value=len(PERIODOS),
                                                marks=_period_marks(),
                                                included=False,
                                                updatemode="mouseup",
                                            ),
                                        ]),
                                        html.Div([
                                            html.Label("🎯 Filtros de población"),
                                            dcc.Checklist(
                                                id="chk-filtros",
                                                options=[
                                                    {"label": " Rural", "value": "RURAL"},
                                                    {"label": " Indígena", "value": "INDIGENA"},
                                                    {"label": " Agrícola", "value": "AGRI"},
                                                ],
                                                value=[],
                                                labelStyle={"display": "block", "fontSize": "13px", "marginBottom": "8px"},
                                            ),
                                            dcc.Dropdown(
                                                id="dd-agri-def",
                                                className="lifted-dropdown",
                                                options=AGRI_OPTIONS,
                                                value=AGRI_OPTIONS[0]["value"] if AGRI_OPTIONS else None,
                                                clearable=False,
                                                disabled=True,
                                                placeholder="Definición de 'agrícola'",
                                            ),
                                        ]),
                                        html.Div([
                                            html.Label("🏷️ Nivel / Escala"),
                                            dcc.RadioItems(
                                                id="radio-nivel",
                                                options=[
                                                    {"label": " Regional", "value": "REG"},
                                                    {"label": " Nacional (color por país)", "value": "NAT"},
                                                ],
                                                value="REG",
                                                inline=True,
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dcc.RadioItems(
                                                id="radio-escala",
                                                options=[
                                                    {"label": " Continente", "value": "GLOBAL"},
                                                    {"label": " Por país", "value": "PAIS"},
                                                ],
                                                value="GLOBAL",
                                                inline=True,
                                            ),
                                            html.Div(style={"height": "6px"}),
                                            dcc.Dropdown(
                                                id="dd-pais",
                                                className="lifted-dropdown",
                                                options=COUNTRY_OPTIONS,
                                                value=None,
                                                placeholder="País (si 'Por país')",
                                                disabled=True,
                                            ),
                                        ]),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            className="title-wrap",
                            children=[
                                html.Div(id="map-title", className="title-chip"),
                                html.Div(
                                    className="title-spinner",
                                    children=dcc.Loading(
                                        id="map-loading",
                                        type="dot",
                                        children=html.Div(id="map-loading-sentinel", style={"width": 1, "height": 1}),
                                    ),
                                ),
                            ],
                        ),
                        dcc.Graph(
                            id="mapa",
                            style={"height": "100%", "width": "100%"},
                            config=GRAPH_CONFIG
                        ),
                    ],
                )
            ],
        )
    ],
)

# =============================================================================
# 7) CONTROLS CALLBACKS
# =============================================================================
@app.callback(
    [Output({"type": "floating-panel-wrapper", "index": MATCH}, "className"),
     Output({"type": "panel-content", "index": MATCH}, "className")],
    Input({"type": "panel-header", "index": MATCH}, "n_clicks"),
    State({"type": "floating-panel-wrapper", "index": MATCH}, "className"),
    prevent_initial_call=True,
)
def toggle_panel_animation(n, current_class):
    if n and n > 0:
        if current_class and "icon-only" in current_class:
            return "floating-controls", "controls-content"
        else:
            return "floating-controls icon-only", "controls-content hidden"
    return no_update, no_update

@app.callback(Output("dd-agri-def", "disabled"), Input("chk-filtros", "value"))
def toggle_agri_def(filtros):
    filtros = filtros or []
    return ("AGRI" not in filtros) or (len(AGRI_OPTIONS) == 0)

@app.callback(Output("dd-pais", "disabled"), Input("radio-escala", "value"))
def toggle_country_dropdown(modo):
    return (modo != "PAIS")

# =============================================================================
# 8) MAP BUILDERS (MapLibre-first with Mapbox fallback)
# =============================================================================
def _make_choropleth(df_map, zmin, zmax, center, zoom, indicador_label):
    kwargs_common = dict(
        geojson=GEOJSON_ALL,
        locations="feature_id",
        featureidkey="properties.id",
        color="value_pct",
        color_continuous_scale="YlOrRd",
        range_color=(zmin, zmax),
        center=center,
        zoom=zoom,
        opacity=0.92,
        hover_name="region_name",
    )
    try:
        # Plotly >=5.24 (MapLibre)
        fig = px.choropleth_map(
            df_map,
            map_style=MAP_STYLE,
            **kwargs_common
        )
        return fig
    except Exception:
        # Fallback to Mapbox traces if MapLibre API not present
        warnings.filterwarnings("ignore", message=".*choropleth_mapbox.*", category=DeprecationWarning)
        fig = px.choropleth_mapbox(
            df_map,
            mapbox_style=MAP_STYLE,
            **kwargs_common
        )
        return fig

# =============================================================================
# 9) MAP CALLBACK
# =============================================================================
@app.callback(
    [Output("mapa", "figure"),
     Output("map-title", "children"),
     Output("map-loading-sentinel", "children")],
    [
        Input("dd-indicador", "value"),
        Input("sl-periodo", "value"),
        Input("chk-filtros", "value"),
        Input("dd-agri-def", "value"),
        Input("radio-nivel", "value"),
        Input("radio-escala", "value"),
        Input("dd-pais", "value"),
    ],
    [State("mapa", "relayoutData"), State("mapa", "figure")]
)
def update_map(indicador_label, periodo_idx, filtros, agri_def_label, nivel, escala, pais_iso3, relayout_data, prev_fig):
    if not indicador_label or not periodo_idx:
        return go.Figure(), "Selecciona una variable y un periodo", ""

    indicador_col = INDICATOR_TO_COL.get(indicador_label, None)
    periodo_sel = IDX_TO_PERIOD.get(periodo_idx, PERIODOS[-1])
    subset_key, agri_def = resolve_subset(filtros, agri_def_label)

    # Build data slice
    if nivel == "NAT":
        df_map = build_national_layer(periodo_sel, subset_key, agri_def, indicador_col,
                                      country_filter=pais_iso3 if escala == "PAIS" else None)
    else:
        df_map = _regional_slice(periodo_sel, subset_key, agri_def, indicador_col)
        if escala == "PAIS" and pais_iso3 and not df_map.empty:
            df_map = df_map[df_map["country_code"] == pais_iso3]

    if df_map is None or df_map.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos para esta combinación. Ajusta filtros.",
                           showarrow=False, y=0.5, x=0.5)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        return fig, f"{indicador_label} · {('Nacional' if nivel=='NAT' else 'Regional')} · {periodo_sel}", ""

    # Color scale (robust)
    zmin, zmax = _robust_min_max(df_map["value_pct"])

    # GeoJSON subset & order
    feature_ids = _feature_ids_for_scope(df_map, escala, pais_iso3)
    if feature_ids:
        df_map = df_map.set_index("feature_id").reindex(feature_ids).reset_index()

    # Center/zoom heuristic
    if escala == "PAIS" and pais_iso3 and pais_iso3 in COUNTRY_BBOX:
        center, zoom = _center_zoom_from_bbox(*COUNTRY_BBOX[pais_iso3])
    else:
        center, zoom = _bbox_from_features(feature_ids)

    # Preserve user pan/zoom across updates (MapLibre uses layout.map.*, Mapbox uses layout.mapbox.*)
    if relayout_data:
        if "map.center" in relayout_data:
            zoom = relayout_data.get("map.zoom", zoom)
            center = relayout_data.get("map.center", center)
        elif "mapbox.center" in relayout_data:
            zoom = relayout_data.get("mapbox.zoom", zoom)
            center = relayout_data.get("mapbox.center", center)

    # Build arrays and dynamic hover (lists → lighter JSON; play nice with Patch)
    z = df_map["value_pct"].astype(np.float32).tolist()
    customdata = np.stack([
        df_map["country_name"].astype(str).values,
        df_map["region_name"].astype(str).values,
        df_map["ci95_lo"].astype(np.float32).values,
        df_map["ci95_hi"].astype(np.float32).values,
        df_map["se_pp"].astype(np.float32).values,
        df_map["sample_n"].astype(np.float32).values,
    ], axis=-1).tolist()

    subpop_txt = _subset_ui_text(subset_key, agri_def)
    hover_lines = ["<b>%{customdata[0]}</b> – %{customdata[1]}<br>"]
    if subpop_txt:
        hover_lines.append(f"<i>Subpoblación: {subpop_txt}</i><br>")
    hover_lines += [
        f"{indicador_label}: " + "%{z:.1f}%<br>",
        "IC95%: [%{customdata[2]:.1f} – %{customdata[3]:.1f}]<br>",
        "EE: %{customdata[4]:.1f} p.p.<br>",
        "n: %{customdata[5]:,.0f}",
    ]
    hovertemplate = "".join(hover_lines) + "<extra></extra>"

    # Patch fast path if polygons didn't change
    same_scope = False
    if isinstance(prev_fig, dict) and prev_fig.get("data"):
        prev_locs = prev_fig["data"][0].get("locations")
        same_scope = list(prev_locs) == feature_ids

    if HAVE_PATCH and same_scope:
        patch = Patch()
        patch["data"][0]["z"] = z
        patch["data"][0]["customdata"] = customdata
        patch["data"][0]["hovertemplate"] = hovertemplate
        patch["layout"]["coloraxis"]["cmin"] = zmin
        patch["layout"]["coloraxis"]["cmax"] = zmax

        nivel_txt = "Nacional" if nivel == "NAT" else "Regional"
        scope_title = COUNTRY_NAME_BY_CODE.get(pais_iso3, "América Latina") if (escala == "PAIS" and pais_iso3) else "América Latina"
        map_title = f"{indicador_label} · {nivel_txt} · {scope_title} · Periodo {periodo_sel}"
        if subpop_txt:
            map_title += f" · Subpoblación: {subpop_txt}"
        return patch, map_title, ""

    # Full rebuild
    fig = _make_choropleth(df_map, zmin, zmax, center, zoom, indicador_label)
    fig.update_traces(hovertemplate=hovertemplate, customdata=customdata)

    nivel_txt = "Nacional" if nivel == "NAT" else "Regional"
    scope_title = COUNTRY_NAME_BY_CODE.get(pais_iso3, "América Latina") if (escala == "PAIS" and pais_iso3) else "América Latina"
    map_title = f"{indicador_label} · {nivel_txt} · {scope_title} · Periodo {periodo_sel}"
    if subpop_txt:
        map_title += f" · Subpoblación: {subpop_txt}"

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="%"),
        uirevision="keep",
    )
    return fig, map_title, ""

# =============================================================================
# 10) RUN
# =============================================================================
if __name__ == "__main__":
    is_render = bool(os.getenv("RENDER") or os.getenv("PORT"))
    host = "0.0.0.0" if is_render else "127.0.0.1"
    port = int(os.getenv("PORT", "8050"))
    debug_flag = os.getenv("DASH_DEBUG", "1") == "1"
    print(f"→ Open http://localhost:{port}")
    app.run(host=host, port=port, debug=debug_flag)
