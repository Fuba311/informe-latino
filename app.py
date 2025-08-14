# -*- coding: utf-8 -*-
"""
LATAM Territorial Poverty Map — Dash app (partition-aware, Render-ready).

It will look for your preprocessed outputs as follows:

1) Prefer a local Hive-partitioned dataset written by your script:
     - ./panel_territorial_ds/
     - OR any ./panel_territorial_ds_* (timestamped) folder (chooses the best)
   (These match how your pre-process saves partitions: periodo=/country_code=/)

2) If no dataset dir is found:
     - Use a monolithic ./panel_territorial.parquet (if present)
     - Otherwise, download files from GitHub raw (configurable)
     - Or download and unzip a dataset ZIP if DATASET_ZIP_URL is provided.

Env vars (all optional):
  DATASET_DIR        Absolute or relative path to a dataset directory
  DATASET_ZIP_URL    HTTPS URL to a ZIP that contains panel_territorial_ds/...
  DATA_REPO          default: Fuba311/informe-latino
  DATA_BRANCH        default: main
  PANEL_URL          direct URL to panel_territorial.parquet
  GEOJSON_URL        direct URL to latam_regiones_simplified.geojson
  MAP_STYLE          default: carto-positron
  DASH_DEBUG         0/1 (default 1 locally)

Run:
  pip install -r requirements.txt
  python app.py
On Render:
  gunicorn app:server --workers 2 --threads 8 --timeout 120
"""

import os
import re
import io
import json
import zipfile
import warnings
from pathlib import Path
from functools import lru_cache
from typing import Tuple, Dict, List, Optional

import requests
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State, MATCH
try:
    from dash import Patch
    HAVE_PATCH = True
except Exception:
    HAVE_PATCH = False

# Attempt pyarrow.dataset for fast partitioned reads
try:
    import pyarrow.dataset as ds
    HAVE_DATASET = True
except Exception:
    ds = None  # type: ignore
    HAVE_DATASET = False
import re
CLEAN_PARENS_RE = re.compile(r"\s*\([^)]+\)\s*$")


print("--- STARTING LATAM POVERTY DASHBOARD ---")

# =============================================================================
# 1) PATHS / FETCH HELPERS
# =============================================================================
BASE_DIR = Path(__file__).parent.resolve()

NO_DATA_COLOR = "#B0B0B0"  # grey used to paint polygons with no data for the selected period
NO_DATA_LEGEND_LABEL = "Sin datos (gris)"

DATA_REPO = os.getenv("DATA_REPO", "Fuba311/informe-latino")
DATA_BRANCH = os.getenv("DATA_BRANCH", "main")

PANEL_FILE = "panel_territorial.parquet"
GEOJSON_FILE = "latam_regiones_simplified.geojson"
DATASET_DIR_NAME = "panel_territorial_ds"

def _raw_url(owner_repo: str, branch: str, filename: str) -> str:
    return f"https://raw.githubusercontent.com/{owner_repo}/{branch}/{filename}"

PANEL_URL = os.getenv("PANEL_URL") or _raw_url(DATA_REPO, DATA_BRANCH, PANEL_FILE)
GEOJSON_URL = os.getenv("GEOJSON_URL") or _raw_url(DATA_REPO, DATA_BRANCH, GEOJSON_FILE)
DATASET_ZIP_URL = os.getenv("DATASET_ZIP_URL")  # optional
DATASET_DIR_ENV = os.getenv("DATASET_DIR")      # optional

PANEL_PATH = BASE_DIR / PANEL_FILE
GEOJSON_PATH = BASE_DIR / GEOJSON_FILE

def download_if_needed(url: str, dest: Path, timeout: int = 180, chunk: int = 1 << 20) -> Path:
    """Idempotent streaming download with atomic write."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    print(f"↓ Downloading {url} → {dest.name}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for part in r.iter_content(chunk_size=chunk):
                if part:
                    f.write(part)
    os.replace(tmp, dest)
    return dest

def maybe_download_and_unzip_dataset(zip_url: str, out_dir: Path) -> None:
    """Download a dataset ZIP and unzip into out_dir (idempotent)."""
    if out_dir.exists() and any(out_dir.rglob("*.parquet")):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"↓ Downloading dataset ZIP from {zip_url}")
    with requests.get(zip_url, stream=True, timeout=600) as r:
        r.raise_for_status()
        buf = io.BytesIO()
        for chunk in r.iter_content(chunk_size=1 << 20):
            if chunk:
                buf.write(chunk)
        buf.seek(0)
    with zipfile.ZipFile(buf) as zf:
        names = zf.namelist()
        root_prefix = DATASET_DIR_NAME + "/"
        if any(n.startswith(root_prefix) for n in names):
            # Strip the root prefix to land exactly at out_dir/
            for n in names:
                if n.endswith("/") or not n.startswith(root_prefix):
                    continue
                rel = Path(n[len(root_prefix):])
                target = out_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(n) as src, open(target, "wb") as dst:
                    dst.write(src.read())
        else:
            zf.extractall(out_dir)
    print(f"✓ Unzipped dataset into {out_dir}")

def discover_dataset_dir(base_dir: Path) -> Optional[Path]:
    """Find a dataset directory matching your pre-process naming:
       - panel_territorial_ds/
       - panel_territorial_ds_YYYYmmdd_HHMMSS/
       Prefer exact name; else choose the one with most parquet files.
    """
    # 1) If env var points somewhere valid, use it
    if DATASET_DIR_ENV:
        p = Path(DATASET_DIR_ENV)
        if not p.is_absolute():
            p = base_dir / p
        if p.exists() and any(p.rglob("*.parquet")):
            print(f"✓ Using dataset from DATASET_DIR={p}")
            return p

    # 2) Exact name in base dir
    exact = base_dir / DATASET_DIR_NAME
    if exact.exists() and any(exact.rglob("*.parquet")):
        return exact

    # 3) Any timestamped siblings
    candidates = []
    for d in base_dir.glob(DATASET_DIR_NAME + "*"):
        if d.is_dir() and any(d.rglob("*.parquet")):
            # Score by number of parquet files and mtime
            try:
                nfiles = sum(1 for _ in d.rglob("*.parquet"))
            except Exception:
                nfiles = 0
            mtime = d.stat().st_mtime
            candidates.append((nfiles, mtime, d))
    if candidates:
        candidates.sort(reverse=True)  # most files, then newest
        best = candidates[0][2]
        print(f"✓ Using discovered dataset dir: {best}")
        return best

    return None

# Optionally download and unzip dataset into the canonical folder
dataset_dir = BASE_DIR / DATASET_DIR_NAME
if DATASET_ZIP_URL:
    try:
        maybe_download_and_unzip_dataset(DATASET_ZIP_URL, dataset_dir)
    except Exception as e:
        print(f"⚠ Failed to download/unzip dataset ZIP: {e}")

# Always ensure GeoJSON exists (small)
try:
    download_if_needed(GEOJSON_URL, GEOJSON_PATH)
except Exception as e:
    print(f"⚠ GeoJSON download failed: {e}")

# If neither dataset nor parquet exists, try to fetch parquet
DISCOVERED_DATASET_DIR = discover_dataset_dir(BASE_DIR)
if DISCOVERED_DATASET_DIR is None and not (BASE_DIR / PANEL_FILE).exists():
    try:
        download_if_needed(PANEL_URL, PANEL_PATH)
    except Exception as e:
        print(f"⚠ Parquet download failed: {e}")

# =============================================================================
# 2) LOAD GEOJSON & PRECOMPUTE BBOX
# =============================================================================
with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
    GEOJSON_ALL = json.load(f)

FEATURES: List[dict] = GEOJSON_ALL.get("features", [])
FEATURE_BY_ID: Dict[str, dict] = {feat.get("properties", {}).get("id"): feat for feat in FEATURES}

COUNTRY_FEATURE_IDS: Dict[str, set] = {}
for feat in FEATURES:
    fid = feat.get("properties", {}).get("id")
    if not fid or "|" not in fid:
        continue
    iso3 = fid.split("|", 1)[0]
    COUNTRY_FEATURE_IDS.setdefault(iso3, set()).add(fid)

def _compute_bbox_from_geometry(geom) -> Tuple[float, float, float, float]:
    lons, lats = [], []
    def crawl(arr):
        if not isinstance(arr, (list, tuple)): return
        if arr and isinstance(arr[0], (float, int)) and len(arr) == 2:
            lons.append(float(arr[0])); lats.append(float(arr[1]))
        else:
            for a in arr: crawl(a)
    coords = (geom or {}).get("coordinates", [])
    crawl(coords)
    if not lons or not lats:
        return (-90.0, -56.0, -30.0, 15.0)
    return (min(lons), min(lats), max(lons), max(lats))

BBOX_BY_ID: Dict[str, Tuple[float, float, float, float]] = {}
for fid, feat in FEATURE_BY_ID.items():
    bbox = feat.get("bbox")
    if bbox and len(bbox) == 4:
        lonW, latS, lonE, latN = bbox
    else:
        lonW, latS, lonE, latN = _compute_bbox_from_geometry(feat.get("geometry"))
    BBOX_BY_ID[fid] = (float(lonW), float(latS), float(lonE), float(latN))

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
# 3) DATA BACKEND (partitioned fast path; monolithic fallback)
# =============================================================================
USE_DATASET = HAVE_DATASET and (DISCOVERED_DATASET_DIR is not None)

# Columns used in scans
BASE_COLS = [
    "country_code","country_name","periodo",
    "region_code","region_name","feature_id",
    "subset_key","agri_def",
    "indicator_label","indicator_col",
    "value_pct","se_pp","ci95_lo","ci95_hi","sample_n",
]

if USE_DATASET:
    DS = ds.dataset(str(DISCOVERED_DATASET_DIR), format="parquet", partitioning="hive")
    print(f"ℹ Using partitioned dataset at {DISCOVERED_DATASET_DIR}")

    def _scan(cols: List[str], filt=None) -> pd.DataFrame:
        sc = DS.scanner(columns=cols, filter=filt)
        tbl = sc.to_table()
        df = tbl.to_pandas(types_mapper=pd.ArrowDtype)
        # Normalize dtypes
        for c in ("value_pct","se_pp","ci95_lo","ci95_hi","sample_n"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
        for c in ("country_code","country_name","periodo","subset_key","agri_def",
                  "indicator_label","indicator_col","region_name","feature_id"):
            if c in df.columns:
                df[c] = df[c].astype("string")
        if "region_code" in df.columns:
            df["region_code"] = pd.to_numeric(df["region_code"], errors="coerce").astype("Int64")
        return df

    # Enumerations (cached)
    @lru_cache(maxsize=1)
    def _periods() -> List[str]:
        df = _scan(["periodo"])
        cats = sorted(df["periodo"].dropna().astype(str).unique().tolist())
        pref = ["2015-2016","2017-2018","2019","2020-2021","2022-2023"]
        return [p for p in pref if p in cats] + [p for p in cats if p not in pref]

    PERIODOS = _periods()
    PERIOD_TO_IDX = {p: i for i, p in enumerate(PERIODOS, start=1)}
    IDX_TO_PERIOD = {i: p for p, i in PERIOD_TO_IDX.items()}

    @lru_cache(maxsize=1)
    def _indicators():
        df = _scan(["indicator_label","indicator_col"]).drop_duplicates()
        df = df.sort_values("indicator_label")
        options = [{"label": r["indicator_label"], "value": r["indicator_label"]} for _, r in df.iterrows()]
        mapping = dict(zip(df["indicator_label"], df["indicator_col"]))
        return options, mapping

    INDICATOR_OPTIONS, INDICATOR_TO_COL = _indicators()

    @lru_cache(maxsize=1)
    def _countries():
        df = _scan(["country_code","country_name"]).drop_duplicates()
        df = df.sort_values("country_name")
        options = [{"label": n, "value": c} for c, n in zip(df["country_code"], df["country_name"])]
        name_by_code = dict(zip(df["country_code"], df["country_name"]))
        return options, name_by_code

    COUNTRY_OPTIONS, COUNTRY_NAME_BY_CODE = _countries()

    @lru_cache(maxsize=1)
    def _agri_options():
        df = _scan(["agri_def"]).dropna().drop_duplicates()
        df = df[df["agri_def"].astype(str) != "N/A"]
        vals = sorted(df["agri_def"].astype(str).tolist())
        def _clean(s: str) -> str:
            s = re.sub(r"\s*\([^)]+\)\s*$", "", s).strip()
            return s.replace("Algún","Algún").replace("agricultura","agricultura").replace("autoempleado","autoempleado")
        return [{"label": _clean(v), "value": v} for v in vals]
    AGRI_OPTIONS = _agri_options()

    @lru_cache(maxsize=16)
    def _regions_skeleton(periodo_sel: str) -> pd.DataFrame:
        f = (ds.field("periodo") == periodo_sel) & (ds.field("region_code") > 0)
        cols = ["country_code","country_name","region_code","region_name","feature_id"]
        return _scan(cols, f).drop_duplicates()

    @lru_cache(maxsize=128)
    def _regional_slice(periodo_sel: str, subset_key: str, agri_def: str, indicator_col: str,
                        country_filter: Optional[str]) -> pd.DataFrame:
        f = (
            (ds.field("periodo") == periodo_sel) &
            (ds.field("subset_key") == subset_key) &
            (ds.field("agri_def") == agri_def) &
            (ds.field("indicator_col") == indicator_col) &
            (ds.field("region_code") > 0)
        )
        if country_filter:
            f = f & (ds.field("country_code") == country_filter)
        return _scan(BASE_COLS, f)

    @lru_cache(maxsize=128)
    def _national_layer_all_countries(periodo_sel: str, subset_key: str, agri_def: str, indicator_col: str,
                                      country_filter: Optional[str]) -> pd.DataFrame:
        f = (
            (ds.field("periodo") == periodo_sel) &
            (ds.field("subset_key") == subset_key) &
            (ds.field("agri_def") == agri_def) &
            (ds.field("indicator_col") == indicator_col) &
            (ds.field("region_code") == 0)
        )
        nat = _scan(["country_code","indicator_label","value_pct","se_pp","ci95_lo","ci95_hi","sample_n"], f)
        if nat.empty:
            return nat
        regions = _regions_skeleton(periodo_sel).copy()
        if country_filter:
            regions = regions[regions["country_code"] == country_filter]
        for c in ["value_pct","se_pp","ci95_lo","ci95_hi","sample_n"]:
            regions[c] = regions["country_code"].map(dict(zip(nat["country_code"], nat[c]))).astype("float32")
        regions["indicator_col"] = indicator_col
        regions["indicator_label"] = nat["indicator_label"].iloc[0]
        regions["subset_key"] = subset_key
        regions["agri_def"] = agri_def
        regions["periodo"] = periodo_sel
        return regions

    def build_national_layer(periodo_sel, subset_key, agri_def, indicator_col, country_filter=None):
        return _national_layer_all_countries(periodo_sel, subset_key, agri_def, indicator_col, country_filter)

else:
    # -------------------- Monolithic fallback --------------------
    print("ℹ Using monolithic Parquet (no dataset dir found).")
    if not PANEL_PATH.exists():
        raise FileNotFoundError(f"Missing {PANEL_PATH}. Provide DATASET_ZIP_URL or PANEL_URL or commit the file.")
    panel = pd.read_parquet(PANEL_PATH)

    panel["feature_id"] = panel["feature_id"].astype(str)
    for c in ["value_pct", "se_pp", "ci95_lo", "ci95_hi"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").astype("float32")
    panel["sample_n"] = pd.to_numeric(panel["sample_n"], errors="coerce").astype("float32")

    ind_map = (panel[["indicator_label", "indicator_col"]]
               .drop_duplicates()
               .sort_values("indicator_label"))
    INDICATOR_OPTIONS = [{"label": lbl, "value": lbl} for lbl in ind_map["indicator_label"].tolist()]
    INDICATOR_TO_COL = dict(zip(ind_map["indicator_label"], ind_map["indicator_col"]))

    if pd.api.types.is_categorical_dtype(panel["periodo"]) and panel["periodo"].cat.ordered:
        PERIODOS = panel["periodo"].cat.categories.tolist()
    else:
        PERIODOS = sorted(panel["periodo"].astype(str).unique().tolist())
    PERIOD_TO_IDX = {p: i for i, p in enumerate(PERIODOS, start=1)}
    IDX_TO_PERIOD = {i: p for p, i in PERIOD_TO_IDX.items()}

    countries = (panel[["country_code", "country_name"]]
                 .drop_duplicates()
                 .sort_values("country_name"))
    COUNTRY_OPTIONS = [{"label": n, "value": c} for c, n in zip(countries["country_code"], countries["country_name"])]
    COUNTRY_NAME_BY_CODE = dict(zip(countries["country_code"], countries["country_name"]))

    def _clean_agri_label(s: str) -> str:
        if not isinstance(s, str): return s
        s = re.sub(r"\s*\([^)]+\)\s*$", "", s).strip()
        return s.replace("Algún","Algún").replace("agricultura","agricultura").replace("autoempleado","autoempleado")
    agri_defs_raw = (panel.loc[panel["agri_def"].ne("N/A"), "agri_def"]
                     .dropna().drop_duplicates().sort_values().astype(str).tolist())
    AGRI_OPTIONS = [{"label": _clean_agri_label(x), "value": x} for x in agri_defs_raw] if agri_defs_raw else []

    _panel_reg = panel[panel["region_code"].astype("int64") > 0]
    REG_INDEX: Dict[tuple, np.ndarray] = {}
    for key, df_g in _panel_reg.groupby(["periodo", "subset_key", "agri_def", "indicator_col"], observed=True):
        REG_INDEX[key] = df_g.index.values.astype(np.int64)

    REGIONS_BY_PERIOD: Dict[str, pd.DataFrame] = {}
    for per, df_g in _panel_reg.groupby("periodo", observed=True):
        REGIONS_BY_PERIOD[str(per)] = df_g[[
            "country_code","country_name","region_code","region_name","feature_id"
        ]].drop_duplicates().copy()

    @lru_cache(maxsize=128)
    def _regional_slice(periodo_sel: str, subset_key: str, agri_def: str, indicator_col: str,
                        country_filter: Optional[str]) -> pd.DataFrame:
        idx = REG_INDEX.get((periodo_sel, subset_key, agri_def, indicator_col))
        if idx is None or len(idx) == 0:
            return pd.DataFrame(columns=list(panel.columns))
        df = panel.loc[idx].copy()
        if country_filter:
            df = df[df["country_code"] == country_filter]
        return df

    @lru_cache(maxsize=128)
    def _national_layer_all_countries(periodo_sel: str, subset_key: str, agri_def: str, indicator_col: str,
                                      country_filter: Optional[str]) -> pd.DataFrame:
        nat = panel[
            (panel["periodo"] == periodo_sel) &
            (panel["subset_key"] == subset_key) &
            (panel["agri_def"] == agri_def) &
            (panel["indicator_col"] == indicator_col) &
            (panel["region_code"].astype("int64") == 0)
        ][["country_code","indicator_label","value_pct","se_pp","ci95_lo","ci95_hi","sample_n"]].copy()
        if nat.empty:
            return nat
        regions = REGIONS_BY_PERIOD.get(str(periodo_sel))
        if regions is None or regions.empty:
            return pd.DataFrame(columns=list(panel.columns))
        regions = regions.copy()
        for c in ["value_pct","se_pp","ci95_lo","ci95_hi","sample_n"]:
            mapping = dict(zip(nat["country_code"], nat[c]))
            regions[c] = regions["country_code"].map(mapping).astype("float32")
        regions["indicator_col"] = indicator_col
        regions["indicator_label"] = nat["indicator_label"].iloc[0]
        regions["subset_key"] = subset_key
        regions["agri_def"] = agri_def
        regions["periodo"] = periodo_sel
        if country_filter:
            regions = regions[regions["country_code"] == country_filter]
        return regions

    def build_national_layer(periodo_sel, subset_key, agri_def, indicator_col, country_filter=None):
        return _national_layer_all_countries(periodo_sel, subset_key, agri_def, indicator_col, country_filter)

# =============================================================================
# 4) SHARED HELPERS
# =============================================================================
def _subset_ui_text(subset_key: str, agri_def: str) -> str:
    if subset_key == "Todos":
        return ""
    if "Agrícola" in subset_key and agri_def and agri_def != "N/A":
        clean_agri = CLEAN_PARENS_RE.sub("", str(agri_def)).strip()
        return f"{subset_key} — {clean_agri}"
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

def _feature_ids_for_scope(df_map: pd.DataFrame, escala: str, pais_iso3: Optional[str]):
    """Return the full set of feature ids we want to render for the chosen scope.
    GLOBAL: every polygon we have in the GeoJSON (so 'no data' still shows in grey).
    PAIS: all polygons belonging to that ISO3 in the GeoJSON.
    """
    if escala == "PAIS" and pais_iso3:
        return sorted(list(COUNTRY_FEATURE_IDS.get(pais_iso3, set())))
    # GLOBAL: paint everything we have in the GeoJSON, not just the current slice rows
    return sorted(FEATURE_BY_ID.keys())

def _fill_names_from_geojson(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing region/country names for rows created by reindexing to all feature_ids."""
    if "feature_id" not in df.columns:
        return df
    # Ensure columns exist
    for c in ("region_name", "country_name"):
        if c not in df.columns:
            df[c] = pd.Series([None] * len(df), dtype="string")
    missing = df["region_name"].isna() | (df["region_name"].astype(str) == "nan")
    if missing.any():
        fids = df.loc[missing, "feature_id"].astype(str).tolist()
        reg_names = []
        country_names = []
        for fid in fids:
            props = (FEATURE_BY_ID.get(fid) or {}).get("properties", {})
            # Try common property names; fall back to id suffix
            rname = props.get("region_name") or props.get("name") or fid.split("|", 1)[-1]
            iso3 = fid.split("|", 1)[0] if "|" in fid else (props.get("iso3") or "")
            cname = COUNTRY_NAME_BY_CODE.get(iso3) or props.get("country_name") or iso3 or "—"
            reg_names.append(str(rname))
            country_names.append(str(cname))
        df.loc[missing, "region_name"] = pd.Series(reg_names, index=df.index[missing], dtype="string")
        df.loc[missing, "country_name"] = pd.Series(country_names, index=df.index[missing], dtype="string")
    return df


def _add_missing_layer(fig, missing_ids: List[str], use_maplibre: bool):
    """Overlay a grey layer for polygons without data in the selected period and add a legend entry.
       NOTE: Choroplethmap (MapLibre) does not support 'opacity', so we avoid it here.
    """
    if not missing_ids:
        return

    const_scale = [[0, NO_DATA_COLOR], [1, NO_DATA_COLOR]]  # e.g., "#B0B0B0"

    if use_maplibre and hasattr(go, "Choroplethmap"):
        tr = go.Choroplethmap(
            geojson=GEOJSON_ROUTE,
            locations=missing_ids,
            featureidkey="properties.id",
            z=[0] * len(missing_ids),
            zmin=0, zmax=1,
            colorscale=const_scale,
            showscale=False,
            hoverinfo="skip",
            name=NO_DATA_LEGEND_LABEL,
            showlegend=True,
            marker=dict(line=dict(width=0)),
        )
    else:
        tr = go.Choroplethmapbox(
            geojson=GEOJSON_ROUTE,
            locations=missing_ids,
            featureidkey="properties.id",
            z=[0] * len(missing_ids),
            zmin=0, zmax=1,
            colorscale=const_scale,
            showscale=False,
            hoverinfo="skip",
            name=NO_DATA_LEGEND_LABEL,
            showlegend=True,
            marker=dict(line=dict(width=0)),
        )

    # Add and push the grey layer *behind* the colored layer
    fig.add_trace(tr)
    fig.data = (fig.data[-1],) + fig.data[:-1]


# =============================================================================
# 5) APP INIT & CSS
# =============================================================================
app = dash.Dash(__name__, title="Mapa de Pobreza Territorial LATAM")
server = app.server

# --- add near Dash init ---
from flask_compress import Compress
from flask import make_response, send_from_directory

Compress(app.server)  # gzip/brotli if client supports it

GEOJSON_ROUTE = "/latam_regiones_simplified.geojson"

@app.server.route(GEOJSON_ROUTE)
def _serve_geojson():
    resp = make_response(send_from_directory(str(BASE_DIR), GEOJSON_FILE))
    # cache hard for a year – Render free dyno restarts won't matter for a static asset
    resp.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return resp


GRAPH_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d","select2d","autoScale2d","toggleSpikelines",
        "hoverClosestCartesian","hoverCompareCartesian"
    ],
    "toImageButtonOptions": {"format": "png", "scale": 2},
}
MAP_STYLE = os.getenv("MAP_STYLE", "carto-positron")

def _period_marks():
    return { (idx := PERIOD_TO_IDX[p]): p for p in PERIODOS }

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
      body{ font-family:"Segoe UI","Roboto",Arial,sans-serif; background:#f8f9fa; margin:0; }
      .muted{color:var(--muted);}
      .section-card{ background:var(--card); border:1px solid var(--border); border-radius:12px;
        padding:12px 12px; box-shadow:var(--shadow); margin:12px auto; max-width: 100%; overflow: visible; }
      .floating-controls{ position:absolute; top:16px; left:8px; width:360px; background:var(--panel-bg);
        border:1px solid #e9ecef; border-radius:12px; box-shadow:var(--shadow);
        overflow:visible !important; z-index:2000; transition:width .2s ease; }
      .floating-controls.icon-only{ width:46px; overflow:hidden !important; }
      .control-panel-header{ display:flex; align-items:center; gap:10px; padding:12px 14px;
        background:var(--panel-header); border-bottom:1px solid #d6ebff; cursor:pointer; }
      .control-panel-header .header-icon{font-size:18px;}
      .control-panel-header .header-text{font-weight:700;color:var(--brand);font-size:15px;}
      .toggle-btn{ margin-left:auto; border:1px solid var(--brand); background:var(--brand-soft);
        color:var(--brand); border-radius:8px; font-size:12px; padding:4px 10px; cursor:pointer; }
      .controls-content{ padding:14px; max-height:74vh; overflow-y:auto; }
      .controls-content.hidden{display:none;}
      .controls-content > div { margin-bottom: 16px; }
      .controls-content label { display:block; font-weight:700; font-size:14px; margin-bottom:6px; }
      .lifted-dropdown .Select-menu-outer, .lifted-dropdown .Select__menu{ z-index:3000 !important; }
      .title-wrap{ position:absolute; top:14px; left:50%; transform:translateX(-50%);
        z-index:1200; text-align:center; pointer-events:none; }
      .title-chip{ background:rgba(255,255,255,.95); padding:8px 20px; border-radius:20px;
        font-size:14px; font-weight:700; color:#004085; box-shadow:0 2px 8px rgba(0,0,0,.12);
        border:1px solid rgba(0,64,133,.2); display:inline-block; }
      .title-spinner{ margin-top: 10px; }
      .title-spinner .dash-spinner{ transform:scale(.8); opacity:.85; }
      @media (max-width:900px){
        .floating-controls{ top:56px; left:6px; width:86vw; }
        .floating-controls.icon-only{ width:44px; }
      }
      .hero{ margin-bottom:28px; }
      .hero h1{ text-align:center; color:var(--brand); margin:8px 0 6px; font-size:28px; line-height:1.15;}
      .hero p { text-align:center; margin:0; font-size:16px; line-height:1.35;}
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
                                                options=(AGRI_OPTIONS or []),
                                                value=(AGRI_OPTIONS[0]["value"] if AGRI_OPTIONS else None) if AGRI_OPTIONS else None,
                                                clearable=False,
                                                disabled=(len(AGRI_OPTIONS) == 0),
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
# 6) CONTROLS CALLBACKS
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
    return ("AGRI" not in filtros) or (not AGRI_OPTIONS)

@app.callback(Output("dd-pais", "disabled"), Input("radio-escala", "value"))
def toggle_country_dropdown(modo):
    return (modo != "PAIS")


# =============================================================================
# 7) MAP BUILDERS
# =============================================================================
def _make_choropleth(df_map, zmin, zmax, center, zoom, indicador_label, missing_ids=None, geojson_url=None):
    # Use your Flask route if nothing is passed in
    if geojson_url is None:
        geojson_url = GEOJSON_ROUTE

    kwargs_common = dict(
        geojson=geojson_url,
        locations="feature_id",
        featureidkey="properties.id",
        color="value_pct",
        color_continuous_scale="YlOrRd",
        range_color=(zmin, zmax),
        center=center,
        zoom=zoom,
        hover_name="region_name",
    )

    # Try MapLibre first (px.choropleth_map); fallback to Mapbox
    try:
        fig = px.choropleth_map(df_map, map_style=MAP_STYLE, **kwargs_common)
        maplibre = True
    except Exception:
        warnings.filterwarnings("ignore", message=".*choropleth_mapbox.*", category=DeprecationWarning)
        fig = px.choropleth_mapbox(df_map, mapbox_style=MAP_STYLE, **kwargs_common)
        maplibre = False

    # (Optional) If you're on Mapbox, and still want some transparency on the colored layer,
    # only apply opacity to the colored trace(s) to avoid MapLibre incompatibilities.
    if not maplibre:
        for tr in fig.data:
            if getattr(tr, "showscale", False):
                tr.update(opacity=0.92)

    # Add grey 'no data' layer
    _add_missing_layer(fig, (missing_ids or []), use_maplibre=maplibre)

    # Legend space for the grey entry
    fig.update_layout(
        legend=dict(
            title="",
            orientation="h",
            yanchor="bottom", y=0.015,
            xanchor="left",   x=0.01,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=1,
        )
    )
    return fig



# =============================================================================
# 8) MAP CALLBACK
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
    """
    Build the map with a grey 'no data' layer. Requires:
      - _feature_ids_for_scope to return the full set of polygons for the scope
      - _fill_names_from_geojson to fill names for polygons added by reindexing
      - _make_choropleth(..., missing_ids=...) to inject the grey layer + legend
    """
    # Basic guards (keep original behavior for missing UI state)
    if not indicador_label or not periodo_idx:
        fig = go.Figure()
        fig.add_annotation(text="Selecciona una variable y un periodo",
                           showarrow=False, y=0.5, x=0.5)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        return fig, "Selecciona una variable y un periodo", ""

    indicador_col = INDICATOR_TO_COL.get(indicador_label, None)
    if not indicador_col:
        fig = go.Figure()
        fig.add_annotation(text="Variable desconocida. Intenta con otra.",
                           showarrow=False, y=0.5, x=0.5)
        fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))
        return fig, indicador_label, ""

    periodo_sel = IDX_TO_PERIOD.get(periodo_idx, PERIODOS[-1])
    subset_key, agri_def = resolve_subset(filtros, agri_def_label)

    # ---- Data slice
    if nivel == "NAT":
        df_map = build_national_layer(
            periodo_sel,
            subset_key,
            agri_def,
            indicador_col,
            country_filter=pais_iso3 if escala == "PAIS" else None,
        )
    else:
        df_map = _regional_slice(
            periodo_sel,
            subset_key,
            agri_def,
            indicador_col,
            country_filter=pais_iso3 if (escala == "PAIS" and pais_iso3) else None,
        )

    # ---- Feature subset & row order: always include ALL polygons for scope
    feature_ids = _feature_ids_for_scope(df_map, escala, pais_iso3)

    if df_map is None or df_map.empty:
        # Build a base frame with all polygons for the scope (NaNs everywhere)
        df_map = pd.DataFrame({
            "feature_id": feature_ids,
            "country_name": pd.Series([None] * len(feature_ids), dtype="string"),
            "region_name": pd.Series([None] * len(feature_ids), dtype="string"),
            "value_pct": np.nan,
            "se_pp": np.nan,
            "ci95_lo": np.nan,
            "ci95_hi": np.nan,
            "sample_n": np.nan,
        })
    else:
        # Reindex to include polygons with no data, which will be painted grey
        df_map = df_map.set_index("feature_id").reindex(feature_ids).reset_index()

    # Fill names for polygons that were introduced by reindexing
    df_map = _fill_names_from_geojson(df_map)

    # ---- Color scale bounds (robust to NaNs)
    zmin, zmax = _robust_min_max(df_map["value_pct"])

    # ---- Missing ids for the grey layer (NaN value => no data)
    missing_mask = ~np.isfinite(pd.to_numeric(df_map["value_pct"], errors="coerce"))
    missing_ids = df_map.loc[missing_mask, "feature_id"].astype(str).tolist()

    # ---- Center/zoom
    if escala == "PAIS" and pais_iso3 and pais_iso3 in COUNTRY_BBOX:
        center, zoom = _center_zoom_from_bbox(*COUNTRY_BBOX[pais_iso3])
    else:
        center, zoom = _bbox_from_features(feature_ids)

    # Preserve user pan/zoom (Mapbox / MapLibre / generic)
    if relayout_data and "mapbox.center" in relayout_data:
        zoom = relayout_data.get("mapbox.zoom", zoom)
        center = relayout_data.get("mapbox.center", center)
    if relayout_data and "maplibre.center" in relayout_data:
        zoom = relayout_data.get("maplibre.zoom", zoom)
        center = relayout_data.get("maplibre.center", center)
    if relayout_data and "map.center" in relayout_data:
        zoom = relayout_data.get("map.zoom", zoom)
        center = relayout_data.get("map.center", center)

    # ---- Hover/customdata for the colored layer only
    # (Grey layer is added inside _make_choropleth with hover disabled)
    z = (
        df_map["value_pct"]
        .astype("float32").round(1)
        .astype("float16").astype(float)
        .tolist()
    )

    custom_stack = np.stack([
        df_map["country_name"].astype(str).values,
        df_map["region_name"].astype(str).values,
        df_map["ci95_lo"].astype("float32").round(1).astype("float16").astype(float).values,
        df_map["ci95_hi"].astype("float32").round(1).astype("float16").astype(float).values,
        df_map["se_pp"].astype("float32").round(1).astype("float16").astype(float).values,
        df_map["sample_n"].astype("float32").round(0).astype("float16").astype(float).values,
    ], axis=-1)
    customdata = custom_stack.tolist()

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

    # ---- Full rebuild (we intentionally skip the Patch fast path
    #      so the grey layer stays in sync)
    fig = _make_choropleth(
        df_map,
        zmin,
        zmax,
        center,
        zoom,
        indicador_label,
        missing_ids=missing_ids
    )

    # Apply hover + customdata ONLY to the colored trace (the one with a color scale)
    def _apply_to_colored(tr):
        # Choropleth traces that show the colorbar are the colored layer
        return getattr(tr, "showscale", False) is True

    for tr in fig.data:
        if _apply_to_colored(tr):
            tr.hovertemplate = hovertemplate
            tr.customdata = customdata
            # Also set z explicitly to keep fast re-draws consistent
            tr.z = z

    # ---- Title
    nivel_txt = "Nacional" if nivel == "NAT" else "Regional"
    scope_title = (
        COUNTRY_NAME_BY_CODE.get(pais_iso3, "América Latina")
        if (escala == "PAIS" and pais_iso3) else "América Latina"
    )
    map_title = f"{indicador_label} · {nivel_txt} · {scope_title} · Periodo {periodo_sel}"
    if subpop_txt:
        map_title += f" · Subpoblación: {subpop_txt}"

    # ---- Layout polish
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="%"),
        uirevision="keep",
    )

    return fig, map_title, ""


# =============================================================================
# 9) RUN
# =============================================================================
if __name__ == "__main__":
    is_render = bool(os.getenv("RENDER") or os.getenv("PORT"))
    host = "0.0.0.0" if is_render else "127.0.0.1"
    port = int(os.getenv("PORT", "8050"))
    debug_flag = os.getenv("DASH_DEBUG", "1") == "1"
    print(f"→ Open http://localhost:{port}")
    app.run(host=host, port=port, debug=debug_flag)

