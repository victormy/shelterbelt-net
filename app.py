"""Streamlit UI for the Shelterbelt Segmentation Pipeline.

Launch with:
    GEOAI_ROOT=/path/to/data streamlit run app.py

The sidebar reads defaults from ``config.py`` and lets the user override
them.  Overridden values are passed as keyword arguments to the pipeline
functions.
"""

from __future__ import annotations

import traceback
from pathlib import Path

import streamlit as st

import config

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Shelterbelt Segmentation",
    page_icon=None,
    layout="wide",
)

_CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — reads defaults from config, user can override
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## Configuration")
    st.text_input("GEOAI_ROOT", value=str(config.GEOAI_ROOT), disabled=True)
    ui_year = st.text_input("Year", value=config.YEAR)
    ui_data_type = st.text_input("Data type", value=config.DATA_TYPE)
    ui_architecture = st.selectbox(
        "Architecture",
        ["_conv", "_trans"],
        index=["_conv", "_trans"].index(config.ARCHITECTURE),
    )
    ui_normalize = st.selectbox(
        "Inference normalization",
        ["minmax", "dataset", "imagenet"],
        index=["minmax", "dataset", "imagenet"].index(config.NORMALIZE if config.NORMALIZE in ["minmax", "dataset", "imagenet"] else "minmax"),
        help="Must match training normalization. 'dataset' (/ 255.0) matches geoai dataset and is required for 16-bit ps_scenes.",
    )
    ui_channels = config.num_channels(ui_data_type)
    st.text_input("Channels (auto)", value=str(ui_channels), disabled=True)


# ---------------------------------------------------------------------------
# Stage definitions — kwargs_extra supplies UI overrides
# ---------------------------------------------------------------------------
def _common() -> dict:
    return {"year": ui_year, "data_type": ui_data_type}


STAGES: list[dict] = [
    {
        "num": 0,
        "key": "stack",
        "title": "Multi-seasonal stack",
        "desc": "Stack monthly RGB basemaps into multi-temporal composites.",
        "func": "stack_multiseasonal",
        "kwargs_extra": lambda: {"year": ui_year},
    },
    {
        "num": 1,
        "key": "reclassify",
        "title": "Reclassify masks",
        "desc": "Filter reference polygons by stand type into binary masks.",
        "func": "reclassify_masks",
        "kwargs_extra": lambda: {"year": ui_year},
    },
    {
        "num": 2,
        "key": "chip",
        "title": "Create training dataset",
        "desc": "Chip tiles and vector masks into 256x256 GeoTIFF pairs.",
        "func": "chip_training_dataset",
        "kwargs_extra": _common,
    },
    {
        "num": 3,
        "key": "train",
        "title": "Train / validate / test",
        "desc": "Train segmentation model, evaluate on held-out test set.",
        "func": "train_model",
        "kwargs_extra": lambda: {**_common(), "architecture": ui_architecture},
    },
    {
        "num": 4,
        "key": "inference",
        "title": "Run inference",
        "desc": "Sliding-window prediction over all AOI tiles.",
        "func": "run_inference",
        "kwargs_extra": lambda: {
            **_common(),
            "architecture": ui_architecture,
            "normalize": ui_normalize,
        },
    },
    {
        "num": 5,
        "key": "mosaic",
        "title": "Mosaic and reproject",
        "desc": "Merge prediction tiles and reproject to Ukraine Albers.",
        "func": "mosaic_predictions",
        "kwargs_extra": _common,
    },
]

# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------
if "stage_status" not in st.session_state:
    st.session_state.stage_status = {s["key"]: "pending" for s in STAGES}
if "stage_logs" not in st.session_state:
    st.session_state.stage_logs = {s["key"]: "" for s in STAGES}

# ---------------------------------------------------------------------------
# Sidebar — stage status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("## Stage status")
    for stage in STAGES:
        status = st.session_state.stage_status[stage["key"]]
        if status == "done":
            label = f"[done] {stage['num']}. {stage['title']}"
            st.markdown(f'<p class="status-done">{label}</p>', unsafe_allow_html=True)
        elif status == "error":
            label = f"[error] {stage['num']}. {stage['title']}"
            st.markdown(f'<p class="status-error">{label}</p>', unsafe_allow_html=True)
        else:
            label = f"[pending] {stage['num']}. {stage['title']}"
            st.markdown(f'<p class="status-pending">{label}</p>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown("# Shelterbelt Segmentation Pipeline")

for stage in STAGES:
    key = stage["key"]
    with st.expander(
        f'{stage["num"]}. {stage["title"]}',
        expanded=(st.session_state.stage_status[key] != "done"),
    ):
        st.markdown(f'<span class="muted-text">{stage["desc"]}</span>', unsafe_allow_html=True)

        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            run_clicked = st.button("Run", key=f"run_{key}")
        with col_status:
            status = st.session_state.stage_status[key]
            if status == "done":
                st.markdown('<span class="status-done">Completed</span>', unsafe_allow_html=True)
            elif status == "error":
                st.markdown('<span class="status-error">Failed</span>', unsafe_allow_html=True)

        progress_bar = st.progress(0.0, text="")
        log_area = st.empty()

        if st.session_state.stage_logs[key]:
            log_area.markdown(
                f'<div class="log-pane">{st.session_state.stage_logs[key]}</div>',
                unsafe_allow_html=True,
            )

        if run_clicked:
            import pipeline

            st.session_state.stage_status[key] = "running"
            st.session_state.stage_logs[key] = ""
            log_lines: list[str] = []

            def make_progress(pbar, log_el, lines):
                def cb(frac: float, msg: str):
                    pbar.progress(min(frac, 1.0), text=msg)
                    lines.append(msg)
                    log_el.markdown(
                        f'<div class="log-pane">{chr(10).join(lines[-50:])}</div>',
                        unsafe_allow_html=True,
                    )
                return cb

            fn = getattr(pipeline, stage["func"])
            extras = stage["kwargs_extra"]
            kwargs = extras() if callable(extras) else dict(extras)

            try:
                fn(progress=make_progress(progress_bar, log_area, log_lines), **kwargs)
                st.session_state.stage_status[key] = "done"
                st.session_state.stage_logs[key] = "\n".join(log_lines)
                progress_bar.progress(1.0, text="Done")
                st.rerun()
            except Exception:
                st.session_state.stage_status[key] = "error"
                err = traceback.format_exc()
                log_lines.append(f"\nERROR:\n{err}")
                st.session_state.stage_logs[key] = "\n".join(log_lines)
                log_area.markdown(
                    f'<div class="log-pane">{chr(10).join(log_lines[-50:])}</div>',
                    unsafe_allow_html=True,
                )
                st.error(f"Stage {stage['num']} failed. See log above.")
