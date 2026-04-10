"""TransferIQ Streamlit experience."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.transfer_value_system import METADATA_DIR, PLAYER_LIBRARY_PATH, RAW_DATA_PATH, TEST_PREDICTIONS_PATH, TransferValuePredictor

EUR_TO_INR_RATE = 106.6
EUR_TO_INR_RATE_DATE = "2026-03-29"


st.set_page_config(page_title="TransferIQ", page_icon="⚽", layout="wide")


st.markdown(
    """
    <style>
    :root {
        --bg-1: #06131a;
        --bg-2: #0d2027;
        --panel: rgba(8, 23, 32, 0.72);
        --panel-strong: rgba(12, 30, 42, 0.9);
        --line: rgba(255,255,255,0.08);
        --text: #edf8f5;
        --muted: #9fc0b7;
        --aqua: #6ef1dc;
        --mint: #8df5a3;
        --ice: #8dc8ff;
        --gold: #ffd16d;
        --coral: #ff8a7a;
    }
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        background: transparent;
    }
    .stApp {
        min-height: 100vh;
        background:
            radial-gradient(circle at 15% 0%, rgba(110, 241, 220, 0.16), transparent 28%),
            radial-gradient(circle at 85% 10%, rgba(141, 200, 255, 0.14), transparent 24%),
            linear-gradient(180deg, var(--bg-1) 0%, var(--bg-2) 55%, #102e35 100%);
        color: var(--text);
        overflow-x: hidden;
    }
    [data-testid="stHeader"] {
        background: transparent;
        height: 0;
    }
    [data-testid="stToolbar"] {
        right: 0.35rem;
        top: 0.15rem;
        z-index: 20;
    }
    [data-testid="stDecoration"] {
        display: none;
    }
    .block-container {
        width: min(100%, 1600px);
        max-width: none;
        padding: 0.9rem 2rem 2.5rem;
    }
    h1, h2, h3, h4, p, label, span, div {
        color: var(--text);
    }
    .hero-shell, .glass, .metric-card, .spotlight-card, .insight-card {
        border: 1px solid var(--line);
        background: var(--panel);
        backdrop-filter: blur(18px);
        border-radius: 28px;
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.18);
    }
    .hero-shell {
        overflow: hidden;
        padding: 0;
        margin-bottom: 4.5rem;
        position: relative;
    }
    .landing-grid {
        display: grid;
        grid-template-columns: 1.15fr 0.85fr;
        min-height: clamp(460px, 60vh, 680px);
    }
    .landing-copy {
        padding: 2.35rem 3rem 4.6rem;
        position: relative;
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        background: rgba(110, 241, 220, 0.12);
        color: var(--aqua);
        font-size: 0.8rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        font-weight: 700;
    }
    .landing-title {
        font-size: clamp(2.8rem, 5vw, 5.6rem);
        line-height: 0.95;
        margin: 1.2rem 0 1rem 0;
        letter-spacing: -0.04em;
    }
    .landing-title span {
        color: var(--aqua);
    }
    .landing-text {
        color: var(--muted);
        max-width: 42rem;
        font-size: 1.08rem;
        line-height: 1.7;
    }
    .landing-go {
        width: clamp(88px, 7vw, 108px);
        margin: -8.4rem clamp(3.8rem, 7vw, 7rem) 0 auto;
        position: relative;
        z-index: 14;
        filter: drop-shadow(0 14px 20px rgba(0, 0, 0, 0.28));
    }
    .landing-go div[data-testid="stButton"] {
        margin: 0;
    }
    .landing-go button {
        width: 100%;
        aspect-ratio: 1;
        min-height: auto;
        padding: 0;
        border-radius: 50%;
        border: 2px solid rgba(16, 28, 36, 0.92);
        color: #0d1820;
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        background:
            radial-gradient(circle at 50% 50%, rgba(255,255,255,0.98) 0 58%, rgba(232,238,245,0.98) 59% 100%),
            radial-gradient(circle at 50% 26%, #16232d 0 8%, transparent 9%),
            radial-gradient(circle at 23% 56%, #16232d 0 8%, transparent 9%),
            radial-gradient(circle at 77% 56%, #16232d 0 8%, transparent 9%),
            radial-gradient(circle at 36% 81%, #16232d 0 6.5%, transparent 7.5%),
            radial-gradient(circle at 64% 81%, #16232d 0 6.5%, transparent 7.5%);
        box-shadow:
            inset 0 -8px 12px rgba(0, 0, 0, 0.14),
            inset 0 8px 10px rgba(255, 255, 255, 0.55),
            0 10px 22px rgba(0, 0, 0, 0.22);
        transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
    }
    .landing-go button:hover {
        transform: translateY(-4px) scale(1.04);
        box-shadow:
            inset 0 -8px 12px rgba(0, 0, 0, 0.16),
            inset 0 8px 10px rgba(255, 255, 255, 0.58),
            0 16px 28px rgba(0, 0, 0, 0.26);
    }
    .landing-go button:active {
        transform: translateY(-1px) scale(0.98);
    }
    .landing-go button:focus-visible {
        outline: 3px solid rgba(141, 245, 163, 0.75);
        outline-offset: 3px;
    }
    .landing-go-launching {
        visibility: hidden;
        pointer-events: none;
        margin-bottom: 0;
    }
    .landing-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin: 1.6rem 0 2rem 0;
    }
    .landing-pill {
        padding: 0.7rem 0.95rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.06);
        color: #dcece8;
        font-size: 0.92rem;
    }
    .kick-stage {
        position: relative;
        min-height: 100%;
        overflow: hidden;
        background:
            linear-gradient(180deg, rgba(111,241,220,0.06), transparent 32%),
            linear-gradient(180deg, rgba(13,32,39,0.1), rgba(8,23,32,0.95));
    }
    .kick-stage::before {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        bottom: 0;
        height: 26%;
        background:
            linear-gradient(180deg, rgba(17, 61, 57, 0.18), rgba(22, 78, 63, 0.34)),
            radial-gradient(circle at 50% 120%, rgba(148, 236, 166, 0.18), transparent 38%);
        border-top: 1px solid rgba(126, 234, 188, 0.12);
    }
    .hero-shell.launch-active .kick-stage::before {
        background:
            linear-gradient(180deg, rgba(20, 78, 70, 0.28), rgba(39, 116, 84, 0.42)),
            radial-gradient(circle at 50% 120%, rgba(176, 251, 204, 0.28), transparent 40%);
    }
    .kick-stage::after {
        content: "";
        position: absolute;
        inset: auto 0 0 0;
        height: 30%;
        background:
            radial-gradient(circle at 50% 120%, rgba(141,245,163,0.22), transparent 35%),
            linear-gradient(180deg, rgba(141,245,163,0.06), rgba(141,245,163,0.18));
    }
    .pitch-lines {
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px);
        background-size: 84px 84px;
        opacity: 0.28;
    }
    .hero-shell.launch-active .pitch-lines {
        opacity: 0.18;
    }
    .ai-orbit {
        position: absolute;
        top: 14%;
        right: 10%;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        border: 1px solid rgba(110,241,220,0.18);
        box-shadow: 0 0 90px rgba(110,241,220,0.12);
    }
    .ai-orbit::before, .ai-orbit::after {
        content: "";
        position: absolute;
        inset: 14px;
        border-radius: 50%;
        border: 1px solid rgba(141,200,255,0.16);
    }
    .ai-orbit::after {
        inset: 40px;
        border-color: rgba(255,209,109,0.14);
    }
    .player-wrap {
        position: absolute;
        right: 18%;
        bottom: 16%;
        width: 260px;
        height: 340px;
    }
    .ball {
        position: absolute;
        width: 36px;
        height: 36px;
        left: 66px;
        bottom: 28px;
        border-radius: 50%;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #f5f8fc 40%, #c9d5e2 72%, #8ca1b7 100%);
        border: 1px solid rgba(255,255,255,0.7);
        box-shadow: 0 0 25px rgba(255,255,255,0.28), 0 10px 18px rgba(0,0,0,0.18);
        animation: ball-flight 2.8s ease-in-out infinite;
        z-index: 5;
    }
    .ball::before,
    .ball::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 50%;
    }
    .ball::before {
        background:
            radial-gradient(circle at 50% 22%, transparent 0 16%, rgba(26,38,47,0.95) 17% 22%, transparent 23%),
            radial-gradient(circle at 22% 56%, transparent 0 16%, rgba(26,38,47,0.95) 17% 22%, transparent 23%),
            radial-gradient(circle at 78% 56%, transparent 0 16%, rgba(26,38,47,0.95) 17% 22%, transparent 23%),
            radial-gradient(circle at 36% 82%, transparent 0 13%, rgba(26,38,47,0.92) 14% 18%, transparent 19%),
            radial-gradient(circle at 64% 82%, transparent 0 13%, rgba(26,38,47,0.92) 14% 18%, transparent 19%);
        opacity: 0.95;
    }
    .ball::after {
        box-shadow: inset 0 -4px 8px rgba(0,0,0,0.14);
    }
    .ball-ghost {
        position: absolute;
        width: 36px;
        height: 36px;
        left: 66px;
        bottom: 28px;
        border-radius: 50%;
        border: 1px solid rgba(255,255,255,0.14);
        background: radial-gradient(circle, rgba(255,255,255,0.1), rgba(255,255,255,0.02));
        opacity: 0;
        filter: blur(0.5px);
        pointer-events: none;
    }
    .player-head {
        position: absolute;
        top: 8px;
        left: 116px;
        width: 52px;
        height: 60px;
        border-radius: 50%;
        background: radial-gradient(circle at 48% 54%, #efc7a3 0%, #dca37f 58%, #b97858 100%);
        box-shadow: 0 10px 26px rgba(0,0,0,0.18);
        z-index: 4;
    }
    .player-head::before {
        content: "";
        position: absolute;
        left: 2px;
        right: 2px;
        top: -4px;
        height: 28px;
        border-radius: 28px 28px 18px 18px;
        background:
            radial-gradient(circle at 55% 30%, rgba(255,255,255,0.18), transparent 30%),
            linear-gradient(180deg, #132632, #09151d 82%);
    }
    .player-head::after {
        content: "";
        position: absolute;
        left: 15px;
        top: 28px;
        width: 22px;
        height: 10px;
        border-bottom: 2px solid rgba(92, 44, 30, 0.5);
        border-radius: 0 0 16px 16px;
        opacity: 0.8;
    }
    .player-body {
        position: absolute;
        top: 70px;
        left: 100px;
        width: 86px;
        height: 116px;
        border-radius: 30px 30px 34px 28px;
        transform: skew(-7deg);
        background: linear-gradient(180deg, #22465b, #122b3b 68%, #0c1e29);
        border: 1px solid rgba(110,241,220,0.18);
        box-shadow: inset 0 0 0 1px rgba(173, 236, 255, 0.06);
    }
    .arm {
        position: absolute;
        width: 18px;
        height: 88px;
        border-radius: 16px;
        background: linear-gradient(180deg, #1d3b4c, #102330);
        transform-origin: top center;
    }
    .arm-left {
        top: 84px;
        left: 76px;
        transform: rotate(30deg);
    }
    .arm-right {
        top: 82px;
        left: 184px;
        transform: rotate(-40deg);
    }
    .leg {
        position: absolute;
        width: 22px;
        height: 122px;
        border-radius: 20px;
        background: linear-gradient(180deg, #173645, #0e2231);
        transform-origin: top center;
    }
    .leg-left {
        top: 176px;
        left: 126px;
        transform: rotate(8deg);
    }
    .leg-right {
        top: 170px;
        left: 174px;
        transform: rotate(-47deg);
        animation: kick 2.8s ease-in-out infinite;
    }
    .foot {
        position: absolute;
        width: 48px;
        height: 14px;
        border-radius: 14px;
        background: linear-gradient(90deg, #ffd16d, #ff8a7a);
    }
    .foot-left {
        left: 118px;
        bottom: 10px;
        transform: rotate(3deg);
    }
    .foot-right {
        left: 194px;
        bottom: 100px;
        transform: rotate(-18deg);
        animation: foot-snap 2.8s ease-in-out infinite;
    }
    .arc {
        position: absolute;
        left: 58px;
        bottom: 44px;
        width: 224px;
        height: 132px;
        border-top: 2px dashed rgba(110,241,220,0.4);
        border-radius: 50%;
        transform: rotate(-18deg);
        opacity: 0.7;
    }
    .hero-shell.launch-active .arc {
        opacity: 1;
        border-top-color: rgba(189, 255, 226, 0.8);
    }
    .dashboard-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
        flex-wrap: wrap;
    }
    .title-chip {
        display: inline-block;
        color: var(--aqua);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.82rem;
        font-weight: 700;
    }
    .dashboard-head {
        font-size: clamp(2rem, 4vw, 3.7rem);
        line-height: 0.96;
        margin: 0.3rem 0 0.6rem 0;
    }
    .subtitle {
        color: var(--muted);
        max-width: 54rem;
        line-height: 1.7;
    }
    .metric-card {
        padding: 1rem 1.15rem;
        min-height: 140px;
    }
    .player-banner {
        display: grid;
        grid-template-columns: 1.35fr 0.65fr;
        gap: 1rem;
        padding: 1.1rem 1.2rem;
        margin: 0.2rem 0 0.9rem;
        border: 1px solid rgba(110,241,220,0.12);
        background: linear-gradient(135deg, rgba(110,241,220,0.08), rgba(141,200,255,0.05));
        border-radius: 24px;
    }
    .player-name {
        font-size: clamp(1.4rem, 2.1vw, 2.2rem);
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    .player-meta {
        color: var(--muted);
        font-size: 0.96rem;
        margin-top: 0.35rem;
        line-height: 1.6;
    }
    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.9rem;
    }
    .mini-pill {
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        color: var(--text);
        font-size: 0.88rem;
    }
    .guess-strip {
        display: grid;
        gap: 0.75rem;
        align-content: start;
    }
    .guess-card {
        padding: 1rem 1.05rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(5, 18, 25, 0.45);
    }
    .guess-title {
        color: var(--muted);
        font-size: 0.78rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
    }
    .guess-value {
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 0.45rem;
    }
    .guess-help {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: 0.35rem;
    }
    .metric-label {
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-size: 0.76rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-top: 0.55rem;
    }
    .metric-sub {
        color: var(--muted);
        font-size: 0.92rem;
        margin-top: 0.4rem;
    }
    .spotlight-card {
        padding: 1.2rem;
        margin-top: 0.6rem;
    }
    .insight-card {
        padding: 1.2rem;
        height: 100%;
    }
    div[data-testid="stVerticalBlock"] > div:has(> div > .metric-card) {
        height: 100%;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.6rem;
        flex-wrap: wrap;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(16, 43, 52, 0.92);
        border: 1px solid rgba(173, 231, 220, 0.16);
        border-radius: 16px;
        padding: 0.5rem 0.8rem;
        min-height: 3rem;
        color: #eefcf8 !important;
        font-weight: 700 !important;
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.03);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(110,241,220,0.28), rgba(141,200,255,0.26)) !important;
        color: #ffffff !important;
        border-color: rgba(141,245,163,0.45) !important;
        box-shadow: 0 0 0 1px rgba(141,245,163,0.18), 0 10px 24px rgba(0,0,0,0.18);
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(25, 58, 69, 0.98) !important;
        color: #ffffff !important;
    }
    div[data-testid="column"] {
        width: 100%;
    }
    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input,
    .stFileUploader section,
    .stSlider [data-baseweb="slider"] {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 18px !important;
        color: var(--text) !important;
    }
    .stFileUploader section {
        background: rgba(16, 41, 49, 0.92) !important;
        border: 1px solid rgba(165, 230, 219, 0.14) !important;
    }
    .stFileUploader section small,
    .stFileUploader section span,
    .stFileUploader section div,
    .stFileUploader label {
        color: #e9faf6 !important;
    }
    .stFileUploader [data-testid="stBaseButton-secondary"] {
        background: linear-gradient(135deg, #2a7379, #4f9ea6) !important;
        border: 1px solid rgba(214, 255, 248, 0.5) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.18);
    }
    .stFileUploader [data-testid="stBaseButton-secondary"]:hover {
        background: linear-gradient(135deg, #33868d, #63b2ba) !important;
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(20, 47, 56, 0.95) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #eefcf8 !important;
    }
    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        overflow: hidden;
        background: rgba(9, 27, 35, 0.9) !important;
    }
    [data-testid="stDataFrame"] div,
    [data-testid="stTable"] div {
        color: #eefcf8 !important;
    }
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stTable"] [role="columnheader"] {
        background: rgba(24, 61, 72, 0.95) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    [data-testid="stDataFrame"] [role="gridcell"],
    [data-testid="stTable"] [role="gridcell"] {
        background: rgba(11, 33, 41, 0.96) !important;
        color: #eefcf8 !important;
    }
    [data-testid="stDataFrame"] [role="row"]:nth-child(even) [role="gridcell"] {
        background: rgba(14, 38, 47, 0.98) !important;
    }
    .stNumberInput input,
    .stTextInput input,
    .stSelectbox input {
        color: #f4fffb !important;
        -webkit-text-fill-color: #f4fffb !important;
        caret-color: #f4fffb !important;
        background: rgba(12, 32, 40, 0.96) !important;
        font-weight: 700 !important;
    }
    .stNumberInput [data-testid="stNumberInputStepUp"],
    .stNumberInput [data-testid="stNumberInputStepDown"] {
        background: rgba(20, 54, 64, 0.92) !important;
        color: #ffffff !important;
        border-left: 1px solid rgba(255,255,255,0.08) !important;
    }
    .stNumberInput button,
    .stNumberInput button svg {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        box-shadow: 0 0 0 2px rgba(255,255,255,0.14);
    }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"],
    .stSlider div[data-testid="stSliderTickBar"] {
        color: #ecfbf7 !important;
    }
    .stSelectbox label,
    .stNumberInput label,
    .stSlider label {
        font-weight: 600;
    }
    div[data-baseweb="popover"] {
        background: rgba(6, 19, 26, 0.96) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 18px !important;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35) !important;
    }
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] div {
        background: transparent !important;
        color: #eaf8f3 !important;
    }
    div[data-baseweb="popover"] [aria-selected="true"] {
        background: rgba(110,241,220,0.14) !important;
        color: #ffffff !important;
    }
    .stMarkdown, .stCaption, .stMetric {
        color: var(--text);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 20px;
        padding: 0.9rem 1rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 18px;
        border: 1px solid rgba(170, 241, 228, 0.42);
        background: linear-gradient(135deg, #1f6469, #2f7880 48%, #4d8f97);
        color: #ffffff !important;
        font-weight: 800;
        min-height: 3rem;
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.2);
        text-shadow: 0 1px 1px rgba(0,0,0,0.18);
    }
    .stButton > button *,
    .stDownloadButton > button * {
        color: #ffffff !important;
        fill: #ffffff !important;
    }
    .stButton > button:hover {
        border-color: rgba(207, 255, 247, 0.75);
        background: linear-gradient(135deg, #26757a, #38919a 52%, #5ca7af);
        color: #ffffff !important;
    }
    .stButton > button:focus,
    .stButton > button:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(141, 245, 163, 0.22), 0 12px 26px rgba(0, 0, 0, 0.22) !important;
    }
    .stDownloadButton > button {
        width: 100%;
        border-radius: 18px;
        border: 1px solid rgba(255, 223, 158, 0.48);
        background: linear-gradient(135deg, #8b5a20, #b87a2a 52%, #d29a46);
        color: #ffffff !important;
        font-weight: 800;
        min-height: 3rem;
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.2);
        text-shadow: 0 1px 1px rgba(0,0,0,0.18);
    }
    .stDownloadButton > button:hover {
        border-color: rgba(255, 240, 201, 0.72);
        background: linear-gradient(135deg, #a46a26, #ca8b34 52%, #e3ae55);
        color: #ffffff !important;
    }
    details[data-testid="stExpander"] {
        background: rgba(12, 32, 40, 0.92);
        border: 1px solid rgba(165, 230, 219, 0.14);
        border-radius: 18px;
        padding: 0.25rem 0.45rem;
        margin-top: 0.75rem;
    }
    details[data-testid="stExpander"] summary {
        background: rgba(15, 36, 46, 0.96) !important;
        border-radius: 14px !important;
        padding: 0.8rem 0.95rem !important;
        color: #f4fffb !important;
        font-weight: 700 !important;
    }
    details[data-testid="stExpander"] summary:hover {
        background: rgba(22, 51, 63, 0.98) !important;
    }
    details[data-testid="stExpander"] summary *,
    details[data-testid="stExpander"] summary p,
    details[data-testid="stExpander"] summary span,
    details[data-testid="stExpander"] summary svg {
        color: #f4fffb !important;
        fill: #8df5a3 !important;
    }
    @keyframes kick {
        0% { transform: rotate(-47deg); }
        20% { transform: rotate(-47deg); }
        40% { transform: rotate(12deg); }
        52% { transform: rotate(-56deg); }
        100% { transform: rotate(-47deg); }
    }
    @keyframes foot-snap {
        0% { transform: rotate(-18deg); }
        40% { transform: rotate(12deg); }
        55% { transform: rotate(-26deg); }
        100% { transform: rotate(-18deg); }
    }
    @keyframes ball-flight {
        0% { transform: translate(0, 0) scale(1); opacity: 1; }
        22% { transform: translate(0, 0) scale(1); opacity: 1; }
        55% { transform: translate(118px, -96px) scale(1.03); opacity: 1; }
        100% { transform: translate(214px, -132px) scale(0.9); opacity: 0.72; }
    }
    @keyframes ball-launch {
        0% { transform: translate(0, 0) scale(1); opacity: 1; filter: blur(0); }
        16% { transform: translate(18px, -8px) scale(1.05); opacity: 1; }
        45% { transform: translate(154px, -82px) scale(1.7); opacity: 0.48; filter: blur(1px); }
        78% { transform: translate(270px, -154px) scale(2.7); opacity: 0.16; filter: blur(2.5px); }
        100% { transform: translate(340px, -198px) scale(3.25); opacity: 0; filter: blur(4px); }
    }
    @keyframes ghost-stay {
        0% { opacity: 0; }
        18% { opacity: 0.22; }
        100% { opacity: 0; }
    }
    .hero-shell.launch-active .ball {
        animation: ball-launch 1.18s cubic-bezier(0.17, 0.84, 0.44, 1) forwards;
    }
    .hero-shell.launch-active .ball-ghost {
        animation: ghost-stay 0.8s ease-out forwards;
    }
    .hero-shell.launch-active .leg-right {
        animation: kick 0.68s ease-out forwards;
    }
    .hero-shell.launch-active .foot-right {
        animation: foot-snap 0.68s ease-out forwards;
    }
    @media (max-width: 980px) {
        .block-container { padding: 0.35rem 1rem 1.8rem; }
        .landing-grid { grid-template-columns: 1fr; }
        .kick-stage { min-height: 360px; }
        .landing-copy { padding: 2rem 1.2rem 4.2rem; }
        .hero-shell { margin-bottom: 4rem; }
        .landing-go { width: clamp(82px, 12vw, 98px); margin: -7.2rem 5.4rem 0 auto; }
        .player-wrap { right: 50%; transform: translateX(40%); bottom: 10%; }
        .ai-orbit { width: 160px; height: 160px; top: 8%; right: 8%; }
    }
    @media (max-width: 640px) {
        .block-container { padding: 0.25rem 0.75rem 1.4rem; }
        .hero-shell, .glass, .metric-card, .spotlight-card, .insight-card {
            border-radius: 22px;
        }
        .player-banner { grid-template-columns: 1fr; }
        .landing-title { font-size: 2.5rem; }
        .landing-text { font-size: 0.96rem; }
        .kick-stage { min-height: 310px; }
        .landing-copy { padding-bottom: 3.9rem; }
        .hero-shell { margin-bottom: 3.6rem; }
        .landing-go { width: 78px; margin: -6.1rem 4.15rem 0 auto; }
        .landing-go button { font-size: 0.86rem; }
        .player-wrap { transform: translateX(32%) scale(0.82); }
        .dashboard-head { font-size: 2rem; }
        .metric-card, .spotlight-card, .insight-card { padding: 1rem; }
        .metric-value { font-size: 1.7rem; }
        .guess-value { font-size: 1.2rem; }
        .stTabs [data-baseweb="tab"] {
            flex: 1 1 calc(50% - 0.6rem);
            justify-content: center;
            padding: 0.5rem 0.7rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> TransferValuePredictor:
    return TransferValuePredictor()


@st.cache_data
def load_library() -> pd.DataFrame:
    return pd.read_csv(PLAYER_LIBRARY_PATH)


@st.cache_data
def load_raw_profiles() -> pd.DataFrame:
    return pd.read_csv(RAW_DATA_PATH)


@st.cache_data
def load_metrics() -> dict:
    with open(METADATA_DIR / "training_summary.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(TEST_PREDICTIONS_PATH)


def init_state() -> None:
    defaults = {
        "entered": False,
        "selected_player": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def build_row_from_library(record: pd.Series, raw_profiles: pd.DataFrame | None = None) -> pd.DataFrame:
    if raw_profiles is not None and "player_id" in raw_profiles.columns:
        matching_raw = raw_profiles.loc[raw_profiles["player_id"] == record.get("player_id", -1)]
        if not matching_raw.empty:
            return matching_raw.head(1).reset_index(drop=True).copy()

    market_value = float(record.get("current_market_value", 0))
    goals = int(record.get("total_goals", 0))
    assists = int(record.get("total_assists", 0))
    injuries = int(record.get("total_injuries", 0))
    return pd.DataFrame(
        [
            {
                "player_id": record.get("player_id", 0),
                "player_name": record["player_name"],
                "current_club_name": record.get("current_club_name", "Unknown"),
                "contract_expires": "30-06-2028",
                "seasons": "25/26, 24/25, 23/24",
                "competitions": "Premier League, Domestic Cup",
                "clubs": f"{record.get('current_club_name', 'Unknown')}, Previous Club",
                "total_goals": goals,
                "total_assists": assists,
                "current_market_value": market_value,
                "total_injuries": injuries,
                "total_days_missed": max(0.0, injuries * 18.0),
                "transfer_date": "01-07-2026",
                "from_team_name": record.get("from_team_name", "Previous Club"),
                "to_team_name": record.get("to_team_name", record.get("current_club_name", "Unknown")),
                "avg_sentiment_3m": float(record.get("avg_sentiment_3m", 0.5)),
                "sentiment_trend": 0.03,
                "sentiment_volatility": 0.10,
                "avg_monthly_mentions": 900.0,
                "mention_trend": 35.0,
                "engagement_rate": 14.0,
                "positive_sentiment_ratio": 0.62,
                "negative_sentiment_ratio": 0.18,
                "event_count": 2.0,
                "peak_sentiment": 0.84,
                "lowest_sentiment": 0.40,
                "recent_event": True,
            }
        ]
    )


def ai_brief(prediction: pd.Series) -> str:
    momentum = "accelerating" if prediction["prediction_delta_vs_current_value"] > 0 else "conservative"
    injury_text = str(prediction["injury_risk_category"]).replace("_", " ").lower()
    sentiment_text = str(prediction["sentiment_band"]).lower()
    return (
        f"TransferIQ flags this profile as {momentum} value formation. "
        f"The player sits in a {prediction['career_stage'].lower()} phase with {sentiment_text} public sentiment, "
        f"{injury_text} injury exposure, and an AI confidence score of {prediction['prediction_confidence']:.0f}%."
    )


def format_eur(value: float) -> str:
    return f"EUR {float(value):,.0f}"


def format_indian_number(value: float) -> str:
    rounded = int(round(float(value)))
    sign = "-" if rounded < 0 else ""
    digits = str(abs(rounded))
    if len(digits) <= 3:
        return sign + digits
    last_three = digits[-3:]
    remaining = digits[:-3]
    groups: list[str] = []
    while len(remaining) > 2:
        groups.append(remaining[-2:])
        remaining = remaining[:-2]
    if remaining:
        groups.append(remaining)
    return sign + ",".join(reversed(groups)) + "," + last_three


def format_inr(value_eur: float) -> str:
    return f"INR {format_indian_number(float(value_eur) * EUR_TO_INR_RATE)}"


def format_dual_currency(value_eur: float) -> str:
    return f"{format_eur(value_eur)} / {format_inr(value_eur)}"


def valuation_range(prediction: pd.Series) -> tuple[float, float]:
    anchored_prediction = max(float(prediction["ensemble_prediction"]), 0.0)
    spread = max(anchored_prediction * (1 - float(prediction["prediction_confidence"]) / 100.0) * 0.35, 75000.0)
    low = max(anchored_prediction - spread, 0.0)
    high = max(anchored_prediction + spread, 0.0)
    return low, high


def value_label(delta: float) -> tuple[str, str]:
    if delta >= 1_000_000:
        return "Upside call", "Model sees clear headroom above the current market tag."
    if delta >= 0:
        return "Fair-to-upside", "The profile looks slightly underpriced or fairly set today."
    return "Protect value", "Model reads the profile as expensive relative to current signals."


def driver_cards(metrics: dict) -> list[tuple[str, str, str]]:
    return [
        ("AI Signal Stack", f"{metrics['feature_count']} engineered signals", "Performance, contract, sentiment, mobility, and risk layers are fused into one valuation engine."),
        ("Model Core", f"{int(metrics['ensemble_weight_xgb'] * 100)}% XGBoost / {int(metrics['ensemble_weight_lstm'] * 100)}% LSTM", "The ensemble automatically leans toward the stronger signal source on this dataset."),
        ("Dataset Reach", f"{metrics['full_engineered_rows']} transfer rows", "Historical transfer, sentiment, and contextual metadata are available for simulation and benchmarking."),
    ]


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def harmonize_interactive_inputs(base_row: pd.DataFrame, input_row: pd.DataFrame, injuries: int, sentiment: float) -> pd.DataFrame:
    adjusted = input_row.copy()
    base = base_row.iloc[0]

    base_injuries = float(base["total_injuries"])
    base_days = float(base["total_days_missed"])
    base_days_per_injury = base_days / max(base_injuries, 1.0) if base_days > 0 else 18.0
    adjusted["total_days_missed"] = 0.0 if injuries <= 0 else round(injuries * base_days_per_injury, 1)

    base_sentiment = float(base["avg_sentiment_3m"])
    sentiment_delta = sentiment - base_sentiment
    adjusted["positive_sentiment_ratio"] = _clip(float(base["positive_sentiment_ratio"]) + sentiment_delta * 0.40, 0.05, 0.95)
    adjusted["negative_sentiment_ratio"] = _clip(float(base["negative_sentiment_ratio"]) - sentiment_delta * 0.30, 0.01, 0.90)
    adjusted["peak_sentiment"] = _clip(max(sentiment, float(base["peak_sentiment"]) + sentiment_delta * 0.55), 0.0, 1.0)
    adjusted["lowest_sentiment"] = _clip(min(sentiment, float(base["lowest_sentiment"]) + sentiment_delta * 0.45), 0.0, 1.0)
    adjusted["sentiment_trend"] = _clip(float(base["sentiment_trend"]) + sentiment_delta * 0.12, -1.0, 1.0)
    adjusted["sentiment_volatility"] = _clip(float(base["sentiment_volatility"]) - abs(sentiment_delta) * 0.04, 0.0, 1.0)
    return adjusted


def apply_interactive_performance_guard(
    prediction: pd.Series,
    baseline_prediction: pd.Series,
    base_row: pd.DataFrame,
    goals: int,
    assists: int,
) -> pd.Series:
    adjusted = prediction.copy()
    base_goals = int(base_row.at[0, "total_goals"])
    base_assists = int(base_row.at[0, "total_assists"])
    goal_delta = goals - base_goals
    assist_delta = assists - base_assists
    base_contributions = base_goals + base_assists
    current_contributions = goals + assists
    contribution_delta = current_contributions - base_contributions

    if goal_delta == 0 and assist_delta == 0:
        adjusted["ensemble_prediction"] = max(float(adjusted["ensemble_prediction"]), 0.0)
        adjusted["prediction_delta_vs_current_value"] = adjusted["ensemble_prediction"] - adjusted["current_market_value"]
        return adjusted

    market_anchor = max(float(base_row.at[0, "current_market_value"]), 1_000_000.0)
    goal_unit = max(market_anchor * 0.0015, 20_000.0)
    assist_unit = max(market_anchor * 0.0010, 15_000.0)
    contribution_unit = max(market_anchor * 0.0013, 18_000.0)
    directional_target = max(
        0.0,
        float(baseline_prediction["ensemble_prediction"])
        + goal_delta * goal_unit
        + assist_delta * assist_unit,
    )

    raw_model_value = max(float(adjusted["ensemble_prediction"]), 0.0)
    if goal_delta >= 0 and assist_delta >= 0:
        adjusted["ensemble_prediction"] = max(raw_model_value, directional_target)
    elif goal_delta <= 0 and assist_delta <= 0:
        adjusted["ensemble_prediction"] = min(raw_model_value, directional_target * 0.65 + raw_model_value * 0.35)
    else:
        if directional_target >= float(baseline_prediction["ensemble_prediction"]):
            adjusted["ensemble_prediction"] = max(raw_model_value, directional_target)
        else:
            adjusted["ensemble_prediction"] = min(raw_model_value, directional_target * 0.65 + raw_model_value * 0.35)

    if contribution_delta > 0:
        contribution_floor = max(
            0.0,
            float(baseline_prediction["ensemble_prediction"]) + contribution_delta * contribution_unit,
        )
        adjusted["ensemble_prediction"] = max(float(adjusted["ensemble_prediction"]), contribution_floor)

    adjusted["ensemble_prediction"] = max(float(adjusted["ensemble_prediction"]), 0.0)
    adjusted["prediction_delta_vs_current_value"] = adjusted["ensemble_prediction"] - adjusted["current_market_value"]
    return adjusted


def render_filterable_table(
    dataframe: pd.DataFrame,
    key_prefix: str,
    default_visible_columns: list[str] | None = None,
    height: int = 420,
) -> None:
    if dataframe.empty:
        st.info("No rows are available for this table.")
        return

    all_columns = dataframe.columns.tolist()
    visible_columns = [column for column in (default_visible_columns or all_columns) if column in all_columns]
    if not visible_columns:
        visible_columns = all_columns[: min(len(all_columns), 7)]
    display_df = dataframe[visible_columns].copy().where(pd.notna(dataframe[visible_columns]), None)

    column_meta: list[dict[str, object]] = []
    for column in visible_columns:
        series = display_df[column]
        options: list[str] = []
        if not pd.api.types.is_numeric_dtype(series):
            unique_values = sorted(
                {
                    str(value).strip()
                    for value in series.dropna().astype(str)
                    if str(value).strip() and str(value).strip().lower() != "nan"
                }
            )
            if 0 < len(unique_values) <= 18:
                options = unique_values
        column_meta.append(
            {
                "name": column,
                "label": column,
                "numeric": bool(pd.api.types.is_numeric_dtype(series)),
                "options": options,
                "market_mode": column == "current_market_value_eur",
            }
        )

    rows_json = json.dumps(display_df.to_dict(orient="records"))
    meta_json = json.dumps(column_meta)
    table_id = f"{key_prefix}_interactive_table"
    component_height = max(height + 170, 560)

    html_block = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8" />
      <style>
        :root {
          color-scheme: dark;
          --bg: #091821;
          --panel: #0c1d27;
          --panel-2: #112632;
          --line: rgba(255, 255, 255, 0.10);
          --line-strong: rgba(141, 245, 163, 0.24);
          --text: #eefcf8;
          --muted: #9fc0b7;
          --accent: #6ef1dc;
          --accent-2: #8dc8ff;
          --header: #173543;
          --row: #0d2430;
          --row-alt: #112b38;
        }
        * {
          box-sizing: border-box;
        }
        body {
          margin: 0;
          font-family: "Segoe UI", Arial, sans-serif;
          background: transparent;
          color: var(--text);
        }
        .table-shell {
          border: 1px solid var(--line);
          border-radius: 20px;
          background: linear-gradient(180deg, rgba(10, 25, 33, 0.98), rgba(8, 22, 30, 0.98));
          padding: 14px;
        }
        .toolbar {
          display: flex;
          align-items: center;
          gap: 10px;
          margin-bottom: 12px;
        }
        .search-input {
          width: 100%;
          border-radius: 14px;
          border: 1px solid rgba(173, 231, 220, 0.20);
          background: rgba(255, 255, 255, 0.05);
          color: var(--text);
          padding: 12px 14px;
          font-size: 14px;
          font-weight: 600;
          outline: none;
        }
        .search-input::placeholder {
          color: #bdd7cf;
        }
        .clear-button {
          border: 1px solid rgba(173, 231, 220, 0.22);
          background: linear-gradient(135deg, #1f6469, #2f7880 48%, #4d8f97);
          color: #ffffff;
          border-radius: 14px;
          padding: 12px 14px;
          font-size: 13px;
          font-weight: 700;
          cursor: pointer;
          white-space: nowrap;
        }
        .count-line {
          color: var(--muted);
          font-size: 13px;
          margin-bottom: 10px;
        }
        .table-wrap {
          border: 1px solid var(--line);
          border-radius: 18px;
          overflow: auto;
          max-height: __TABLE_HEIGHT__px;
          background: rgba(6, 18, 25, 0.84);
        }
        table {
          width: 100%;
          min-width: 100%;
          border-collapse: separate;
          border-spacing: 0;
        }
        thead th {
          position: sticky;
          top: 0;
          z-index: 4;
          background: var(--header);
          border-bottom: 1px solid rgba(255, 255, 255, 0.12);
          border-right: 1px solid rgba(255, 255, 255, 0.08);
          padding: 0;
          text-align: left;
          min-width: 160px;
        }
        thead th:last-child,
        tbody td:last-child {
          border-right: none;
        }
        .head-bar {
          position: relative;
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 8px;
          padding: 11px 12px;
        }
        .head-label {
          display: inline-flex;
          align-items: center;
          gap: 6px;
          font-size: 13px;
          font-weight: 700;
          color: #f2fffb;
          line-height: 1.35;
          word-break: break-word;
        }
        .sort-flag {
          color: var(--accent);
          font-size: 11px;
          letter-spacing: 0.04em;
        }
        .head-button {
          width: 28px;
          min-width: 28px;
          height: 28px;
          border: 1px solid rgba(255, 255, 255, 0.10);
          border-radius: 9px;
          background: rgba(255, 255, 255, 0.06);
          color: #ffffff;
          font-size: 13px;
          cursor: pointer;
        }
        .head-button:hover {
          background: rgba(110, 241, 220, 0.16);
          border-color: rgba(110, 241, 220, 0.26);
        }
        .column-menu {
          position: absolute;
          top: calc(100% - 2px);
          right: 10px;
          width: 240px;
          display: none;
          padding: 12px;
          border-radius: 16px;
          border: 1px solid rgba(255, 255, 255, 0.12);
          background: rgba(7, 20, 28, 0.98);
          box-shadow: 0 20px 36px rgba(0, 0, 0, 0.36);
        }
        .column-menu.open {
          display: block;
        }
        .menu-title {
          color: var(--text);
          font-size: 13px;
          font-weight: 700;
          margin-bottom: 10px;
        }
        .menu-label {
          display: block;
          color: var(--muted);
          font-size: 11px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          margin: 10px 0 6px;
        }
        .menu-group {
          display: grid;
          gap: 8px;
        }
        .menu-row {
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 8px;
        }
        .menu-button,
        .menu-input,
        .menu-select {
          width: 100%;
          border-radius: 12px;
          border: 1px solid rgba(255, 255, 255, 0.10);
          background: rgba(255, 255, 255, 0.05);
          color: #f2fffb;
          padding: 10px 12px;
          font-size: 12px;
          font-weight: 600;
          outline: none;
        }
        .menu-button {
          cursor: pointer;
        }
        .menu-button:hover {
          background: rgba(110, 241, 220, 0.16);
          border-color: rgba(110, 241, 220, 0.24);
        }
        tbody td {
          padding: 11px 12px;
          border-bottom: 1px solid rgba(255, 255, 255, 0.08);
          border-right: 1px solid rgba(255, 255, 255, 0.05);
          background: var(--row);
          color: #eefcf8;
          font-size: 13px;
          vertical-align: top;
        }
        tbody tr:nth-child(even) td {
          background: var(--row-alt);
        }
        tbody tr:hover td {
          background: rgba(26, 66, 82, 0.96);
        }
        .numeric-cell {
          text-align: right;
          font-variant-numeric: tabular-nums;
        }
        .empty-row {
          text-align: center;
          color: var(--muted);
          padding: 16px;
        }
      </style>
    </head>
    <body>
      <div class="table-shell" id="__TABLE_ID__">
        <div class="toolbar">
          <input id="__TABLE_ID___search" class="search-input" type="text" placeholder="Search player name, club, career stage, or any table value" />
          <button id="__TABLE_ID___clear" class="clear-button" type="button">Clear</button>
        </div>
        <div class="count-line" id="__TABLE_ID___count"></div>
        <div class="table-wrap">
          <table>
            <thead id="__TABLE_ID___thead"></thead>
            <tbody id="__TABLE_ID___tbody"></tbody>
          </table>
        </div>
      </div>

      <script>
        const rows = __ROWS_JSON__;
        const columns = __META_JSON__;
        const root = document.getElementById("__TABLE_ID__");
        const searchInput = document.getElementById("__TABLE_ID___search");
        const clearButton = document.getElementById("__TABLE_ID___clear");
        const countLine = document.getElementById("__TABLE_ID___count");
        const head = document.getElementById("__TABLE_ID___thead");
        const body = document.getElementById("__TABLE_ID___tbody");

        const state = {
          search: "",
          sortColumn: "",
          sortDirection: "asc",
          filters: {}
        };

        columns.forEach((column) => {
          state.filters[column.name] = {
            exact: "",
            contains: "",
            min: "",
            max: "",
            marketMode: "All rows"
          };
        });

        function escapeHtml(value) {
          return String(value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
        }

        function formatValue(value) {
          if (value === null || value === undefined || value === "") {
            return "";
          }
          if (typeof value === "number") {
            if (Number.isInteger(value)) {
              return new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 }).format(value);
            }
            return new Intl.NumberFormat("en-US", {
              minimumFractionDigits: 0,
              maximumFractionDigits: 4
            }).format(value);
          }
          return escapeHtml(String(value));
        }

        function closeMenus() {
          root.querySelectorAll(".column-menu.open").forEach((menu) => menu.classList.remove("open"));
        }

        function compareValues(left, right, numeric) {
          if (left === null || left === undefined || left === "") return 1;
          if (right === null || right === undefined || right === "") return -1;
          if (numeric) {
            return Number(left) - Number(right);
          }
          return String(left).localeCompare(String(right), undefined, { sensitivity: "base" });
        }

        function filteredRows() {
          let active = rows.filter((row) => {
            if (state.search) {
              const matchesSearch = columns.some((column) => {
                const raw = row[column.name];
                return String(raw ?? "").toLowerCase().includes(state.search);
              });
              if (!matchesSearch) {
                return false;
              }
            }

            for (const column of columns) {
              const filter = state.filters[column.name];
              const raw = row[column.name];

              if (column.numeric) {
                const numericValue = Number(raw ?? 0);
                if (column.market_mode) {
                  if (filter.marketMode === "Known values only" && numericValue <= 0) {
                    return false;
                  }
                  if (filter.marketMode === "Zero values only" && numericValue > 0) {
                    return false;
                  }
                }
                if (filter.min !== "" && numericValue < Number(filter.min)) {
                  return false;
                }
                if (filter.max !== "" && numericValue > Number(filter.max)) {
                  return false;
                }
                continue;
              }

              const textValue = String(raw ?? "");
              if (filter.exact && textValue !== filter.exact) {
                return false;
              }
              if (filter.contains && !textValue.toLowerCase().includes(filter.contains.toLowerCase())) {
                return false;
              }
            }

            return true;
          });

          if (state.sortColumn) {
            const columnMeta = columns.find((column) => column.name === state.sortColumn);
            active = active.slice().sort((left, right) => {
              const comparison = compareValues(left[state.sortColumn], right[state.sortColumn], Boolean(columnMeta && columnMeta.numeric));
              return state.sortDirection === "asc" ? comparison : -comparison;
            });
          }

          return active;
        }

        function renderRows() {
          const active = filteredRows();
          countLine.textContent = "Showing " + active.length.toLocaleString("en-US") + " of " + rows.length.toLocaleString("en-US") + " rows.";

          if (!active.length) {
            body.innerHTML = '<tr><td class="empty-row" colspan="' + columns.length + '">No rows match the current search or filters.</td></tr>';
            return;
          }

          body.innerHTML = active.map((row) => {
            return "<tr>" + columns.map((column) => {
              const className = column.numeric ? "numeric-cell" : "";
              return '<td class="' + className + '">' + formatValue(row[column.name]) + "</td>";
            }).join("") + "</tr>";
          }).join("");
        }

        function renderHeader() {
          const headerHtml = columns.map((column) => {
            const filter = state.filters[column.name];
            const sortFlag = state.sortColumn === column.name ? (state.sortDirection === "asc" ? "ASC" : "DESC") : "";
            const exactOptions = Array.isArray(column.options)
              ? column.options.map((option) => '<option value="' + escapeHtml(option) + '"' + (filter.exact === option ? " selected" : "") + ">" + escapeHtml(option) + "</option>").join("")
              : "";
            const menuBody = column.numeric
              ? `
                  <label class="menu-label">Sort</label>
                  <div class="menu-row">
                    <button class="menu-button" type="button" data-action="sort-asc">Ascending</button>
                    <button class="menu-button" type="button" data-action="sort-desc">Descending</button>
                  </div>
                  <button class="menu-button" type="button" data-action="sort-clear">Clear sort</button>
                  ${column.market_mode ? `
                    <label class="menu-label">Value mode</label>
                    <select class="menu-select" data-action="market-mode">
                      <option${filter.marketMode === "All rows" ? " selected" : ""}>All rows</option>
                      <option${filter.marketMode === "Known values only" ? " selected" : ""}>Known values only</option>
                      <option${filter.marketMode === "Zero values only" ? " selected" : ""}>Zero values only</option>
                    </select>
                  ` : ""}
                  <label class="menu-label">Range</label>
                  <div class="menu-row">
                    <input class="menu-input" type="number" step="any" data-action="min" placeholder="Min" value="${escapeHtml(filter.min)}" />
                    <input class="menu-input" type="number" step="any" data-action="max" placeholder="Max" value="${escapeHtml(filter.max)}" />
                  </div>
                  <button class="menu-button" type="button" data-action="clear-filter">Clear filter</button>
                `
              : `
                  <label class="menu-label">Sort</label>
                  <div class="menu-row">
                    <button class="menu-button" type="button" data-action="sort-asc">Ascending</button>
                    <button class="menu-button" type="button" data-action="sort-desc">Descending</button>
                  </div>
                  <button class="menu-button" type="button" data-action="sort-clear">Clear sort</button>
                  ${column.options && column.options.length ? `
                    <label class="menu-label">Exact value</label>
                    <select class="menu-select" data-action="exact">
                      <option value="">All values</option>
                      ${exactOptions}
                    </select>
                  ` : `
                    <label class="menu-label">Contains</label>
                    <input class="menu-input" type="text" data-action="contains" placeholder="Type to filter" value="${escapeHtml(filter.contains)}" />
                  `}
                  <button class="menu-button" type="button" data-action="clear-filter">Clear filter</button>
                `;

            return `
              <th>
                <div class="head-bar">
                  <span class="head-label">${escapeHtml(column.label)}${sortFlag ? `<span class="sort-flag">${sortFlag}</span>` : ""}</span>
                  <button class="head-button" type="button" data-column="${escapeHtml(column.name)}">&#9662;</button>
                  <div class="column-menu" data-menu="${escapeHtml(column.name)}">
                    <div class="menu-title">${escapeHtml(column.label)}</div>
                    <div class="menu-group">${menuBody}</div>
                  </div>
                </div>
              </th>
            `;
          }).join("");

          head.innerHTML = "<tr>" + headerHtml + "</tr>";

          root.querySelectorAll(".head-button").forEach((button) => {
            button.addEventListener("click", (event) => {
              event.stopPropagation();
              const menu = button.parentElement.querySelector(".column-menu");
              const alreadyOpen = menu.classList.contains("open");
              closeMenus();
              if (!alreadyOpen) {
                menu.classList.add("open");
              }
            });
          });

          root.querySelectorAll(".column-menu").forEach((menu) => {
            menu.addEventListener("click", (event) => event.stopPropagation());
            const columnName = menu.getAttribute("data-menu");
            menu.querySelectorAll("[data-action]").forEach((control) => {
              const action = control.getAttribute("data-action");
              const apply = () => {
                if (action === "sort-asc") {
                  state.sortColumn = columnName;
                  state.sortDirection = "asc";
                } else if (action === "sort-desc") {
                  state.sortColumn = columnName;
                  state.sortDirection = "desc";
                } else if (action === "sort-clear") {
                  if (state.sortColumn === columnName) {
                    state.sortColumn = "";
                  }
                } else if (action === "min") {
                  state.filters[columnName].min = control.value;
                } else if (action === "max") {
                  state.filters[columnName].max = control.value;
                } else if (action === "contains") {
                  state.filters[columnName].contains = control.value;
                } else if (action === "exact") {
                  state.filters[columnName].exact = control.value;
                } else if (action === "market-mode") {
                  state.filters[columnName].marketMode = control.value;
                } else if (action === "clear-filter") {
                  state.filters[columnName] = { exact: "", contains: "", min: "", max: "", marketMode: "All rows" };
                }
                renderHeader();
                renderRows();
              };

              if (control.tagName === "BUTTON") {
                control.addEventListener("click", apply);
              } else if (control.tagName === "SELECT") {
                control.addEventListener("change", apply);
              } else {
                control.addEventListener("input", apply);
              }
            });
          });
        }

        searchInput.addEventListener("input", (event) => {
          state.search = event.target.value.trim().toLowerCase();
          renderRows();
        });

        clearButton.addEventListener("click", () => {
          state.search = "";
          state.sortColumn = "";
          state.sortDirection = "asc";
          columns.forEach((column) => {
            state.filters[column.name] = { exact: "", contains: "", min: "", max: "", marketMode: "All rows" };
          });
          searchInput.value = "";
          renderHeader();
          renderRows();
        });

        document.addEventListener("click", () => closeMenus());
        renderHeader();
        renderRows();
      </script>
    </body>
    </html>
    """

    html_block = (
        html_block.replace("__ROWS_JSON__", rows_json)
        .replace("__META_JSON__", meta_json)
        .replace("__TABLE_HEIGHT__", str(height))
        .replace("__TABLE_ID__", table_id)
    )
    components.html(html_block, height=component_height, scrolling=False)


def render_landing() -> None:
    hero_classes = "hero-shell"
    st.markdown(
        f"""
        <div class="{hero_classes}">
            <div class="landing-grid">
                <div class="landing-copy">
                    <div class="eyebrow">AI Transfer Intelligence</div>
                    <div class="landing-title">Predict the next <span>transfer swing</span> before the market moves.</div>
                    <div class="landing-text">
                        TransferIQ blends engineered football signals, sentiment dynamics, injury context, and sequential learning into one flexible valuation cockpit.
                        Explore player profiles on laptop or mobile, run bulk simulations, and surface the story behind every price signal.
                    </div>
                    <div class="landing-pills">
                        <div class="landing-pill">Real LSTM + XGBoost Ensemble</div>
                        <div class="landing-pill">Responsive Scouting Dashboard</div>
                        <div class="landing-pill">AI Briefing + Visual Diagnostics</div>
                    </div>
                </div>
                <div class="kick-stage">
                    <div class="pitch-lines"></div>
                    <div class="ai-orbit"></div>
                    <div class="player-wrap">
                        <div class="arc"></div>
                        <div class="ball-ghost"></div>
                        <div class="ball"></div>
                        <div class="player-head"></div>
                        <div class="player-body"></div>
                        <div class="arm arm-left"></div>
                        <div class="arm arm-right"></div>
                        <div class="leg leg-left"></div>
                        <div class="leg leg-right"></div>
                        <div class="foot foot-left"></div>
                        <div class="foot foot-right"></div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    go_class = "landing-go"
    st.markdown(f'<div class="{go_class}">', unsafe_allow_html=True)
    if st.button("Go", width="stretch"):
        st.session_state["entered"] = True
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def render_metric_cards(metrics: dict) -> None:
    cols = st.columns(3)
    values = [
        ("Ensemble RMSE", f"EUR {metrics['ensemble_metrics']['rmse']:,.0f}", "Holdout error after weighting tree + sequence signals."),
        ("Ensemble R²", f"{metrics['ensemble_metrics']['r2']:.3f}", "Explained variance on the evaluation split."),
        ("Feature Space", f"{metrics['feature_count']}", "Signals used by the live inference engine."),
    ]
    for col, (label, value, text) in zip(cols, values):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-sub">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_prediction_workspace(predictor: TransferValuePredictor, library: pd.DataFrame, raw_profiles: pd.DataFrame) -> None:
    player_options = library["player_name"].dropna().astype(str).tolist()
    if not player_options:
        st.warning("No players are available in the prediction library.")
        return

    if st.session_state.get("selected_player") not in player_options:
        st.session_state["selected_player"] = player_options[0]

    chosen = st.selectbox("Search player", player_options, key="selected_player")
    matches = library.loc[library["player_name"].astype(str) == str(chosen)]
    if matches.empty:
        st.session_state["selected_player"] = player_options[0]
        matches = library.loc[library["player_name"].astype(str) == st.session_state["selected_player"]]

    selected_record = matches.iloc[0]
    base_row = build_row_from_library(selected_record, raw_profiles=raw_profiles)
    baseline_prediction = predictor.predict(base_row).iloc[0]
    st.markdown(
        f"""
        <div class="player-banner">
            <div>
                <div class="player-name">{selected_record['player_name']}</div>
                <div class="player-meta">
                    Current club: {selected_record.get('current_club_name', 'Unknown')}<br/>
                    Route: {selected_record.get('from_team_name', 'Previous Club')} to {selected_record.get('to_team_name', selected_record.get('current_club_name', 'Unknown'))}
                </div>
                <div class="pill-row">
                    <div class="mini-pill">Current value {format_dual_currency(selected_record.get('current_market_value', 0))}</div>
                    <div class="mini-pill">{int(selected_record.get('total_goals', 0))} goals</div>
                    <div class="mini-pill">{int(selected_record.get('total_assists', 0))} assists</div>
                    <div class="mini-pill">{selected_record.get('career_stage', 'Scouted profile')}</div>
                </div>
            </div>
            <div class="guess-strip">
                <div class="guess-card">
                    <div class="guess-title">Player Snapshot</div>
                    <div class="guess-value">{selected_record.get('sentiment_band', 'Balanced')}</div>
                    <div class="guess-help">Sentiment band with {selected_record.get('injury_risk_category', 'Unknown')} injury exposure and {int(selected_record.get('competition_score', 0))} competition score.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        st.markdown('<div class="spotlight-card">', unsafe_allow_html=True)
        st.subheader("Live AI Valuation Workspace")
        st.caption("Adjust the player profile and get an individual market guess with a confidence-aware range.")
        base_goals = int(base_row.at[0, "total_goals"])
        base_assists = int(base_row.at[0, "total_assists"])
        base_injuries = int(base_row.at[0, "total_injuries"])
        goals = st.slider("Goals", 0, max(55, base_goals + 15), base_goals)
        assists = st.slider("Assists", 0, max(30, base_assists + 15), base_assists)
        market_value = st.number_input("Current market value", min_value=0.0, value=float(base_row.at[0, "current_market_value"]), step=50000.0)
        injuries = st.slider("Injury count", 0, max(12, base_injuries + 6), base_injuries)
        sentiment = st.slider("Average sentiment", 0.0, 1.0, float(base_row.at[0, "avg_sentiment_3m"]), step=0.01)
        mentions = st.number_input("Monthly mentions", min_value=0.0, value=float(base_row.at[0, "avg_monthly_mentions"]), step=100.0)
        engagement = st.slider("Engagement rate", 1.0, 25.0, float(base_row.at[0, "engagement_rate"]), step=0.5)
        st.markdown("</div>", unsafe_allow_html=True)

        input_row = base_row.copy()
        input_row["total_goals"] = goals
        input_row["total_assists"] = assists
        input_row["current_market_value"] = market_value
        input_row["total_injuries"] = injuries
        input_row["avg_sentiment_3m"] = sentiment
        input_row["avg_monthly_mentions"] = mentions
        input_row["engagement_rate"] = engagement
        input_row = harmonize_interactive_inputs(base_row, input_row, injuries, sentiment)

        prediction = predictor.predict(input_row).iloc[0]
        prediction = apply_interactive_performance_guard(
            prediction,
            baseline_prediction,
            base_row,
            goals,
            assists,
        )
        low_guess, high_guess = valuation_range(prediction)
        call_title, call_text = value_label(float(prediction["prediction_delta_vs_current_value"]))

        compare_df = pd.DataFrame(
            {
                "Layer": ["Current Value", "XGBoost", "LSTM", "Ensemble"],
                "EUR": [
                    prediction["current_market_value"],
                    prediction["xgb_prediction"],
                    prediction["lstm_prediction"],
                    prediction["ensemble_prediction"],
                ],
            }
        )
        waterfall = px.bar(
            compare_df,
            x="Layer",
            y="EUR",
            color="Layer",
            color_discrete_sequence=["#5876ff", "#53d6c7", "#ffd16d", "#8df5a3"],
        )
        waterfall.update_layout(height=330, margin=dict(l=12, r=12, t=10, b=12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(waterfall, width="stretch")

    with right:
        st.markdown('<div class="insight-card">', unsafe_allow_html=True)
        st.subheader("AI Scout Brief")
        st.metric(
            "Predicted Transfer Value",
            format_dual_currency(prediction["ensemble_prediction"]),
            delta=format_dual_currency(prediction["prediction_delta_vs_current_value"]),
        )
        st.metric("Confidence", f"{prediction['prediction_confidence']:.0f}%")
        st.caption(f"Approx. INR conversion uses 1 EUR = INR {EUR_TO_INR_RATE:.1f} as of {EUR_TO_INR_RATE_DATE}.")
        st.markdown(
            f"""
            <div class="guess-strip" style="margin:0.9rem 0 1rem;">
                <div class="guess-card">
                    <div class="guess-title">Individual Guess</div>
                    <div class="guess-value">{format_dual_currency(low_guess)} to {format_dual_currency(high_guess)}</div>
                    <div class="guess-help">{call_title}: {call_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(ai_brief(prediction))
        st.markdown(
            f"""
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:0.8rem;">
                <div class="metric-card">
                    <div class="metric-label">Career Stage</div>
                    <div class="metric-value" style="font-size:1.35rem;">{prediction['career_stage']}</div>
                    <div class="metric-sub">Age proxy {prediction['age_proxy']:.0f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sentiment Mode</div>
                    <div class="metric-value" style="font-size:1.35rem;">{prediction['sentiment_band']}</div>
                    <div class="metric-sub">Public perception is part of the valuation mix.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        radar = go.Figure()
        radar.add_trace(
            go.Scatterpolar(
                r=[
                    min(goals / 25, 1.0) * 100,
                    min(assists / 15, 1.0) * 100,
                    sentiment * 100,
                    min(mentions / 3000, 1.0) * 100,
                    max(0, 100 - injuries * 8),
                ],
                theta=["Finishing", "Creation", "Sentiment", "Buzz", "Durability"],
                fill="toself",
                line=dict(color="#6ef1dc"),
            )
        )
        radar.update_layout(
            height=330,
            margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.12)"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            ),
            showlegend=False,
        )
        st.plotly_chart(radar, width="stretch")


def render_bulk_lab(predictor: TransferValuePredictor) -> None:
    st.markdown('<div class="spotlight-card">', unsafe_allow_html=True)
    st.subheader("Bulk Simulation Lab")
    st.caption("Upload raw-schema or already-engineered player rows. The app will preview a sample, but predictions run on the full uploaded dataset.")
    uploaded = st.file_uploader("Upload bulk CSV", type=["csv"])
    if uploaded is not None:
        bulk_df = pd.read_csv(uploaded)
    else:
        bulk_df = pd.read_csv(RAW_DATA_PATH).head(20)

    st.caption(f"Loaded rows: {len(bulk_df):,} | Preview shown below: {min(len(bulk_df), 8):,}")
    st.dataframe(bulk_df.head(8), width="stretch")
    if st.button("Run AI Bulk Scan", width="stretch"):
        bulk_results = predictor.predict(bulk_df)
        ranked_results = bulk_results.sort_values("ensemble_prediction", ascending=False).reset_index(drop=True)
        leader = ranked_results.head(15).copy()
        top_display_columns = [
            "player_name",
            "current_market_value",
            "ensemble_prediction",
            "prediction_confidence",
            "prediction_delta_vs_current_value",
        ]
        table_columns = [
            "player_name",
            "current_club_name",
            "current_market_value",
            "ensemble_prediction",
            "prediction_confidence",
            "prediction_delta_vs_current_value",
            "total_goals",
            "total_assists",
            "total_injuries",
            "avg_sentiment_3m",
            "career_stage",
            "sentiment_band",
            "injury_risk_category",
            "transfer_window",
            "competition_score",
            "age_proxy",
        ]
        available_top_columns = [col for col in top_display_columns if col in ranked_results.columns]
        available_table_columns = [col for col in table_columns if col in ranked_results.columns]
        top15_display = leader[available_top_columns].copy()
        full_display = ranked_results[available_table_columns].copy()
        top15_display.insert(0, "rank", range(1, len(top15_display) + 1))
        full_display.insert(0, "rank", range(1, len(full_display) + 1))
        rename_map = {
            "current_market_value": "current_market_value_eur",
            "ensemble_prediction": "predicted_transfer_fee_eur",
            "prediction_confidence": "prediction_confidence_pct",
            "prediction_delta_vs_current_value": "delta_vs_current_value_eur",
        }
        top15_display = top15_display.rename(columns=rename_map)
        full_display = full_display.rename(columns=rename_map)
        st.caption(f"Predictions completed for {len(ranked_results):,} players.")
        zero_market_rows = int(pd.to_numeric(ranked_results["current_market_value"], errors="coerce").fillna(0.0).eq(0).sum())
        if zero_market_rows:
            st.caption(
                f"{zero_market_rows:,} rows show `current_market_value = 0` because the source dataset has missing or unavailable market values stored as zero."
            )
        st.markdown("**Top 15 Predicted Players By Transfer Fee**")
        st.caption("Players are ranked in descending order by predicted transfer fee, not by original dataset index.")
        st.dataframe(
            top15_display,
            width="stretch",
            hide_index=True,
        )
        chart = px.bar(
            leader.head(10),
            x="player_name",
            y="ensemble_prediction",
            color="prediction_confidence",
            color_continuous_scale="Tealgrn",
        )
        chart.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(chart, width="stretch")
        csv_payload = ranked_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Full Predictions CSV",
            data=csv_payload,
            file_name="bulk_transfer_predictions.csv",
            mime="text/csv",
            width="stretch",
        )
        with st.expander("Show all predicted players"):
            render_filterable_table(
                full_display,
                key_prefix="bulk_predictions",
                default_visible_columns=[
                    "rank",
                    "player_name",
                    "current_club_name",
                    "current_market_value_eur",
                    "predicted_transfer_fee_eur",
                    "prediction_confidence_pct",
                    "delta_vs_current_value_eur",
                ],
                height=420,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def render_ai_insights(metrics: dict, test_predictions: pd.DataFrame) -> None:
    left, right = st.columns([1.05, 0.95], gap="large")
    with left:
        top_features = pd.DataFrame(metrics["top_features"])
        feature_chart = px.bar(
            top_features.head(12),
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Mint",
        )
        feature_chart.update_layout(
            height=430,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis={"categoryorder": "total ascending"},
        )
        st.plotly_chart(feature_chart, width="stretch")
    with right:
        eval_df = test_predictions.dropna(subset=["ensemble_prediction"]).copy()
        eval_df["absolute_error"] = (eval_df["target_value"] - eval_df["ensemble_prediction"]).abs()
        scatter = px.scatter(
            eval_df,
            x="target_value",
            y="ensemble_prediction",
            hover_name="player_name",
            size="absolute_error",
            color="absolute_error",
            color_continuous_scale="Turbo",
        )
        scatter.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(scatter, width="stretch")

    cards = st.columns(3)
    for col, (title, value, text) in zip(cards, driver_cards(metrics)):
        with col:
            st.markdown(
                f"""
                <div class="insight-card">
                    <div class="metric-label">{title}</div>
                    <div class="metric-value" style="font-size:1.4rem;">{value}</div>
                    <div class="metric-sub">{text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


init_state()

if not st.session_state["entered"]:
    render_landing()
else:
    predictor = load_predictor()
    library = load_library()
    raw_profiles = load_raw_profiles()
    metrics = load_metrics()
    test_predictions = load_test_predictions()
    top_left, top_right = st.columns([0.8, 0.2])
    with top_left:
        st.markdown(
            """
            <div class="title-chip">TransferIQ Command Center</div>
            <div class="dashboard-head">AI-driven transfer valuation for scouts, analysts, and decision-makers.</div>
            <div class="subtitle">
                Move from discovery to decision using an interactive player lab, bulk simulation mode, and model explainability views designed to stay readable on both laptop and mobile screens.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        st.markdown("<div style='height: 1.35rem;'></div>", unsafe_allow_html=True)
        if st.button("Entrance", width="stretch"):
            st.session_state["entered"] = False
            st.rerun()

    render_metric_cards(metrics)
    tab_live, tab_bulk, tab_ai = st.tabs(["AI Player Lab", "Bulk Scan", "Model Intelligence"])
    with tab_live:
        render_prediction_workspace(predictor, library, raw_profiles)
    with tab_bulk:
        render_bulk_lab(predictor)
    with tab_ai:
        render_ai_insights(metrics, test_predictions)
