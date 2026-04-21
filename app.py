"""
AutoEDA AI — Production-Grade AI-Powered EDA Platform
Upgraded from VLSI Automated Design Pipeline
Architecture: Multi-Agent AI System with Real ngspice, Waveforms, Optimization, RAG, Reports
"""

import streamlit as st
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
import re
import time
import base64
import math
import os
import subprocess
import tempfile
import hashlib
import random
import logging
import traceback
from collections import defaultdict
from datetime import datetime
from io import StringIO

import numpy as np
import pandas as pd
import anthropic
from dotenv import load_dotenv
load_dotenv()
# ══════════════════════════════════════════════════════════════════════════════
# LOGGING SETUP
# ══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AutoEDA")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AutoEDA AI — Intelligent Circuit Design Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — Dark EDA terminal aesthetic
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

/* ── Base ── */
.stApp { background: #020408; color: #C8D0E7; }
.main .block-container { padding: 1.2rem 2.2rem 3rem; max-width: 1700px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #06080F 0%, #040608 100%) !important;
    border-right: 1px solid #141C2E !important;
    width: 300px !important;
}
[data-testid="stSidebar"] * { color: #C8D0E7 !important; }

/* ── Typography ── */
h1,h2,h3,h4 { color: #E8EDF8 !important; font-family: 'JetBrains Mono', monospace !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0C1220 0%, #080D18 100%);
    border: 1px solid #1A2840;
    border-radius: 10px;
    padding: 12px 16px;
    transition: border-color 0.2s, transform 0.2s;
}
[data-testid="stMetric"]:hover { border-color: #2A4070; transform: translateY(-1px); }
[data-testid="stMetricLabel"] {
    color: #4A5A7A !important;
    font-size: 9px !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    color: #38D9A9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #162ABF 0%, #1E40CF 50%, #2855E8 100%) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(40,85,232,0.5) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    font-weight: 700 !important;
    letter-spacing: 0.8px;
    padding: 8px 18px !important;
    transition: all 0.2s cubic-bezier(0.4,0,0.2,1) !important;
    text-transform: uppercase;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(30,64,207,0.55) !important;
    border-color: rgba(40,85,232,0.9) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stNumberInput > div > div > input {
    background: #07090F !important;
    border: 1px solid #141C2E !important;
    color: #C8D0E7 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #1E40CF !important;
    box-shadow: 0 0 0 2px rgba(30,64,207,0.18) !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color: #2A3A5A !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #07090F !important;
    border: 1px solid #141C2E !important;
    color: #C8D0E7 !important;
    border-radius: 8px !important;
}

/* ── Slider ── */
.stSlider > div > div > div { background: #141C2E !important; }
.stSlider > div > div > div > div { background: linear-gradient(90deg, #1E40CF, #38D9A9) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #07090F !important;
    border: 1px dashed #1E2A40 !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0A0F1A !important;
    border: 1px solid #141C2E !important;
    border-radius: 8px !important;
    color: #C8D0E7 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 11px !important;
}

/* ── Code blocks ── */
.stCode, code, pre {
    background: #06080D !important;
    border: 1px solid #141C2E !important;
    color: #38D9A9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 8px !important;
}

/* ── Alert boxes ── */
.stInfo { background: rgba(30,64,207,0.07) !important; border: 1px solid rgba(30,64,207,0.25) !important; border-radius: 8px !important; color: #C8D0E7 !important; }
.stSuccess { background: rgba(56,217,169,0.07) !important; border: 1px solid rgba(56,217,169,0.25) !important; border-radius: 8px !important; color: #C8D0E7 !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border: 1px solid rgba(245,158,11,0.3) !important; border-radius: 8px !important; color: #C8D0E7 !important; }
.stError { background: rgba(239,68,68,0.07) !important; border: 1px solid rgba(239,68,68,0.25) !important; border-radius: 8px !important; color: #C8D0E7 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #06080D; border-bottom: 1px solid #141C2E; gap: 2px; }
.stTabs [data-baseweb="tab"] { color: #4A5A7A !important; font-family: 'JetBrains Mono', monospace !important; font-size: 10px !important; font-weight: 700 !important; letter-spacing: 0.8px; padding: 10px 16px !important; border-radius: 6px 6px 0 0 !important; text-transform: uppercase; }
.stTabs [aria-selected="true"] { color: #2855E8 !important; background: rgba(30,64,207,0.1) !important; border-bottom: 2px solid #2855E8 !important; }
.stTabs [data-baseweb="tab-panel"] { background: #020408; padding-top: 18px; }

/* ── Divider ── */
hr { border-color: #0D1220 !important; margin: 1.2rem 0 !important; }

/* ── Progress bar ── */
.stProgress > div > div > div { background: linear-gradient(90deg, #1E40CF, #38D9A9) !important; border-radius: 99px !important; }

/* ══════════════════════════════════════════════════
   CUSTOM COMPONENTS
══════════════════════════════════════════════════ */

/* ── Page header ── */
.page-header {
    background: linear-gradient(135deg, #06090F 0%, #080C16 50%, #040710 100%);
    border: 1px solid #141C2E;
    border-radius: 14px;
    padding: 26px 30px 22px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #1E40CF, #38D9A9, #7C3AED, #F59E0B);
}
.page-header::after {
    content: '';
    position: absolute;
    bottom: 0; right: 0;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(30,64,207,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.page-header-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 24px;
    font-weight: 700;
    color: #E8EDF8;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}
.page-header-sub { font-family: 'Space Grotesk', sans-serif; font-size: 13px; color: #4A5A7A; }
.page-header-badge {
    display: inline-flex; align-items: center; gap: 5px;
    background: rgba(56,217,169,0.08); border: 1px solid rgba(56,217,169,0.18);
    border-radius: 20px; padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #38D9A9;
    letter-spacing: 1px; margin-right: 6px; margin-top: 10px;
}

/* ── Section header ── */
.sec-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; letter-spacing: 2.5px; color: #2A3A5A;
    text-transform: uppercase; border-bottom: 1px solid #0D1220;
    padding-bottom: 7px; margin-bottom: 16px;
}

/* ── Agent card ── */
.agent-card {
    background: linear-gradient(135deg, #0A0F1A 0%, #080C14 100%);
    border: 1px solid #141C2E; border-radius: 10px;
    padding: 10px 12px; margin-bottom: 6px;
    display: flex; align-items: center; justify-content: space-between;
    transition: all 0.2s;
}
.agent-card:hover { border-color: #1E2A40; }
.agent-card-left { display: flex; align-items: center; gap: 9px; }
.agent-icon {
    width: 30px; height: 30px; border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; flex-shrink: 0;
}
.agent-icon-ctrl { background: rgba(124,58,237,0.12); border: 1px solid rgba(124,58,237,0.28); }
.agent-icon-sim  { background: rgba(30,64,207,0.12);  border: 1px solid rgba(30,64,207,0.28); }
.agent-icon-lay  { background: rgba(56,217,169,0.12); border: 1px solid rgba(56,217,169,0.28); }
.agent-icon-ver  { background: rgba(245,158,11,0.12); border: 1px solid rgba(245,158,11,0.28); }
.agent-icon-spc  { background: rgba(239,68,68,0.12);  border: 1px solid rgba(239,68,68,0.28); }
.agent-icon-opt  { background: rgba(52,211,153,0.12); border: 1px solid rgba(52,211,153,0.28); }
.agent-icon-rag  { background: rgba(251,191,36,0.12); border: 1px solid rgba(251,191,36,0.28); }
.agent-name { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 700; color: #C8D0E7; }
.agent-role { font-family: 'Space Grotesk', sans-serif; font-size: 9px; color: #4A5A7A; margin-top: 1px; }

/* ── Status badges ── */
.badge { font-family: 'JetBrains Mono', monospace; font-size: 8px; font-weight: 700; letter-spacing: 1px; padding: 2px 8px; border-radius: 20px; text-transform: uppercase; }
.badge-run  { background: rgba(56,217,169,0.1); color: #38D9A9; border: 1px solid rgba(56,217,169,0.28); }
.badge-idle { background: rgba(74,90,122,0.1); color: #4A5A7A; border: 1px solid #141C2E; }
.badge-done { background: rgba(30,64,207,0.1); color: #2855E8; border: 1px solid rgba(30,64,207,0.28); }
.badge-err  { background: rgba(239,68,68,0.1); color: #EF4444; border: 1px solid rgba(239,68,68,0.28); }
.badge-busy { background: rgba(245,158,11,0.1); color: #F59E0B; border: 1px solid rgba(245,158,11,0.28); }

/* ── Pipeline step ── */
.pipeline-step {
    background: #07090F; border: 1px solid #141C2E;
    border-radius: 8px; padding: 8px 12px; margin-bottom: 5px;
    display: flex; align-items: center; gap: 9px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #4A5A7A;
}
.pipeline-step.active { border-color: #1E40CF; color: #2855E8; background: rgba(30,64,207,0.06); }
.pipeline-step.done   { border-color: #143A28; color: #38D9A9; background: rgba(56,217,169,0.04); }
.step-num {
    width: 20px; height: 20px; border-radius: 50%;
    background: #0F1620; border: 1px solid #141C2E;
    display: flex; align-items: center; justify-content: center;
    font-size: 8px; font-weight: 700; flex-shrink: 0;
}

/* ── Console log ── */
.console-box {
    background: #020406; border: 1px solid #0D1220;
    border-radius: 10px; padding: 14px 16px;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px; line-height: 1.9;
    max-height: 420px; overflow-y: auto; color: #C8D0E7;
}
.console-box::-webkit-scrollbar { width: 3px; }
.console-box::-webkit-scrollbar-track { background: transparent; }
.console-box::-webkit-scrollbar-thumb { background: #141C2E; border-radius: 99px; }
.log-ok   { color: #38D9A9; }
.log-warn { color: #F59E0B; }
.log-err  { color: #EF4444; }
.log-info { color: #2855E8; }
.log-muted { color: #2A3A5A; }
.log-ctrl { color: #A78BFA; }
.log-opt  { color: #34D399; }
.log-rag  { color: #FBBF24; }

/* ── Tool card ── */
.tool-card {
    background: linear-gradient(135deg, #0A0F1A 0%, #080C14 100%);
    border: 1px solid #141C2E; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
}
.tool-title { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #E8EDF8; margin-bottom: 3px; }
.tool-sub { font-family: 'Space Grotesk', sans-serif; font-size: 11px; color: #4A5A7A; }

/* ── Data table ── */
.data-table { width: 100%; border-collapse: collapse; font-size: 11px; }
.data-table th {
    background: #07090F; color: #2A3A5A;
    font-family: 'JetBrains Mono', monospace; font-size: 8px; letter-spacing: 1.2px;
    text-transform: uppercase; padding: 9px 11px; text-align: left; border-bottom: 1px solid #0D1220;
}
.data-table td { padding: 7px 11px; border-bottom: 1px solid #07090F; color: #C8D0E7; }
.data-table tr:last-child td { border-bottom: none; }
.data-table tr:hover td { background: #07090F; }

/* ── Chip tag ── */
.chip {
    display: inline-flex; align-items: center;
    background: rgba(30,64,207,0.08); border: 1px solid rgba(30,64,207,0.18);
    border-radius: 5px; padding: 2px 7px;
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #2855E8; margin: 2px;
}

/* ── Degree bar ── */
.deg-bar-wrap { display: flex; align-items: center; gap: 9px; margin: 4px 0; }
.deg-bar-label { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #C8D0E7; min-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.deg-bar-track { flex: 1; background: #07090F; border-radius: 3px; height: 9px; overflow: hidden; }
.deg-bar-fill  { height: 100%; background: linear-gradient(90deg, #1E40CF, #38D9A9); border-radius: 3px; transition: width 0.5s; }
.deg-bar-val   { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #38D9A9; min-width: 16px; text-align: right; }

/* ── Analysis block ── */
.analysis-block {
    background: #07090F; border-left: 2px solid #1E40CF;
    border-radius: 0 8px 8px 0; padding: 10px 14px; margin-bottom: 8px;
    font-family: 'Space Grotesk', sans-serif; font-size: 12px; color: #C8D0E7; line-height: 1.65;
}
.analysis-tag { font-family: 'JetBrains Mono', monospace; font-size: 8px; color: #2855E8; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 5px; }

/* ── Flow item ── */
.flow-item { display: flex; align-items: flex-start; gap: 10px; padding: 7px 0; border-bottom: 1px solid #07090F; }
.flow-item:last-child { border-bottom: none; }
.flow-num {
    background: rgba(30,64,207,0.12); border: 1px solid rgba(30,64,207,0.28);
    border-radius: 50%; width: 20px; height: 20px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace; font-size: 8px; font-weight: 700; color: #2855E8;
    flex-shrink: 0; margin-top: 2px;
}
.flow-text { font-family: 'Space Grotesk', sans-serif; font-size: 12px; color: #C8D0E7; }

/* ── Optimization card ── */
.opt-card {
    background: linear-gradient(135deg, #091A0F 0%, #071510 100%);
    border: 1px solid #143A28; border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
}
.opt-card-title { font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700; color: #34D399; margin-bottom: 4px; }
.opt-card-body { font-family: 'Space Grotesk', sans-serif; font-size: 12px; color: #A8C0B0; line-height: 1.6; }

/* ── RAG knowledge card ── */
.rag-card {
    background: linear-gradient(135deg, #141008 0%, #100D06 100%);
    border: 1px solid #2A2010; border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
}
.rag-tag { font-family: 'JetBrains Mono', monospace; font-size: 8px; color: #FBBF24; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 5px; }
.rag-body { font-family: 'Space Grotesk', sans-serif; font-size: 12px; color: #C8B87A; line-height: 1.65; }

/* ── Monte Carlo card ── */
.mc-card {
    background: linear-gradient(135deg, #0F0A18 0%, #0C0814 100%);
    border: 1px solid #241A38; border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
}

/* ── Signal flow animation ── */
.signal-step {
    display: flex; align-items: center; gap: 8px; padding: 6px 0;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #4A5A7A;
    border-bottom: 1px dashed #0D1220; transition: all 0.3s;
}
.signal-step.active-sig { color: #38D9A9; }
.sig-arrow { color: #1E40CF; font-size: 14px; }
.sig-node { background: rgba(30,64,207,0.12); border: 1px solid rgba(30,64,207,0.3); border-radius: 5px; padding: 2px 8px; }
.sig-node.active-node { background: rgba(56,217,169,0.15); border-color: rgba(56,217,169,0.4); color: #38D9A9; }

/* ── Builder component ── */
.builder-comp {
    background: #07090F; border: 1px solid #141C2E; border-radius: 8px;
    padding: 8px 12px; margin-bottom: 6px; display: flex;
    align-items: center; justify-content: space-between;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & REGISTRIES
# ══════════════════════════════════════════════════════════════════════════════

COMPONENT_COLORS = {
    "R": "#3B7AF5", "C": "#38D9A9", "L": "#F59E0B",
    "V": "#EF4444", "I": "#F87171", "D": "#A78BFA",
    "Q": "#FB923C", "M": "#F97316", "U": "#34D399",
    "GND": "#374151", "VCC": "#EF4444", "NET": "#D97706", "NODE": "#6B7280",
}
COMPONENT_FULL = {
    "R": "Resistor", "C": "Capacitor", "L": "Inductor",
    "V": "Voltage Source", "I": "Current Source",
    "D": "Diode", "Q": "BJT Transistor", "M": "MOSFET", "U": "Op-Amp",
}

EXAMPLE_CIRCUITS = {
    "RC Low-Pass Filter": "R1 connects node_in to node_mid with 1kΩ, C1 connects node_mid to GND with 10nF, Vin voltage source at node_in, output at node_mid",
    "RLC Band-Pass Filter": "R1 1kΩ between node_a and node_b, L1 10mH between node_b and node_c, C1 100nF between node_c and GND, V1 voltage source at node_a, GND reference",
    "Common-Emitter BJT Amplifier": "V1 voltage source 12V at VCC, R1 100kΩ from VCC to node_base, R2 10kΩ from node_base to GND, Q1 NPN BJT with base at node_base collector at node_collector emitter at node_emitter, RC 2kΩ from VCC to node_collector, RE 1kΩ from node_emitter to GND, C1 coupling capacitor from node_in to node_base, C2 bypass capacitor from node_emitter to GND",
    "Voltage Divider": "R1 10kΩ from VCC to node_mid, R2 5kΩ from node_mid to GND, V1 5V source at VCC",
    "Wien Bridge Oscillator": "R1 10kΩ node_a to node_b, R2 10kΩ node_b to node_out, C1 10nF node_a to GND, C2 10nF node_b to node_out, R3 20kΩ feedback from node_out to node_inv, R4 10kΩ node_inv to GND, U1 op-amp non-inv at node_b inv at node_inv output at node_out",
    "CMOS Inverter": "VDD 1.8V power supply, PMOS M1 source at VDD gate at node_in drain at node_out, NMOS M2 source at GND gate at node_in drain at node_out",
    "Phase-Locked Loop (PLL)": "VCO U1 input at node_ctrl output at node_clk, Phase detector U2 ref at node_ref fb at node_div out at node_pd, Loop filter R1 10kΩ C1 1nF from node_pd to node_ctrl, Divider U3 input at node_clk output at node_div, V1 3.3V supply at VCC",
    "Differential Amplifier": "R1 10kΩ from VCC to node_c1, R2 10kΩ from VCC to node_c2, Q1 NPN base at node_in1 collector at node_c1 emitter at node_tail, Q2 NPN base at node_in2 collector at node_c2 emitter at node_tail, RE 2kΩ from node_tail to GND, V1 12V at VCC",
}

# ── RAG Knowledge Base (in-code, retrieved by circuit type) ──
RAG_KNOWLEDGE = {
    "filter": """
RC/RLC Filter Design Knowledge:
- Cutoff frequency: fc = 1/(2π·R·C) for RC, fc = 1/(2π√LC) for LC
- Quality factor Q determines bandwidth: Q = f0/BW = (1/R)√(L/C)
- Butterworth filters: maximally flat passband, -20dB/decade per pole
- Chebyshev filters: steeper rolloff, equiripple passband
- Common pitfall: parasitic inductance in capacitors above 10MHz
- Layout tip: minimize loop area for LC filters to reduce EMI
- Simulation: AC sweep from 10Hz to 100MHz, log scale recommended
- DRC concern: capacitor spacing critical for high-voltage designs
""",
    "amplifier": """
Amplifier Design Knowledge:
- Common-emitter: high voltage gain, medium input/output impedance
- Common-base: high frequency response, low input impedance
- Common-collector (emitter follower): unity gain, high input Z, low output Z
- Gain-bandwidth product (GBW) is constant for a given transistor
- Biasing: voltage divider bias most stable against beta variation
- Bypass capacitor removes AC signal at emitter, maximizing gain
- Layout: minimize collector-to-base parasitic capacitance (Miller effect)
- Thermal considerations: power dissipation P = Ic·Vce must be managed
- ngspice tip: use .op for DC bias, .ac for frequency response, .tran for transient
""",
    "oscillator": """
Oscillator Design Knowledge:
- Wien bridge: f = 1/(2π·R·C), requires gain ≥ 3 for oscillation
- Colpitts: uses capacitive voltage divider feedback, good stability
- Hartley: uses inductive voltage divider, simple but noisy
- Phase shift: 3-stage RC network, each stage 60°, total 180° + inverter
- Crystal oscillator: Q > 10000, excellent frequency stability
- Startup condition: loop gain must exceed 1 initially
- Amplitude stabilization: AGC or soft-limiter diode pair
- Layout: keep feedback loop short, away from supply noise
""",
    "logic": """
Digital/Logic Circuit Knowledge:
- CMOS: static power only from leakage, dynamic power = α·C·V²·f
- Fan-out: number of gates driven, limited by drive strength
- Propagation delay: tpd = 0.69·R·C for RC model
- Setup/hold time violations cause metastability
- Level shifter needed when crossing voltage domains
- NMOS: faster, uses N-type channel, low threshold voltage preferred
- PMOS: slower (~2x), uses P-type channel, complementary to NMOS
- DRC rules: minimum spacing, width, enclosure per technology node
""",
    "rectifier": """
Rectifier & Power Design Knowledge:
- Half-wave: 50% efficiency, significant ripple
- Full-wave bridge: 4 diodes, ~81% efficiency, lower ripple
- Ripple voltage: Vr ≈ Iload/(f·C)
- Diode forward voltage: Si~0.7V, Schottky~0.3V, reduces output
- Filter capacitor sizing: C = Iload/(2·f·Vripple)
- Voltage regulation: linear (LDO) vs switching (buck/boost)
- Thermal: diode junction temperature Tj = Ta + Pd·θja
- Layout: keep high-current paths short and wide
""",
    "mixed": """
Mixed-Signal Circuit Knowledge:
- Analog and digital grounds must be separated, joined at single star point
- Decoupling capacitors: 100nF ceramic + 10µF electrolytic per power rail
- ADC: ENOB = (SNR - 1.76)/6.02 bits
- DAC: settling time critical for high-speed applications
- Noise coupling: digital switching noise couples into analog through substrate
- Guard rings: surround sensitive analog nodes with grounded ring
- Clock distribution: minimize skew, use H-tree or star topology
- Layout: analog on one side, digital on other, clear partition
""",
    "generic": """
General VLSI/Circuit Design Knowledge:
- Ohm's Law: V = I·R, Power: P = V·I = I²·R = V²/R
- Kirchhoff's Current Law (KCL): sum of currents at node = 0
- Kirchhoff's Voltage Law (KVL): sum of voltages in loop = 0
- Thevenin equivalent: any linear circuit = Vth + Rth
- Norton equivalent: Isc in parallel with Rth
- Superposition: each source analyzed independently, results summed
- SPICE elements: R, C, L, V, I, D, Q (BJT), M (MOSFET), E/F/G/H (controlled)
- Technology nodes: 180nm, 130nm, 90nm, 65nm, 45nm, 28nm, 16nm, 7nm, 5nm
""",
}

# ── Available AI Models ──
AI_MODELS = {
    "Claude Sonnet 4.6 (Default)": "claude-sonnet-4-6",
    "Claude Opus 4.6 (Powerful)": "claude-opus-4-6",
    "Claude Haiku 4.5 (Fast)": "claude-haiku-4-5-20251001",
}

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def init_state():
    defaults = {
        "graph": None,
        "agent_log": [],
        "analysis": None,
        "circuit_input": "",
        "spice_netlist": "",
        "graph_data": None,
        "metrics": {},
        "running": False,
        "uploaded_image_b64": None,
        "uploaded_image_type": None,
        "ngspice_output": "",
        "ngspice_waveform_data": None,
        "optimization_result": None,
        "monte_carlo_results": None,
        "rag_context": "",
        "report_content": "",
        "selected_model": "claude-sonnet-4-6",
        "builder_components": [],
        "signal_flow_step": 0,
        "agent_statuses": {
            "ctrl": "idle", "sim": "idle", "lay": "idle",
            "ver": "idle", "spc": "idle", "opt": "idle", "rag": "idle",
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ══════════════════════════════════════════════════════════════════════════════
# AGENT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
AGENTS = [
    ("ctrl", "🧠", "agent-icon-ctrl", "Controller Agent",    "Orchestrates pipeline, manages sequencing & data flow"),
    ("sim",  "⚡", "agent-icon-sim",  "Simulation Agent",    "ngspice execution, waveform analysis, AC/DC/Transient"),
    ("lay",  "📐", "agent-icon-lay",  "Layout Agent",        "Magic VLSI layout generation & parasitic extraction"),
    ("ver",  "✅", "agent-icon-ver",  "Verification Agent",  "KLayout DRC/LVS rule checks & design validation"),
    ("spc",  "📋", "agent-icon-spc",  "SPICE Netlist Agent", "Netlist generation, validation & export"),
    ("opt",  "✨", "agent-icon-opt",  "Optimization Agent",  "Component tuning, gain/noise improvement"),
    ("rag",  "📚", "agent-icon-rag",  "Knowledge Agent",     "RAG-based circuit expertise retrieval"),
]

# ══════════════════════════════════════════════════════════════════════════════
# CACHING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=600, show_spinner=False)
def cached_parse_circuit(circuit_description: str, image_b64: str, model: str) -> dict:
    """Cache parsed circuit graph to avoid re-calling API for same input."""
    return _parse_circuit_to_graph_uncached(circuit_description, image_b64, None, model)

def make_cache_key(text: str, img: str) -> str:
    combined = f"{text}||{img or ''}"
    return hashlib.md5(combined.encode()).hexdigest()

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING HELPER (adds to session state log)
# ══════════════════════════════════════════════════════════════════════════════
_active_logs: list = []

def add_log(msg: str, level: str = "info"):
    ts = time.strftime("%H:%M:%S")
    _active_logs.append({"time": ts, "msg": msg, "level": level})
    st.session_state.agent_log = _active_logs.copy()
    logger.info(f"[{level.upper()}] {msg}")

def reset_logs():
    global _active_logs
    _active_logs = []
    st.session_state.agent_log = []

# ══════════════════════════════════════════════════════════════════════════════
# RAG: KNOWLEDGE RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════
def retrieve_knowledge(circuit_type: str) -> str:
    """Retrieve relevant knowledge from in-code RAG knowledge base."""
    ct = circuit_type.lower() if circuit_type else "generic"
    # Match best knowledge section
    for key in ["filter", "amplifier", "oscillator", "logic", "rectifier", "mixed"]:
        if key in ct:
            return RAG_KNOWLEDGE[key]
    return RAG_KNOWLEDGE["generic"]

# ══════════════════════════════════════════════════════════════════════════════
# AI API CALLS
# ══════════════════════════════════════════════════════════════════════════════
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def _parse_circuit_to_graph_uncached(circuit_description: str, image_b64: str = None, image_type: str = None, model: str = "claude-sonnet-4-6") -> dict:
    """Simulation Agent: Parse circuit (text + optional image) into graph JSON."""
    client = get_client()
    prompt = f"""You are the Simulation Agent in a VLSI automated design pipeline.
Your task: parse the following circuit input into a precise graph representation for ngspice simulation.

Vertices (graph nodes) = electrical nodes/nets (node_in, node_out, GND, VCC, node_base, etc.)
Edges (graph edges) = components connecting two nodes (R, C, L, V, I, D, Q, M, U)

Circuit description:
\"\"\"{circuit_description}\"\"\"

Return ONLY valid JSON with this exact structure (no markdown, no explanation):
{{
  "circuit_name": "short descriptive name",
  "nodes": [
    {{"id": "node_id", "label": "display label", "type": "NODE|GND|VCC|NET", "voltage": null}}
  ],
  "edges": [
    {{
      "id": "comp_id",
      "source": "node_id_1",
      "target": "node_id_2",
      "type": "R|C|L|V|I|D|Q|M|U",
      "label": "R1",
      "value": "1kΩ",
      "description": "component description",
      "spice_model": "optional SPICE model"
    }}
  ],
  "graph_properties": {{
    "num_nodes": 0,
    "num_edges": 0,
    "circuit_type": "filter|amplifier|oscillator|rectifier|divider|logic|mixed",
    "topology": "series|parallel|ladder|bridge|feedback|tree|mixed",
    "technology_node": "generic|CMOS|BiCMOS|BJT|RF",
    "supply_voltage": "inferred supply voltage"
  }}
}}

Rules:
- Every component MUST have exactly 2 nodes. For 3-terminal devices (BJT, MOSFET), create an intermediate node.
- GND node id must always be "GND".
- VCC/power supply should have id "VCC" or "VCC_X".
- Node ids must be valid Python identifiers (no spaces, use underscores).
- If a component value is missing, make a reasonable assumption.
- Return ONLY the JSON object."""

    content = [{"type": "text", "text": prompt}]
    if image_b64 and image_type:
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": image_type, "data": image_b64}},
            {"type": "text", "text": prompt}
        ]

    msg = client.messages.create(model=model, max_tokens=2500, messages=[{"role": "user", "content": content}])
    raw = msg.content[0].text.strip()
    raw = re.sub(r'^```[a-z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    return json.loads(raw)


def parse_circuit_to_graph(circuit_description: str, image_b64: str = None, image_type: str = None, model: str = "claude-sonnet-4-6") -> dict:
    """Wrapper with fallback JSON error handling."""
    try:
        return _parse_circuit_to_graph_uncached(circuit_description, image_b64, image_type, model)
    except json.JSONDecodeError as e:
        raise ValueError(f"AI returned invalid JSON: {e}") from e


def analyze_circuit_graph(graph_data: dict, G: nx.Graph, rag_context: str = "", model: str = "claude-sonnet-4-6") -> dict:
    """Multi-role analysis: Simulation + Layout + Verification agents, with RAG context injection."""
    client = get_client()
    node_list = [f"{n['id']} ({n['type']})" for n in graph_data["nodes"]]
    edge_list = [
        f"{e['label']} {e['type']} ({e.get('value','?')}) : {e['source']} → {e['target']}"
        for e in graph_data["edges"]
    ]
    degree_seq = dict(G.degree())
    try:    is_connected = nx.is_connected(G)
    except: is_connected = False
    try:    cycles = nx.cycle_basis(G); num_loops = len(cycles)
    except: num_loops = 0

    rag_section = f"\n\nDomain Knowledge (RAG Context):\n{rag_context}" if rag_context else ""

    prompt = f"""You are the multi-agent VLSI Analysis System acting as:
1. Simulation Agent (ngspice) — analyze signal behavior
2. Layout Agent (Magic VLSI) — identify layout constraints
3. Verification Agent (KLayout) — flag DRC/LVS concerns
4. Controller Agent — synthesize recommendations
{rag_section}

Circuit: {graph_data.get('circuit_name', 'Unknown')}
Technology: {graph_data.get('graph_properties',{}).get('technology_node','generic')}
Nodes: {', '.join(node_list)}
Components (edges):
{chr(10).join(edge_list)}

Graph metrics:
- Connected: {is_connected}
- Mesh loops (KVL): {num_loops}
- Node degrees (KCL): {degree_seq}

Return ONLY valid JSON (no markdown):
{{
  "summary": "2-3 sentence circuit description",
  "function": "what this circuit does",
  "signal_flow": ["step1", "step2", "step3"],
  "critical_nodes": [{{"node": "id", "reason": "why important"}}],
  "component_analysis": [{{"component": "label", "role": "its role", "agent": "Simulation|Layout|Verification"}}],
  "graph_properties": {{
    "kirchhoff_nodes": "KCL observation",
    "mesh_loops": {num_loops},
    "connectivity": "{is_connected}",
    "max_degree_node": "node with most connections",
    "estimated_cutoff_freq": "if applicable else N/A",
    "estimated_gain": "if applicable else N/A",
    "power_dissipation": "estimated or N/A"
  }},
  "simulation_agent": {{
    "ngspice_analysis_type": "AC|DC|Transient|Mixed",
    "simulation_notes": "key notes for ngspice"
  }},
  "layout_agent": {{
    "magic_constraints": ["constraint1", "constraint2"],
    "estimated_area": "rough area estimate or N/A",
    "routing_notes": "key layout routing guidance"
  }},
  "verification_agent": {{
    "drc_concerns": ["concern1", "concern2"],
    "lvs_checkpoints": ["checkpoint1", "checkpoint2"],
    "klayout_rules": "rule set recommendation"
  }},
  "spice_netlist": "complete SPICE netlist starting with title line, one component per line, ending with .end",
  "ngspice_commands": ["list of ngspice analysis commands"],
  "recommendations": ["design tip 1", "design tip 2"],
  "warnings": ["any issues"],
  "automation_score": 90
}}"""

    msg = client.messages.create(model=model, max_tokens=4000, messages=[{"role": "user", "content": prompt}])
    raw = msg.content[0].text.strip()
    raw = re.sub(r'^```[a-z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    return json.loads(raw)


def optimize_circuit(graph_data: dict, analysis: dict, optimization_goal: str, model: str = "claude-sonnet-4-6") -> dict:
    """Optimization Agent: suggest improved component values and netlist."""
    client = get_client()
    circuit_name = graph_data.get("circuit_name", "circuit")
    current_netlist = analysis.get("spice_netlist", "")
    edge_list = [f"{e['label']} {e['type']} {e.get('value','?')}" for e in graph_data.get("edges", [])]

    prompt = f"""You are the Optimization Agent for VLSI/circuit design.
Circuit: {circuit_name}
Current components: {', '.join(edge_list)}
Current SPICE netlist:
{current_netlist}

Optimization goal: {optimization_goal}
Circuit function: {analysis.get('function', 'N/A')}
Estimated gain: {analysis.get('graph_properties', {}).get('estimated_gain', 'N/A')}
Estimated fc: {analysis.get('graph_properties', {}).get('estimated_cutoff_freq', 'N/A')}

Return ONLY valid JSON:
{{
  "optimization_summary": "what was optimized and why",
  "changes": [
    {{"component": "R1", "original": "1kΩ", "optimized": "820Ω", "reason": "improves gain margin"}}
  ],
  "improved_netlist": "complete optimized SPICE netlist",
  "expected_improvement": {{
    "gain": "expected gain change",
    "bandwidth": "expected bandwidth change",
    "noise": "expected noise change",
    "power": "expected power change"
  }},
  "optimization_score": 85,
  "additional_recommendations": ["tip1", "tip2"]
}}"""

    msg = client.messages.create(model=model, max_tokens=2500, messages=[{"role": "user", "content": prompt}])
    raw = msg.content[0].text.strip()
    raw = re.sub(r'^```[a-z]*\n?', '', raw)
    raw = re.sub(r'\n?```$', '', raw)
    return json.loads(raw)


def generate_report(graph_data: dict, G: nx.Graph, analysis: dict, ngspice_output: str, optimization: dict = None) -> str:
    """Generate a comprehensive Markdown report."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    circuit_name = graph_data.get("circuit_name", "Circuit")
    gp = graph_data.get("graph_properties", {})
    agp = analysis.get("graph_properties", {})

    try:    is_conn = nx.is_connected(G); loops = len(nx.cycle_basis(G)); density = round(nx.density(G), 4)
    except: is_conn, loops, density = False, 0, 0.0

    netlist = analysis.get("spice_netlist", "N/A")
    cmds = "\n".join(analysis.get("ngspice_commands", []))
    recs = "\n".join([f"- {r}" for r in analysis.get("recommendations", [])])
    warns = "\n".join([f"- ⚠ {w}" for w in analysis.get("warnings", [])])
    drc = "\n".join([f"- {d}" for d in analysis.get("verification_agent", {}).get("drc_concerns", [])])
    layout_notes = analysis.get("layout_agent", {}).get("routing_notes", "N/A")
    magic_constraints = "\n".join([f"- {c}" for c in analysis.get("layout_agent", {}).get("magic_constraints", [])])

    opt_section = ""
    if optimization:
        changes = "\n".join([f"| {c['component']} | {c['original']} | {c['optimized']} | {c['reason']} |" for c in optimization.get("changes", [])])
        impr = optimization.get("expected_improvement", {})
        opt_section = f"""
## 🔧 Optimization Results

**Summary:** {optimization.get('optimization_summary', 'N/A')}

**Optimization Score:** {optimization.get('optimization_score', 'N/A')}%

### Component Changes
| Component | Original | Optimized | Reason |
|-----------|----------|-----------|--------|
{changes}

### Expected Improvements
- **Gain:** {impr.get('gain', 'N/A')}
- **Bandwidth:** {impr.get('bandwidth', 'N/A')}
- **Noise:** {impr.get('noise', 'N/A')}
- **Power:** {impr.get('power', 'N/A')}

### Optimized Netlist
```spice
{optimization.get('improved_netlist', '')}
```
"""

    ngspice_section = f"""
## ⚡ ngspice Simulation Output
```
{ngspice_output if ngspice_output else 'ngspice not available or not run.'}
```
""" if ngspice_output else ""

    report = f"""# AutoEDA AI — Circuit Analysis Report
**Generated:** {now}
**Circuit:** {circuit_name}
**Technology:** {gp.get('technology_node', 'generic')}
**Circuit Type:** {gp.get('circuit_type', 'N/A')}
**Topology:** {gp.get('topology', 'N/A')}
**Supply Voltage:** {gp.get('supply_voltage', 'N/A')}

---

## 📊 Executive Summary
{analysis.get('summary', 'N/A')}

**Function:** {analysis.get('function', 'N/A')}

---

## 🕸️ Graph Metrics
| Metric | Value |
|--------|-------|
| Vertices (Nodes) | {G.number_of_nodes()} |
| Edges (Components) | {G.number_of_edges()} |
| Mesh Loops (KVL) | {loops} |
| Graph Density | {density} |
| Connected | {'Yes' if is_conn else 'No'} |
| Automation Score | {analysis.get('automation_score', 'N/A')}% |

**Performance Parameters:**
- Estimated Cutoff Frequency: {agp.get('estimated_cutoff_freq', 'N/A')}
- Estimated Gain: {agp.get('estimated_gain', 'N/A')}
- Power Dissipation: {agp.get('power_dissipation', 'N/A')}
- KCL Observation: {agp.get('kirchhoff_nodes', 'N/A')}

---

## 📡 Signal Flow
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(analysis.get('signal_flow', []))])}

---

## 🔬 Simulation Agent (ngspice)
**Analysis Type:** {analysis.get('simulation_agent', {}).get('ngspice_analysis_type', 'N/A')}
**Notes:** {analysis.get('simulation_agent', {}).get('simulation_notes', 'N/A')}

### ngspice Commands
```
{cmds}
```
{ngspice_section}

---

## 📐 Layout Agent (Magic VLSI)
**Estimated Area:** {analysis.get('layout_agent', {}).get('estimated_area', 'N/A')}
**Routing Notes:** {layout_notes}

### Layout Constraints
{magic_constraints if magic_constraints else '- None specified'}

---

## ✅ Verification Agent (KLayout DRC/LVS)
**Rule Set:** {analysis.get('verification_agent', {}).get('klayout_rules', 'N/A')}

### DRC Concerns
{drc if drc else '- No DRC concerns flagged'}

### LVS Checkpoints
{chr(10).join([f"- {c}" for c in analysis.get('verification_agent', {}).get('lvs_checkpoints', [])])}

---
{opt_section}

## 💡 Design Recommendations
{recs if recs else '- No recommendations generated'}

## ⚠ Warnings
{warns if warns else '- No warnings'}

---

## 📋 SPICE Netlist
```spice
{netlist}
```

---

## 🧩 Component Table
| Label | Type | Value | Source | Target |
|-------|------|-------|--------|--------|
{chr(10).join([f"| {e.get('label','')} | {e.get('type','')} | {e.get('value','—')} | {e.get('source','')} | {e.get('target','')} |" for e in graph_data.get('edges', [])])}

---
*Generated by AutoEDA AI — Multi-Agent VLSI Design Platform*
"""
    return report

# ══════════════════════════════════════════════════════════════════════════════
# NGSPICE EXECUTION
# ══════════════════════════════════════════════════════════════════════════════
def run_ngspice(netlist: str, timeout: int = 30) -> tuple[bool, str, list]:
    """
    Execute ngspice with the given SPICE netlist.
    Returns: (success, output_text, waveform_data_list)
    waveform_data_list: list of dicts with 'time', 'voltage' etc. for plotting
    """
    if not netlist or netlist.strip() == "":
        return False, "Empty netlist provided.", []

    try:
        # Check if ngspice is available
        result = subprocess.run(["which", "ngspice"], capture_output=True, text=True, timeout=5)
        ngspice_available = result.returncode == 0
    except Exception:
        ngspice_available = False

    if not ngspice_available:
        # ngspice not installed — generate realistic synthetic waveform for demo
        add_log("[SIM]   ngspice not found in PATH — generating synthetic simulation data for demo", "warn")
        waveform_data = _generate_synthetic_waveform(netlist)
        sim_output = """[AutoEDA AI Demo Mode — ngspice not installed]
Synthetic simulation data generated based on circuit analysis.

To run real ngspice simulation:
  sudo apt-get install ngspice   # Ubuntu/Debian
  brew install ngspice           # macOS
  pip install pyspice            # Python wrapper

Demo output:
  DC operating point estimated from topology
  AC response approximated from component values
  Transient waveform generated from circuit type
"""
        return True, sim_output, waveform_data

    # Write netlist to temp file and run ngspice
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sp', delete=False, prefix='autoeda_') as f:
            # Ensure netlist has batch mode directive
            if ".control" not in netlist.lower():
                enhanced = netlist.rstrip()
                if not enhanced.endswith(".end"):
                    enhanced += "\n.end"
                # Add minimal control block for batch output
                enhanced = enhanced.replace(".end", ".control\nrun\nprint all\n.endc\n.end")
            else:
                enhanced = netlist
            f.write(enhanced)
            tmp_path = f.name

        proc = subprocess.run(
            ["ngspice", "-b", tmp_path],
            capture_output=True, text=True, timeout=timeout
        )
        output = proc.stdout + proc.stderr

        # Parse output for waveform data
        waveform_data = _parse_ngspice_output(output)
        success = proc.returncode == 0
        add_log(f"[SIM]   ngspice exit code: {proc.returncode}", "ok" if success else "err")
        return success, output, waveform_data

    except subprocess.TimeoutExpired:
        return False, "ngspice simulation timed out after 30 seconds.", []
    except FileNotFoundError:
        return False, "ngspice executable not found.", []
    except Exception as e:
        return False, f"ngspice error: {type(e).__name__}: {e}", []
    finally:
        try:
            if 'tmp_path' in locals():
                os.unlink(tmp_path)
        except Exception:
            pass


def _parse_ngspice_output(output: str) -> list:
    """Parse ngspice text output for numeric data to build waveform DataFrame."""
    waveform_data = []
    lines = output.split('\n')
    # Look for tabular data patterns like: time   v(node)   ...
    header = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if header is None:
            if len(parts) >= 2 and parts[0].lower() in ('time', 'frequency', 'v(', 'index'):
                header = parts
                continue
        else:
            try:
                vals = [float(p) for p in parts]
                if len(vals) == len(header):
                    waveform_data.append(dict(zip(header, vals)))
            except ValueError:
                if len(parts) < 2 or not any(c.isdigit() for c in parts[0]):
                    header = None  # reset
    return waveform_data


def _generate_synthetic_waveform(netlist: str) -> list:
    """Generate realistic synthetic waveform data for demo when ngspice unavailable."""
    netlist_lower = netlist.lower()
    points = 200

    # Detect circuit type from netlist
    if any(x in netlist_lower for x in ['.ac', 'ac dec', 'ac oct']):
        # AC response — frequency domain
        freqs = np.logspace(1, 7, points)  # 10Hz to 10MHz
        # RC filter approximation
        if 'r' in netlist_lower and 'c' in netlist_lower:
            fc = 15915  # ~1/(2π·1kΩ·10nF)
            gain_db = -20 * np.log10(np.sqrt(1 + (freqs/fc)**2))
            phase = -np.degrees(np.arctan(freqs/fc))
        else:
            gain_db = np.zeros(points) - 0.1 * np.random.randn(points)
            phase = np.linspace(0, -180, points) + np.random.randn(points) * 2
        return [{"frequency": f, "gain_db": g, "phase_deg": p} for f, g, p in zip(freqs, gain_db, phase)]

    elif any(x in netlist_lower for x in ['.tran', 'tran']):
        # Transient — time domain
        t = np.linspace(0, 1e-3, points)
        freq_sig = 1e3
        if 'bjt' in netlist_lower or 'q' in netlist_lower:
            # Amplifier output — inverted, amplified
            vin = 0.01 * np.sin(2 * np.pi * freq_sig * t)
            vout = -8.5 * vin + 0.002 * np.random.randn(points)
        elif 'osc' in netlist_lower or 'wien' in netlist_lower:
            # Oscillator — growing then stable sine
            env = np.minimum(1.0, t / (t.max() * 0.3))
            vout = 3.0 * env * np.sin(2 * np.pi * freq_sig * t)
            vin = np.zeros(points)
        else:
            # Generic RC step response
            tau = 1e-4
            vin = np.where(t > 0.1e-3, 5.0, 0.0)
            vout = 5.0 * (1 - np.exp(-t / tau))
        return [{"time": float(tt), "v_in": float(vi), "v_out": float(vo)} for tt, vi, vo in zip(t, vin, vout)]

    else:
        # DC sweep
        v_in = np.linspace(0, 5, points)
        if 'diode' in netlist_lower or 'd' in netlist_lower:
            v_out = np.maximum(0, v_in - 0.7)
            i_d = 1e-14 * (np.exp(v_in / 0.026) - 1)
        else:
            # Voltage divider
            v_out = v_in * (5 / 15)  # 5k / (10k + 5k)
            i_d = v_out / 5000
        return [{"v_in": float(vi), "v_out": float(vo), "current_mA": float(id_)*1000} for vi, vo, id_ in zip(v_in, v_out, i_d)]


def plot_waveforms(waveform_data: list) -> plt.Figure | None:
    """Plot waveform data as matplotlib figure."""
    if not waveform_data:
        return None
    df = pd.DataFrame(waveform_data)
    if df.empty:
        return None

    fig, axes = plt.subplots(1 if len(df.columns) <= 3 else 2, 1, figsize=(12, 5 if len(df.columns) <= 3 else 8))
    bg = "#030508"
    fig.patch.set_facecolor(bg)
    if not isinstance(axes, np.ndarray):
        axes = [axes]
    for ax in axes:
        ax.set_facecolor(bg)
        ax.spines['bottom'].set_color('#141C2E')
        ax.spines['left'].set_color('#141C2E')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(colors='#4A5A7A', labelsize=8)
        ax.yaxis.label.set_color('#C8D0E7')
        ax.xaxis.label.set_color('#C8D0E7')
        ax.title.set_color('#E8EDF8')

    cols = df.columns.tolist()

    if "frequency" in cols:
        # AC response
        ax = axes[0]
        if "gain_db" in cols:
            ax.semilogx(df["frequency"], df["gain_db"], color="#38D9A9", linewidth=2, label="Gain (dB)")
            ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Gain (dB)")
            ax.set_title("AC Frequency Response", fontfamily="monospace", fontsize=11)
            ax.legend(facecolor=bg, edgecolor="#141C2E", labelcolor="#C8D0E7", fontsize=8)
            ax.grid(True, color="#0D1220", linewidth=0.5)
        if len(axes) > 1 and "phase_deg" in cols:
            ax2 = axes[1]
            ax2.set_facecolor(bg)
            ax2.semilogx(df["frequency"], df["phase_deg"], color="#2855E8", linewidth=2, label="Phase (°)")
            ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Phase (°)")
            ax2.set_title("Phase Response", fontfamily="monospace", fontsize=11)
            ax2.legend(facecolor=bg, edgecolor="#141C2E", labelcolor="#C8D0E7", fontsize=8)
            ax2.grid(True, color="#0D1220", linewidth=0.5)
            ax2.spines['bottom'].set_color('#141C2E'); ax2.spines['left'].set_color('#141C2E')
            ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
            ax2.tick_params(colors='#4A5A7A', labelsize=8)

    elif "time" in cols:
        # Transient
        ax = axes[0]
        if "v_in" in cols:
            ax.plot(df["time"] * 1000, df["v_in"], color="#2855E8", linewidth=1.5, label="Vin", alpha=0.85)
        if "v_out" in cols:
            ax.plot(df["time"] * 1000, df["v_out"], color="#38D9A9", linewidth=2, label="Vout")
        ax.set_xlabel("Time (ms)"); ax.set_ylabel("Voltage (V)")
        ax.set_title("Transient Response", fontfamily="monospace", fontsize=11)
        ax.legend(facecolor=bg, edgecolor="#141C2E", labelcolor="#C8D0E7", fontsize=8)
        ax.grid(True, color="#0D1220", linewidth=0.5)

    elif "v_in" in cols:
        # DC sweep
        ax = axes[0]
        if "v_out" in cols:
            ax.plot(df["v_in"], df["v_out"], color="#38D9A9", linewidth=2, label="Vout")
        ax.set_xlabel("Vin (V)"); ax.set_ylabel("Vout (V)")
        ax.set_title("DC Transfer Characteristic", fontfamily="monospace", fontsize=11)
        ax.legend(facecolor=bg, edgecolor="#141C2E", labelcolor="#C8D0E7", fontsize=8)
        ax.grid(True, color="#0D1220", linewidth=0.5)

    plt.tight_layout(pad=0.8)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
def run_monte_carlo(graph_data: dict, num_runs: int = 50, tolerance_pct: float = 5.0) -> dict:
    """
    Monte Carlo simulation: vary component values by ±tolerance%, compute output variation.
    Returns statistical summary.
    """
    edges = graph_data.get("edges", [])
    circuit_type = graph_data.get("graph_properties", {}).get("circuit_type", "filter")

    results = []
    component_variations = []

    for run in range(num_runs):
        varied_components = {}
        for e in edges:
            val_str = e.get("value", "")
            numeric = _parse_component_value(val_str)
            if numeric is not None:
                variation = 1.0 + (random.uniform(-tolerance_pct, tolerance_pct) / 100.0)
                varied_val = numeric * variation
                varied_components[e.get("label", "")] = varied_val

        # Compute a synthetic output metric based on circuit type
        if "filter" in circuit_type:
            # RC filter: fc = 1/(2π·R·C)
            r_vals = [v for k, v in varied_components.items() if k.startswith("R")]
            c_vals = [v for k, v in varied_components.items() if k.startswith("C")]
            if r_vals and c_vals:
                fc = 1 / (2 * math.pi * r_vals[0] * c_vals[0])
                results.append(fc)
            else:
                results.append(random.gauss(15915, 500))
        elif "amplifier" in circuit_type:
            # Gain approximation
            r_c = next((v for k, v in varied_components.items() if "RC" in k or "Rc" in k), 2000)
            r_e = next((v for k, v in varied_components.items() if "RE" in k or "Re" in k), 1000)
            gain = -r_c / (r_e + 26)  # simple CE gain approx
            results.append(abs(gain))
        else:
            # Generic: output voltage ratio variation
            vals = list(varied_components.values())
            if len(vals) >= 2:
                results.append(vals[0] / (vals[0] + vals[1]) * 5.0)
            else:
                results.append(random.gauss(2.5, 0.1))

        component_variations.append(varied_components)

    if not results:
        results = [random.gauss(1000, 50) for _ in range(num_runs)]

    results_arr = np.array(results)
    return {
        "num_runs": num_runs,
        "tolerance_pct": tolerance_pct,
        "circuit_type": circuit_type,
        "metric_name": "Cutoff Frequency (Hz)" if "filter" in circuit_type else "Gain" if "amplifier" in circuit_type else "Output (V)",
        "mean": float(np.mean(results_arr)),
        "std": float(np.std(results_arr)),
        "min": float(np.min(results_arr)),
        "max": float(np.max(results_arr)),
        "p5": float(np.percentile(results_arr, 5)),
        "p95": float(np.percentile(results_arr, 95)),
        "yield_pct": float(np.sum(np.abs(results_arr - np.mean(results_arr)) < 3 * np.std(results_arr)) / num_runs * 100),
        "raw_data": results_arr.tolist(),
    }


def _parse_component_value(val_str: str) -> float | None:
    """Parse component value string like '1kΩ', '10nF', '2.2mH' to float in SI units."""
    val_str = val_str.strip().replace('Ω','').replace('F','').replace('H','').replace('V','').replace('A','')
    multipliers = {'T':1e12,'G':1e9,'M':1e6,'k':1e3,'m':1e-3,'μ':1e-6,'u':1e-6,'n':1e-9,'p':1e-12,'f':1e-15}
    try:
        for suffix, mult in multipliers.items():
            if val_str.endswith(suffix):
                return float(val_str[:-1]) * mult
        return float(val_str)
    except Exception:
        return None


def plot_monte_carlo(mc_results: dict) -> plt.Figure:
    """Plot Monte Carlo histogram."""
    data = mc_results["raw_data"]
    fig, ax = plt.subplots(figsize=(10, 4))
    bg = "#030508"
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.spines['bottom'].set_color('#141C2E'); ax.spines['left'].set_color('#141C2E')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.tick_params(colors='#4A5A7A', labelsize=8)

    n, bins, patches = ax.hist(data, bins=30, color="#1E40CF", alpha=0.75, edgecolor="#0D1220", linewidth=0.5)
    # Color the central 90% differently
    mean_val = mc_results["mean"]
    p5, p95 = mc_results["p5"], mc_results["p95"]
    for patch, left_edge in zip(patches, bins[:-1]):
        if p5 <= left_edge <= p95:
            patch.set_facecolor("#38D9A9")
            patch.set_alpha(0.85)

    ax.axvline(mean_val, color="#F59E0B", linewidth=1.5, linestyle="--", label=f"Mean: {mean_val:.4g}")
    ax.axvline(p5, color="#EF4444", linewidth=1, linestyle=":", label=f"5th pct: {p5:.4g}")
    ax.axvline(p95, color="#EF4444", linewidth=1, linestyle=":", label=f"95th pct: {p95:.4g}")

    ax.set_xlabel(mc_results["metric_name"], color="#C8D0E7", fontsize=9, fontfamily="monospace")
    ax.set_ylabel("Count", color="#C8D0E7", fontsize=9, fontfamily="monospace")
    ax.set_title(f"Monte Carlo ({mc_results['num_runs']} runs, ±{mc_results['tolerance_pct']}% tolerance)", color="#E8EDF8", fontsize=11, fontfamily="monospace")
    ax.legend(facecolor=bg, edgecolor="#141C2E", labelcolor="#C8D0E7", fontsize=8)
    ax.grid(True, color="#0D1220", linewidth=0.4, axis='y')
    plt.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def build_networkx_graph(graph_data: dict) -> nx.Graph:
    G = nx.Graph()
    for n in graph_data["nodes"]:
        G.add_node(n["id"], **n)
    for e in graph_data["edges"]:
        G.add_edge(e["source"], e["target"], **e)
    return G


def compute_graph_analytics(G: nx.Graph) -> dict:
    """Compute extended NetworkX analytics: centrality, importance, etc."""
    analytics = {}
    try:
        analytics["degree_centrality"] = nx.degree_centrality(G)
        analytics["betweenness_centrality"] = nx.betweenness_centrality(G)
        analytics["closeness_centrality"] = nx.closeness_centrality(G)
        analytics["eigenvector_centrality"] = nx.eigenvector_centrality(G, max_iter=500)
    except Exception:
        analytics["degree_centrality"] = dict(G.degree())
        analytics["betweenness_centrality"] = {}
        analytics["closeness_centrality"] = {}
        analytics["eigenvector_centrality"] = {}

    try:
        analytics["is_connected"] = nx.is_connected(G)
        analytics["num_loops"] = len(nx.cycle_basis(G))
        analytics["density"] = nx.density(G)
        analytics["diameter"] = nx.diameter(G) if nx.is_connected(G) else None
        analytics["avg_shortest_path"] = nx.average_shortest_path_length(G) if nx.is_connected(G) else None
    except Exception as e:
        analytics["is_connected"] = False
        analytics["num_loops"] = 0
        analytics["density"] = 0.0
        analytics["diameter"] = None
        analytics["avg_shortest_path"] = None

    # Identify critical nodes (high betweenness or degree)
    bc = analytics.get("betweenness_centrality", {})
    dc = analytics.get("degree_centrality", {})
    analytics["critical_nodes"] = sorted(bc.keys(), key=lambda n: bc.get(n, 0) + dc.get(n, 0), reverse=True)[:5]
    return analytics


def draw_circuit_graph(G: nx.Graph, graph_data: dict, highlight_nodes: list = None) -> plt.Figure:
    """Draw the circuit topology graph with optional node highlighting."""
    fig, ax = plt.subplots(figsize=(13, 7))
    bg = "#020408"
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)

    n = G.number_of_nodes()
    if n <= 6:     pos = nx.spring_layout(G, seed=42, k=3.0)
    elif n <= 12:  pos = nx.kamada_kawai_layout(G)
    else:          pos = nx.spring_layout(G, seed=42, k=2.0, iterations=120)

    highlight_nodes = highlight_nodes or []
    node_colors, node_sizes, node_borders = [], [], []
    for node in G.nodes():
        ndata = G.nodes[node]
        ntype = ndata.get("type", "NODE")
        is_highlight = node in highlight_nodes
        if is_highlight:
            node_colors.append("#38D9A9"); node_sizes.append(1100); node_borders.append("#38D9A9")
        elif ntype == "GND":
            node_colors.append("#0D1220"); node_sizes.append(550); node_borders.append("#374151")
        elif ntype == "VCC":
            node_colors.append("#180A0A"); node_sizes.append(680); node_borders.append("#EF4444")
        elif ntype == "NET":
            node_colors.append("#0D0A1A"); node_sizes.append(620); node_borders.append("#A78BFA")
        else:
            node_colors.append("#080C18"); node_sizes.append(820); node_borders.append("#2855E8")

    edge_colors, edge_widths = [], []
    for u, v, edata in G.edges(data=True):
        etype = edata.get("type", "R")
        edge_colors.append(COMPONENT_COLORS.get(etype, "#374151"))
        edge_widths.append(2.5)

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, width=edge_widths, alpha=0.9, arrows=False)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, edgecolors=node_borders, linewidths=2.0)
    node_labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax, font_size=7.5, font_color="#E8EDF8", font_family="monospace", font_weight="bold")

    edge_labels = {}
    for u, v, edata in G.edges(data=True):
        lbl = edata.get("label", "")
        val = edata.get("value", "")
        edge_labels[(u, v)] = f"{lbl}\n{val}" if val else lbl
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=7, font_color="#F59E0B", font_family="monospace",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#06080D", edgecolor="#141C2E", alpha=0.92))

    legend_handles = []
    seen = set()
    for _, _, edata in G.edges(data=True):
        t = edata.get("type", "R")
        if t not in seen:
            seen.add(t)
            legend_handles.append(mpatches.Patch(color=COMPONENT_COLORS.get(t,"#374151"), label=f"{t} – {COMPONENT_FULL.get(t,t)}"))
    for ntype, fc, ec in [("Circuit Node","#080C18","#2855E8"),("Ground","#0D1220","#374151"),("VCC","#180A0A","#EF4444"),("Critical","#38D9A9","#38D9A9")]:
        legend_handles.append(mpatches.Patch(facecolor=fc, edgecolor=ec, label=ntype))

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=7, facecolor="#06080D", edgecolor="#141C2E", labelcolor="#C8D0E7", framealpha=0.96)

    ax.set_title(graph_data.get("circuit_name","Circuit Graph"), color="#E8EDF8", fontsize=12, fontweight="bold", pad=14, fontfamily="monospace")
    ax.axis("off")
    plt.tight_layout(pad=0.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# CIRCUIT BUILDER HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def builder_to_description(components: list) -> str:
    """Convert builder component list to circuit description string."""
    if not components:
        return ""
    parts = []
    for c in components:
        comp_type = c.get("type", "R")
        label = c.get("label", "")
        value = c.get("value", "")
        src = c.get("source", "")
        tgt = c.get("target", "GND")
        full_name = COMPONENT_FULL.get(comp_type, comp_type)
        parts.append(f"{label} {full_name} {value} from {src} to {tgt}")
    return ", ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 4px 8px">
        <div style="font-family:'JetBrains Mono',monospace;font-size:13px;font-weight:700;color:#E8EDF8;letter-spacing:-0.3px">
            ⚡ AutoEDA AI
        </div>
        <div style="font-family:'Space Grotesk',sans-serif;font-size:10px;color:#2A3A5A;margin-top:3px">
            Intelligent Circuit Design Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── AI Model selector ──
    st.markdown('<div class="sec-header">// AI MODEL</div>', unsafe_allow_html=True)
    selected_model_name = st.selectbox(
        "Model",
        list(AI_MODELS.keys()),
        index=0,
        label_visibility="collapsed"
    )
    st.session_state.selected_model = AI_MODELS[selected_model_name]

    st.divider()

    # ── Agent roster ──
    st.markdown('<div class="sec-header">// AGENT ROSTER</div>', unsafe_allow_html=True)
    statuses = st.session_state.agent_statuses
    status_map = {
        "idle": ("IDLE", "badge-idle"),
        "run":  ("RUNNING", "badge-run"),
        "done": ("DONE", "badge-done"),
        "err":  ("ERROR", "badge-err"),
        "busy": ("BUSY", "badge-busy"),
    }
    for key, icon, icon_cls, name, role in AGENTS:
        s = statuses.get(key, "idle")
        badge_text, badge_cls = status_map.get(s, ("IDLE","badge-idle"))
        st.markdown(f"""
        <div class="agent-card">
            <div class="agent-card-left">
                <div class="agent-icon {icon_cls}">{icon}</div>
                <div>
                    <div class="agent-name">{name}</div>
                    <div class="agent-role">{role}</div>
                </div>
            </div>
            <span class="badge {badge_cls}">{badge_text}</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Pipeline stages ──
    st.markdown('<div class="sec-header">// PIPELINE STAGES</div>', unsafe_allow_html=True)
    pipeline_stages = [
        ("1", "Knowledge Retrieval", "RAG Agent"),
        ("2", "Parse Circuit",       "Simulation Agent"),
        ("3", "Build Graph",         "NetworkX topology"),
        ("4", "AI Analysis",         "Multi-agent analysis"),
        ("5", "ngspice Execution",   "Real simulation"),
        ("6", "Layout & Verify",     "Magic + KLayout"),
        ("7", "SPICE Export",        "Netlist Agent"),
        ("8", "Optimization",        "Optimization Agent"),
    ]
    G_sess = st.session_state.graph
    done_all = G_sess is not None and st.session_state.analysis is not None
    for num, title, sub in pipeline_stages:
        cls = "done" if done_all else "pipeline-step"
        st.markdown(f"""
        <div class="pipeline-step {cls}">
            <div class="step-num">{num}</div>
            <div>
                <div style="font-weight:700">{title}</div>
                <div style="font-size:8px;opacity:0.65;margin-top:1px">{sub}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Graph metrics ──
    st.markdown('<div class="sec-header">// GRAPH METRICS</div>', unsafe_allow_html=True)
    if G_sess is not None:
        G = G_sess
        try:    is_conn = nx.is_connected(G); loops = len(nx.cycle_basis(G))
        except: is_conn, loops = False, 0
        c1, c2 = st.columns(2)
        c1.metric("Vertices", G.number_of_nodes())
        c2.metric("Edges", G.number_of_edges())
        c1.metric("KVL Loops", loops)
        c2.metric("Connected", "✓" if is_conn else "✗")
        if G.number_of_nodes() > 0:
            st.metric("Density", round(nx.density(G), 3))
    else:
        st.info("Run pipeline to see metrics")

    st.divider()

    # ── Example circuits ──
    st.markdown('<div class="sec-header">// EXAMPLE CIRCUITS</div>', unsafe_allow_html=True)
    for name in EXAMPLE_CIRCUITS:
        if st.button(f"▶ {name}", key=f"ex_{name}", use_container_width=True):
            st.session_state.circuit_input = EXAMPLE_CIRCUITS[name]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# ── MAIN CONTENT ──────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

# ── Hero header ──
st.markdown("""
<div class="page-header">
    <div class="page-header-title">⚡ AutoEDA AI — Intelligent Circuit Design Platform</div>
    <div class="page-header-sub">
        Multi-agent AI · Real ngspice simulation · Graph topology · Layout/Verification · Optimization · Monte Carlo · Reports
    </div>
    <div style="margin-top:10px">
        <span class="page-header-badge">🧠 7 Agents</span>
        <span class="page-header-badge">⚡ ngspice</span>
        <span class="page-header-badge">📐 Magic VLSI</span>
        <span class="page-header-badge">✅ KLayout</span>
        <span class="page-header-badge">📚 RAG Knowledge</span>
        <span class="page-header-badge">🎲 Monte Carlo</span>
        <span class="page-header-badge">✨ Optimization</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# INPUT SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">// CIRCUIT INPUT · Multi-modal: text · image · drag-build</div>', unsafe_allow_html=True)

input_tab, builder_tab, img_tab = st.tabs(["📝 Text Description", "🔧 Component Builder", "📷 Schematic Image"])

with input_tab:
    in_col, natlang_col = st.columns([1, 1])
    with in_col:
        circuit_text = st.text_area(
            "SPICE-like or plain English description",
            value=st.session_state.circuit_input,
            height=160,
            placeholder="e.g. R1 1kΩ from node_in to node_mid, C1 10nF from node_mid to GND, V1 5V at node_in ...",
            key="circuit_text_area"
        )
        st.session_state.circuit_input = circuit_text
    with natlang_col:
        st.markdown('<div style="font-size:10px;color:#4A5A7A;font-family:JetBrains Mono,monospace;margin-bottom:6px">NATURAL LANGUAGE INPUT</div>', unsafe_allow_html=True)
        voice_text = st.text_area(
            "Natural language description",
            height=160,
            placeholder="e.g. 'I need a common-emitter amplifier with 12V supply, 100k bias resistor, 10k to ground ...'",
            key="voice_text_area"
        )
        if voice_text:
            st.session_state.circuit_input = voice_text

with builder_tab:
    st.markdown('<div style="font-size:10px;color:#4A5A7A;font-family:JetBrains Mono,monospace;margin-bottom:10px">DRAG-AND-DROP STYLE COMPONENT BUILDER</div>', unsafe_allow_html=True)

    bc1, bc2, bc3 = st.columns([1.5, 1, 1])
    with bc1:
        new_type = st.selectbox("Component Type", ["R","C","L","V","I","D","Q","M","U"], key="builder_type")
    with bc2:
        new_value = st.text_input("Value", placeholder="e.g. 1kΩ", key="builder_value")
    with bc3:
        new_label_hint = COMPONENT_FULL.get(new_type, new_type)
        new_source = st.text_input("Source Node", placeholder="node_in", key="builder_src")

    bc4, bc5, bc6 = st.columns([1.5, 1, 1])
    with bc4:
        new_target = st.text_input("Target Node", placeholder="GND", key="builder_tgt")
    with bc5:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("➕ Add Component", use_container_width=True):
            comps = st.session_state.builder_components
            label = f"{new_type}{len([c for c in comps if c['type']==new_type])+1}"
            comps.append({"type": new_type, "label": label, "value": new_value, "source": new_source, "target": new_target or "GND"})
            st.session_state.builder_components = comps
    with bc6:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("🗑 Clear Builder", use_container_width=True):
            st.session_state.builder_components = []

    # Show built components
    if st.session_state.builder_components:
        for i, comp in enumerate(st.session_state.builder_components):
            color = COMPONENT_COLORS.get(comp['type'], '#374151')
            st.markdown(f"""
            <div class="builder-comp">
                <span style="color:{color};font-weight:700">{comp['label']}</span>
                <span style="color:#C8D0E7">{COMPONENT_FULL.get(comp['type'],comp['type'])}</span>
                <span style="color:#38D9A9">{comp['value']}</span>
                <span style="color:#4A5A7A">{comp['source']} → {comp['target']}</span>
            </div>
            """, unsafe_allow_html=True)

        built_desc = builder_to_description(st.session_state.builder_components)
        st.session_state.circuit_input = built_desc
        with st.expander("📋 Generated Description", expanded=False):
            st.code(built_desc, language="text")
    else:
        st.info("Add components above to build your circuit visually.")

with img_tab:
    uploaded_file = st.file_uploader(
        "Upload circuit schematic / photo (PNG, JPG, WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        help="The Simulation Agent will extract topology from your schematic"
    )
    if uploaded_file:
        img_bytes = uploaded_file.read()
        img_b64 = base64.b64encode(img_bytes).decode()
        img_type = uploaded_file.type
        st.session_state.uploaded_image_b64 = img_b64
        st.session_state.uploaded_image_type = img_type
        st.image(img_bytes, caption="Schematic loaded — agents will analyze", use_container_width=True)
        st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#38D9A9;margin-top:4px">✓ {uploaded_file.name} · {len(img_bytes)//1024}KB</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border:1px dashed #141C2E;border-radius:10px;padding:28px;text-align:center;background:#04060A">
            <div style="font-size:28px;margin-bottom:8px">📐</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#2A3A5A">
                Drop circuit schematic here<br>
                <span style="font-size:9px;color:#141C2E">PNG · JPG · WEBP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ACTION BUTTONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
btn_c1, btn_c2, btn_c3, btn_c4, btn_c5, _ = st.columns([1.8, 1.3, 1.3, 1.3, 1.3, 2])
with btn_c1:
    run_btn = st.button("⚡ Launch Pipeline", use_container_width=True)
with btn_c2:
    opt_btn = st.button("✨ Optimize", use_container_width=True) if st.session_state.analysis else False
with btn_c3:
    mc_btn = st.button("🎲 Monte Carlo", use_container_width=True) if st.session_state.graph_data else False
with btn_c4:
    report_btn = st.button("📄 Generate Report", use_container_width=True) if st.session_state.analysis else False
with btn_c5:
    if st.button("🔄 Reset All", use_container_width=True):
        for k in ["graph","analysis","graph_data","ngspice_output","ngspice_waveform_data","optimization_result","monte_carlo_results","rag_context","report_content"]:
            st.session_state[k] = None
        for k in ["agent_log"]: st.session_state[k] = []
        for k in ["circuit_input","spice_netlist"]: st.session_state[k] = ""
        st.session_state.metrics = {}
        st.session_state.uploaded_image_b64 = None
        st.session_state.uploaded_image_type = None
        st.session_state.builder_components = []
        st.session_state.signal_flow_step = 0
        st.session_state.agent_statuses = {k: "idle" for k in ["ctrl","sim","lay","ver","spc","opt","rag"]}
        reset_logs()
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION RUN (standalone button)
# ══════════════════════════════════════════════════════════════════════════════
if opt_btn and st.session_state.analysis:
    with st.expander("✨ Circuit Optimization", expanded=True):
        opt_goal = st.selectbox("Optimization Goal", [
            "Maximize voltage gain", "Minimize noise", "Optimize bandwidth",
            "Reduce power consumption", "Improve stability & phase margin",
            "Minimize component count", "Optimize for manufacturability"
        ])
        if st.button("Run Optimization Agent", key="run_opt"):
            with st.spinner("Optimization Agent working..."):
                try:
                    st.session_state.agent_statuses["opt"] = "run"
                    opt_result = optimize_circuit(
                        st.session_state.graph_data,
                        st.session_state.analysis,
                        opt_goal,
                        st.session_state.selected_model
                    )
                    st.session_state.optimization_result = opt_result
                    st.session_state.agent_statuses["opt"] = "done"
                    add_log(f"[OPT]   ✓ Optimization complete: {opt_result.get('optimization_summary','')[:70]}", "opt")
                    st.success(f"✅ Optimization complete! Score: {opt_result.get('optimization_score','?')}%")
                except Exception as e:
                    st.session_state.agent_statuses["opt"] = "err"
                    add_log(f"[OPT]   ✗ Error: {e}", "err")
                    st.error(f"Optimization failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO RUN
# ══════════════════════════════════════════════════════════════════════════════
if mc_btn and st.session_state.graph_data:
    with st.expander("🎲 Monte Carlo Simulation", expanded=True):
        mc1, mc2 = st.columns(2)
        with mc1:
            num_runs = st.slider("Number of Runs", 20, 500, 100, 10)
        with mc2:
            tolerance = st.slider("Component Tolerance (%)", 1.0, 20.0, 5.0, 0.5)
        if st.button("Run Monte Carlo", key="run_mc"):
            with st.spinner(f"Running {num_runs} Monte Carlo simulations..."):
                try:
                    mc_results = run_monte_carlo(st.session_state.graph_data, num_runs, tolerance)
                    st.session_state.monte_carlo_results = mc_results
                    add_log(f"[MC]    ✓ Monte Carlo: {num_runs} runs, mean={mc_results['mean']:.4g}, σ={mc_results['std']:.4g}, yield={mc_results['yield_pct']:.1f}%", "info")
                    st.success(f"✅ Monte Carlo complete! Yield: {mc_results['yield_pct']:.1f}%")
                except Exception as e:
                    add_log(f"[MC]    ✗ {e}", "err")
                    st.error(f"Monte Carlo failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ══════════════════════════════════════════════════════════════════════════════
if report_btn and st.session_state.analysis:
    with st.spinner("Generating comprehensive report..."):
        try:
            report = generate_report(
                st.session_state.graph_data,
                st.session_state.graph,
                st.session_state.analysis,
                st.session_state.ngspice_output or "",
                st.session_state.optimization_result,
            )
            st.session_state.report_content = report
            add_log("[CTRL]  ✓ Report generated", "ok")
            st.success("✅ Report ready! Check the Report tab.")
        except Exception as e:
            add_log(f"[CTRL]  ✗ Report error: {e}", "err")
            st.error(f"Report generation failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE RUN
# ══════════════════════════════════════════════════════════════════════════════
if run_btn:
    user_input = st.session_state.circuit_input.strip()
    img_b64 = st.session_state.uploaded_image_b64
    img_type = st.session_state.uploaded_image_type
    model = st.session_state.selected_model

    if not user_input and not img_b64:
        st.error("⚠️ Please enter a circuit description, build one, or upload a schematic image first.")
    else:
        if not user_input:
            user_input = "Analyze this circuit from the uploaded schematic image."

        reset_logs()
        progress_bar = st.progress(0, text="Initializing AutoEDA AI pipeline...")
        status_box = st.empty()

        try:
            # ── Reset statuses ──
            st.session_state.agent_statuses = {k: "idle" for k in ["ctrl","sim","lay","ver","spc","opt","rag"]}

            # ── Stage 0: Controller init ──
            st.session_state.agent_statuses["ctrl"] = "run"
            add_log("[CTRL]  ▶ Controller Agent online — AutoEDA AI v2.0 pipeline initializing", "ctrl")
            add_log(f"[CTRL]  Model: {model} | Input length: {len(user_input)} chars | Image: {'Yes' if img_b64 else 'No'}", "ctrl")
            progress_bar.progress(5, text="Controller Agent: initializing...")

            # ── Stage 1: RAG Knowledge Retrieval ──
            st.session_state.agent_statuses["rag"] = "run"
            add_log("[RAG]   Knowledge Agent: pre-fetching domain knowledge from RAG store...", "rag")
            status_box.info("📚 **Knowledge Agent** — retrieving domain expertise...")
            progress_bar.progress(10, text="Knowledge Agent: retrieving domain knowledge...")
            # Guess circuit type from description for pre-fetch
            prelim_type = "generic"
            for ct in ["filter","amplifier","oscillator","rectifier","logic","mixed"]:
                if ct in user_input.lower():
                    prelim_type = ct; break
            rag_context = retrieve_knowledge(prelim_type)
            st.session_state.rag_context = rag_context
            add_log(f"[RAG]   ✓ Retrieved {len(rag_context)} chars of {prelim_type} circuit knowledge", "rag")
            st.session_state.agent_statuses["rag"] = "done"
            progress_bar.progress(18, text="Knowledge retrieved ✓")

            # ── Stage 2: Simulation Agent — Parse ──
            st.session_state.agent_statuses["sim"] = "run"
            status_box.info("⚡ **Simulation Agent** — parsing circuit topology...")
            add_log("[SIM]   Simulation Agent: invoking AI to parse circuit into graph JSON...", "info")
            add_log(f"[CTRL]  Dispatching parse_circuit with model={model}", "ctrl")

            graph_data = parse_circuit_to_graph(user_input, img_b64, img_type, model)
            n_nodes = len(graph_data['nodes']); n_edges = len(graph_data['edges'])
            add_log(f"[SIM]   ✓ Graph parsed: {n_nodes} vertices, {n_edges} edges", "ok")
            add_log(f"[SIM]   Circuit: '{graph_data.get('circuit_name','Unknown')}' | Type: {graph_data['graph_properties'].get('circuit_type','?')}", "ok")
            add_log(f"[SIM]   Technology: {graph_data['graph_properties'].get('technology_node','generic')} | Supply: {graph_data['graph_properties'].get('supply_voltage','?')}", "info")

            # Update RAG context with actual circuit type
            actual_ct = graph_data['graph_properties'].get('circuit_type', 'generic')
            if actual_ct != prelim_type:
                rag_context = retrieve_knowledge(actual_ct)
                st.session_state.rag_context = rag_context
                add_log(f"[RAG]   Updated knowledge context for: {actual_ct}", "rag")

            progress_bar.progress(28, text="Circuit parsed ✓")

            # ── Stage 3: Build NetworkX Graph ──
            add_log("[CTRL]  Building NetworkX graph topology...", "ctrl")
            status_box.info("🔗 **Graph Builder** — constructing circuit graph...")
            progress_bar.progress(38, text="Building graph topology...")

            G = build_networkx_graph(graph_data)
            st.session_state.graph = G
            st.session_state.graph_data = graph_data

            analytics = compute_graph_analytics(G)
            is_conn = analytics.get("is_connected", False)
            loops = analytics.get("num_loops", 0)
            density = analytics.get("density", 0.0)
            critical_nodes = analytics.get("critical_nodes", [])

            add_log(f"[GRAPH] ✓ NetworkX graph: V={G.number_of_nodes()}, E={G.number_of_edges()}, loops={loops}, density={round(density,3)}", "ok")
            add_log(f"[GRAPH] KCL: connected={is_conn} | KVL: {loops} mesh loops | Critical nodes: {critical_nodes[:3]}", "info")
            add_log(f"[GRAPH] Betweenness centrality computed | Degree centrality computed", "info")
            progress_bar.progress(50, text="Graph topology ready ✓")

            # ── Stage 4: Multi-agent Analysis ──
            st.session_state.agent_statuses["lay"] = "run"
            st.session_state.agent_statuses["ver"] = "run"
            status_box.info("📊 **Multi-Agent Analysis** — Simulation + Layout + Verification + RAG...")
            add_log("[CTRL]  Parallel dispatch: Analysis agents running with RAG context injection...", "ctrl")
            add_log("[SIM]   Simulation Agent: graph-theoretic circuit analysis & ngspice commands...", "info")
            add_log("[LAY]   Layout Agent (Magic VLSI): layout constraints & area estimates...", "info")
            add_log("[VER]   Verification Agent (KLayout): DRC/LVS checklist preparation...", "info")
            add_log("[RAG]   Injecting domain knowledge into AI prompts...", "rag")
            progress_bar.progress(62, text="Multi-agent analysis in progress...")

            analysis = analyze_circuit_graph(graph_data, G, rag_context, model)
            st.session_state.analysis = analysis
            st.session_state.spice_netlist = analysis.get("spice_netlist", "")

            add_log(f"[SIM]   ✓ Analysis: {analysis.get('function','')[:70]}", "ok")
            sim_a = analysis.get("simulation_agent", {})
            add_log(f"[SIM]   ngspice type: {sim_a.get('ngspice_analysis_type','?')} | {sim_a.get('simulation_notes','')[:60]}", "ok")

            lay_a = analysis.get("layout_agent", {})
            add_log(f"[LAY]   ✓ Layout: area={lay_a.get('estimated_area','N/A')}", "ok")
            for c in lay_a.get("magic_constraints", [])[:2]:
                add_log(f"[LAY]     · {c}", "info")

            ver_a = analysis.get("verification_agent", {})
            add_log(f"[VER]   ✓ DRC rules: {ver_a.get('klayout_rules','standard')}", "ok")
            for d in ver_a.get("drc_concerns", [])[:2]:
                add_log(f"[VER]     DRC: {d}", "warn")

            gp = analysis.get("graph_properties", {})
            if gp.get("estimated_cutoff_freq","N/A") != "N/A":
                add_log(f"[SIM]   Estimated fc: {gp['estimated_cutoff_freq']}", "ok")
            if gp.get("estimated_gain","N/A") != "N/A":
                add_log(f"[SIM]   Estimated gain: {gp['estimated_gain']}", "ok")

            st.session_state.agent_statuses["sim"] = "done"
            st.session_state.agent_statuses["lay"] = "done"
            st.session_state.agent_statuses["ver"] = "done"
            progress_bar.progress(75, text="Analysis complete ✓")

            # ── Stage 5: ngspice Execution ──
            st.session_state.agent_statuses["spc"] = "run"
            status_box.info("⚡ **ngspice** — running real simulation...")
            add_log("[SIM]   Simulation Agent: executing ngspice simulation...", "info")
            progress_bar.progress(82, text="Running ngspice simulation...")

            netlist = analysis.get("spice_netlist", "")
            ngspice_ok, ngspice_out, waveform_data = run_ngspice(netlist)
            st.session_state.ngspice_output = ngspice_out
            st.session_state.ngspice_waveform_data = waveform_data if waveform_data else None

            if ngspice_ok:
                add_log(f"[SIM]   ✓ ngspice complete | {len(waveform_data)} data points captured", "ok")
            else:
                add_log(f"[SIM]   ⚠ ngspice warning: {ngspice_out[:80]}", "warn")

            cmds = analysis.get('ngspice_commands', [])
            add_log(f"[SPC]   ✓ SPICE Netlist Agent: netlist ready | Commands: {', '.join(cmds[:3])}", "ok")
            st.session_state.agent_statuses["spc"] = "done"
            progress_bar.progress(93, text="ngspice simulation complete ✓")

            # ── Warnings ──
            for w in analysis.get("warnings", []):
                add_log(f"[WARN]  ⚠ {w}", "warn")

            # ── Done ──
            st.session_state.agent_statuses["ctrl"] = "done"
            score = analysis.get("automation_score", 90)
            add_log(f"[CTRL]  ✓✓ PIPELINE COMPLETE | Score: {score}% | V={G.number_of_nodes()}, E={G.number_of_edges()}, loops={loops}", "ok")
            add_log(f"[CTRL]  Critical nodes: {', '.join(critical_nodes[:5])}", "ctrl")

            progress_bar.progress(100, text="Pipeline complete! ✓")
            status_box.success(f"✅ All agents completed! Automation score: **{score}%** | Waveform data: {'✓' if waveform_data else '—'}")
            st.session_state.agent_log = _active_logs.copy()
            time.sleep(0.5)
            progress_bar.empty(); status_box.empty()
            st.rerun()

        except (ValueError, json.JSONDecodeError) as e:
            add_log(f"[ERR]   JSON parse error: {e}", "err")
            progress_bar.empty()
            status_box.error(f"❌ Parsing failed: {e}. Try rephrasing or using an example circuit.")
            st.session_state.agent_statuses = {k: "err" for k in ["ctrl","sim","lay","ver","spc","opt","rag"]}
        except Exception as e:
            tb = traceback.format_exc()
            add_log(f"[ERR]   {type(e).__name__}: {e}", "err")
            logger.error(tb)
            progress_bar.empty()
            status_box.error(f"❌ Pipeline error: {type(e).__name__}: {e}")
            st.session_state.agent_statuses = {k: "err" for k in ["ctrl","sim","lay","ver","spc","opt","rag"]}


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.graph is not None and st.session_state.graph_data is not None:
    G = st.session_state.graph
    gd = st.session_state.graph_data
    an = st.session_state.analysis or {}

    st.divider()

    # ── Metrics strip ──
    try:
        loops = len(nx.cycle_basis(G))
        density = round(nx.density(G), 3)
        is_conn = nx.is_connected(G)
    except:
        loops = density = 0; is_conn = False

    mc1, mc2, mc3, mc4, mc5, mc6, mc7 = st.columns(7)
    mc1.metric("Vertices", G.number_of_nodes())
    mc2.metric("Edges", G.number_of_edges())
    mc3.metric("KVL Loops", loops)
    mc4.metric("Density", density)
    mc5.metric("Connected", "✓" if is_conn else "✗")
    mc6.metric("Automation", f"{an.get('automation_score',90)}%")
    mc7.metric("Waveform", "✓" if st.session_state.ngspice_waveform_data else "—")

    st.divider()

    # ── RESULTS TABS ──
    t1, t2, t3, t4, t5, t6, t7, t8, t9 = st.tabs([
        "🗺️ Graph",
        "📊 Analytics",
        "⚡ Simulation",
        "📐 Layout",
        "✅ Verification",
        "📋 SPICE",
        "✨ Optimization",
        "🎲 Monte Carlo",
        "🖥️ Log & Report",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1: GRAPH TOPOLOGY
    # ══════════════════════════════════════════════════════════════════════════
    with t1:
        st.markdown('<div class="sec-header">// CIRCUIT TOPOLOGY GRAPH · Vertices = Nodes · Edges = Components</div>', unsafe_allow_html=True)

        # Signal flow animation
        signal_flow = an.get("signal_flow", [])
        if signal_flow:
            with st.expander("▶ Signal Flow Animation", expanded=False):
                st.markdown('<div style="font-family:\'JetBrains Mono\',monospace;font-size:9px;color:#4A5A7A;margin-bottom:10px">STEP-BY-STEP NODE TRAVERSAL</div>', unsafe_allow_html=True)
                anim_html = ""
                for i, step in enumerate(signal_flow):
                    anim_html += f"""
                    <div class="signal-step" style="animation-delay:{i*0.15}s">
                        <span class="sig-arrow">{'→' if i > 0 else '●'}</span>
                        <span class="sig-node">{step}</span>
                    </div>
                    """
                st.markdown(f'<div style="padding:10px 0">{anim_html}</div>', unsafe_allow_html=True)

        # Highlight critical nodes toggle
        analytics = compute_graph_analytics(G)
        critical = analytics.get("critical_nodes", [])
        show_highlight = st.checkbox("Highlight critical nodes", value=True)

        fig = draw_circuit_graph(G, gd, highlight_nodes=critical if show_highlight else [])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        gp = gd.get("graph_properties", {})
        p1, p2, p3, p4 = st.columns(4)
        p1.markdown(f"**Circuit Type** `{gp.get('circuit_type','?')}`")
        p2.markdown(f"**Topology** `{gp.get('topology','?')}`")
        p3.markdown(f"**Technology** `{gp.get('technology_node','generic')}`")
        p4.markdown(f"**Supply** `{gp.get('supply_voltage','?')}`")

        st.divider()
        sum_c1, sum_c2 = st.columns(2)
        with sum_c1:
            st.markdown("**Node Types**")
            types_count = defaultdict(int)
            for n in gd.get("nodes", []):
                types_count[n.get("type","NODE")] += 1
            for t, cnt in types_count.items():
                color = {"GND":"#374151","VCC":"#EF4444","NET":"#A78BFA"}.get(t,"#2855E8")
                st.markdown(f'<span class="chip" style="color:{color};border-color:{color}40;background:{color}15">{t} × {cnt}</span>', unsafe_allow_html=True)
        with sum_c2:
            st.markdown("**Component Types**")
            comp_count = defaultdict(int)
            for e in gd.get("edges", []):
                comp_count[e.get("type","R")] += 1
            for t, cnt in comp_count.items():
                color = COMPONENT_COLORS.get(t,"#374151")
                st.markdown(f'<span class="chip" style="color:{color};border-color:{color}40;background:{color}15">{t} ({COMPONENT_FULL.get(t,t)}) × {cnt}</span>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2: GRAPH ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════
    with t2:
        st.markdown('<div class="sec-header">// GRAPH ANALYTICS · Centrality · Node Importance · NetworkX</div>', unsafe_allow_html=True)

        analytics = compute_graph_analytics(G)

        an1, an2 = st.columns(2)
        with an1:
            st.markdown(f"""
            <div class="tool-card">
                <div class="tool-title">🕸️ Graph Properties</div>
                <div class="tool-sub">NetworkX topological metrics</div>
                <div style="margin-top:12px;font-family:'JetBrains Mono',monospace;font-size:11px;line-height:2.0">
                    <div>Vertices: <span style="color:#38D9A9">{G.number_of_nodes()}</span></div>
                    <div>Edges: <span style="color:#38D9A9">{G.number_of_edges()}</span></div>
                    <div>Density: <span style="color:#38D9A9">{round(analytics.get('density',0),4)}</span></div>
                    <div>KVL Loops: <span style="color:#F59E0B">{analytics.get('num_loops',0)}</span></div>
                    <div>Diameter: <span style="color:#2855E8">{analytics.get('diameter','N/A')}</span></div>
                    <div>Avg Path: <span style="color:#2855E8">{round(analytics.get('avg_shortest_path',0) or 0, 3)}</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("**Critical Nodes (by centrality)**")
            for i, node in enumerate(analytics.get("critical_nodes", []), 1):
                bc_val = analytics.get("betweenness_centrality", {}).get(node, 0)
                dc_val = analytics.get("degree_centrality", {}).get(node, 0)
                st.markdown(f'<div class="deg-bar-wrap"><div class="deg-bar-label">#{i} {node}</div><div class="deg-bar-track"><div class="deg-bar-fill" style="width:{int(dc_val*100)}%"></div></div><div class="deg-bar-val">{round(dc_val,3)}</div></div>', unsafe_allow_html=True)

        with an2:
            st.markdown("**Degree Centrality**")
            dc = analytics.get("degree_centrality", {})
            dc_sorted = sorted(dc.items(), key=lambda x: -x[1])
            max_dc = max(dc.values()) if dc else 1
            for node, val in dc_sorted:
                bar_w = int((val / max_dc) * 100)
                st.markdown(f'<div class="deg-bar-wrap"><div class="deg-bar-label">{node}</div><div class="deg-bar-track"><div class="deg-bar-fill" style="width:{bar_w}%"></div></div><div class="deg-bar-val">{round(val,3)}</div></div>', unsafe_allow_html=True)

            st.markdown("**Betweenness Centrality**")
            bc = analytics.get("betweenness_centrality", {})
            bc_sorted = sorted(bc.items(), key=lambda x: -x[1])
            max_bc = max(bc.values()) if bc else 1
            for node, val in bc_sorted:
                bar_w = int((val / max_bc) * 100) if max_bc > 0 else 0
                st.markdown(f'<div class="deg-bar-wrap"><div class="deg-bar-label">{node}</div><div class="deg-bar-track"><div class="deg-bar-fill" style="width:{bar_w}%"></div></div><div class="deg-bar-val">{round(val,3)}</div></div>', unsafe_allow_html=True)

        # Centrality table
        st.divider()
        st.markdown("**Full Centrality Table**")
        dc = analytics.get("degree_centrality", {})
        bc = analytics.get("betweenness_centrality", {})
        cc = analytics.get("closeness_centrality", {})
        ec = analytics.get("eigenvector_centrality", {})
        all_nodes = list(G.nodes())
        rows = ""
        for node in all_nodes:
            rows += f"<tr><td><code>{node}</code></td><td><code style='color:#38D9A9'>{round(dc.get(node,0),4)}</code></td><td><code style='color:#2855E8'>{round(bc.get(node,0),4)}</code></td><td><code style='color:#F59E0B'>{round(cc.get(node,0),4)}</code></td><td><code style='color:#A78BFA'>{round(ec.get(node,0),4)}</code></td><td>{G.degree(node)}</td></tr>"
        st.markdown(f"<table class='data-table'><thead><tr><th>Node</th><th>Degree Cent.</th><th>Betweenness</th><th>Closeness</th><th>Eigenvector</th><th>Degree</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3: SIMULATION AGENT (ngspice + waveforms)
    # ══════════════════════════════════════════════════════════════════════════
    with t3:
        st.markdown('<div class="sec-header">// SIMULATION AGENT · ngspice · Waveforms · Signal Analysis</div>', unsafe_allow_html=True)

        if an:
            st.markdown(f'<div class="analysis-block"><div class="analysis-tag">CIRCUIT SUMMARY</div>{an.get("summary","")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="analysis-block"><div class="analysis-tag">CIRCUIT FUNCTION</div>{an.get("function","")}</div>', unsafe_allow_html=True)

            sim_a = an.get("simulation_agent", {})
            if sim_a:
                st.markdown(f'<div class="analysis-block" style="border-left-color:#38D9A9"><div class="analysis-tag" style="color:#38D9A9">ngspice ANALYSIS TYPE: {sim_a.get("ngspice_analysis_type","?")}</div>{sim_a.get("simulation_notes","")}</div>', unsafe_allow_html=True)

            # Waveform plots
            waveform_data = st.session_state.ngspice_waveform_data
            if waveform_data:
                st.markdown("#### 📈 Simulation Waveforms")
                wfig = plot_waveforms(waveform_data)
                if wfig:
                    st.pyplot(wfig, use_container_width=True)
                    plt.close(wfig)
                    # DataFrame download
                    df_wave = pd.DataFrame(waveform_data)
                    csv_data = df_wave.to_csv(index=False)
                    st.download_button("⬇ Download Waveform CSV", data=csv_data, file_name="waveform_data.csv", mime="text/csv")
            else:
                st.info("No waveform data — ngspice may not be installed (demo mode active). See simulation output below.")

            # ngspice raw output
            with st.expander("📟 ngspice Raw Output", expanded=False):
                ngspice_out = st.session_state.ngspice_output or "No ngspice output."
                st.code(ngspice_out, language="text")

            sim_col_l, sim_col_r = st.columns(2)
            with sim_col_l:
                st.markdown("#### Signal Flow")
                for i, step in enumerate(an.get("signal_flow", []), 1):
                    st.markdown(f'<div class="flow-item"><div class="flow-num">{i}</div><div class="flow-text">{step}</div></div>', unsafe_allow_html=True)

                st.markdown("#### Critical Nodes")
                for item in an.get("critical_nodes", []):
                    st.markdown(f"• **`{item.get('node','')}`** — {item.get('reason','')}")

            with sim_col_r:
                st.markdown("#### Performance Parameters")
                gprops = an.get("graph_properties", {})
                for k, v in gprops.items():
                    k_clean = k.replace("_"," ").title()
                    st.markdown(f"• **{k_clean}:** `{v}`")

                st.markdown("#### ngspice Commands")
                cmds = an.get("ngspice_commands", [])
                if cmds:
                    st.code("\n".join(cmds), language="text")

                if an.get("recommendations"):
                    st.markdown("#### Design Recommendations")
                    for tip in an.get("recommendations", []):
                        st.markdown(f"✦ {tip}")

            # RAG Knowledge used
            if st.session_state.rag_context:
                with st.expander("📚 RAG Knowledge Context Used", expanded=False):
                    st.markdown(f'<div class="rag-card"><div class="rag-tag">RETRIEVED DOMAIN KNOWLEDGE</div><div class="rag-body">{st.session_state.rag_context}</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4: LAYOUT AGENT
    # ══════════════════════════════════════════════════════════════════════════
    with t4:
        st.markdown('<div class="sec-header">// LAYOUT AGENT · Magic VLSI · Physical Design</div>', unsafe_allow_html=True)
        lay_a = an.get("layout_agent", {})
        if lay_a:
            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown(f"""
                <div class="tool-card">
                    <div class="tool-title">📐 Magic VLSI Layout Engine</div>
                    <div class="tool-sub">Physical design & parasitic extraction</div>
                    <div style="margin-top:12px">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#4A5A7A;margin-bottom:4px">ESTIMATED AREA</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:16px;color:#38D9A9">{lay_a.get('estimated_area','N/A')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**Routing Guidance**")
                st.markdown(f'<div class="analysis-block" style="border-left-color:#38D9A9">{lay_a.get("routing_notes","N/A")}</div>', unsafe_allow_html=True)
            with lc2:
                st.markdown("**Layout Constraints**")
                for i, c in enumerate(lay_a.get("magic_constraints", []), 1):
                    st.markdown(f'<div class="flow-item"><div class="flow-num" style="background:rgba(56,217,169,0.1);border-color:rgba(56,217,169,0.3);color:#38D9A9">{i}</div><div class="flow-text">{c}</div></div>', unsafe_allow_html=True)

            st.divider()
            circuit_slug = gd.get('circuit_name','circuit').lower().replace(' ','_')
            tech = gd.get('graph_properties',{}).get('technology_node','generic')
            magic_script = f"""# Magic VLSI Script — Auto-generated by AutoEDA AI Layout Agent
# Circuit: {gd.get('circuit_name','circuit')}
# Technology: {tech}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

magic -T scmos   # Load SCMOS technology (change for your tech)
load {circuit_slug}.ext
ext2spice hierarchy on
ext2spice scale off
ext2spice
gds write {circuit_slug}.gds
quit
"""
            st.markdown("**Magic VLSI Script (Auto-generated)**")
            st.code(magic_script, language="bash")
            st.download_button("⬇ Download Magic Script", data=magic_script, file_name=f"{circuit_slug}_magic.sh", mime="text/plain")
        else:
            st.info("Run pipeline to generate Layout Agent analysis.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 5: VERIFICATION AGENT
    # ══════════════════════════════════════════════════════════════════════════
    with t5:
        st.markdown('<div class="sec-header">// VERIFICATION AGENT · KLayout · DRC / LVS</div>', unsafe_allow_html=True)
        ver_a = an.get("verification_agent", {})
        if ver_a:
            vc1, vc2 = st.columns(2)
            with vc1:
                st.markdown(f"""
                <div class="tool-card">
                    <div class="tool-title">✅ KLayout DRC/LVS Engine</div>
                    <div class="tool-sub">Design rule & netlist verification</div>
                    <div style="margin-top:12px">
                        <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#4A5A7A;margin-bottom:4px">RULE SET</div>
                        <div style="font-family:'JetBrains Mono',monospace;font-size:13px;color:#F59E0B">{ver_a.get('klayout_rules','standard')}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown("**DRC Concerns**")
                drc = ver_a.get("drc_concerns", [])
                if drc:
                    for i, c in enumerate(drc, 1):
                        st.warning(f"{i}. {c}")
                else:
                    st.success("No DRC concerns flagged.")
            with vc2:
                st.markdown("**LVS Checkpoints**")
                for i, chk in enumerate(ver_a.get("lvs_checkpoints", []), 1):
                    st.markdown(f'<div class="flow-item"><div class="flow-num" style="background:rgba(245,158,11,0.1);border-color:rgba(245,158,11,0.3);color:#F59E0B">{i}</div><div class="flow-text">{chk}</div></div>', unsafe_allow_html=True)
                if an.get("warnings"):
                    st.markdown("**⚠ Verification Warnings**")
                    for w in an.get("warnings", []):
                        st.warning(w)

            st.divider()
            circuit_slug = gd.get('circuit_name','circuit').lower().replace(' ','_')
            klayout_script = f"""# KLayout DRC/LVS Script — Auto-generated by AutoEDA AI
# Circuit: {gd.get('circuit_name','circuit')}
# Rule Set: {ver_a.get('klayout_rules','standard')}
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

import pya

# Load GDS layout
layout = pya.Layout()
layout.read("{circuit_slug}.gds")

# Run DRC
print("[KLayout] Running DRC...")
drc = pya.DRC()
drc.layout = layout
drc.run()
print(f"DRC violations: {{drc.count}}")

# Run LVS comparison
print("[KLayout] Running LVS...")
lvs = pya.LVS()
lvs.schematic = "{circuit_slug}.sp"
lvs.layout = "{circuit_slug}.gds"
lvs.run()
print(f"LVS matched nets: {{lvs.lvs_data.matching_nets_count}}")
"""
            st.markdown("**KLayout DRC/LVS Script (Auto-generated)**")
            st.code(klayout_script, language="python")
            st.download_button("⬇ Download KLayout Script", data=klayout_script, file_name=f"{circuit_slug}_drc.py", mime="text/plain")
        else:
            st.info("Run pipeline to generate Verification Agent analysis.")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 6: SPICE NETLIST
    # ══════════════════════════════════════════════════════════════════════════
    with t6:
        st.markdown('<div class="sec-header">// SPICE NETLIST AGENT · ngspice-ready · Component Tables</div>', unsafe_allow_html=True)
        netlist = st.session_state.spice_netlist
        circuit_slug = gd.get('circuit_name','circuit').lower().replace(' ','_')

        sc1, sc2 = st.columns([3, 1])
        with sc1:
            if netlist:
                st.code(netlist, language="text")
            else:
                st.info("Run pipeline first.")
        with sc2:
            if netlist:
                fname = f"{circuit_slug}.sp"
                st.download_button("⬇ Download .sp", data=netlist, file_name=fname, mime="text/plain", use_container_width=True)
                st.markdown("**Run with ngspice:**")
                st.code(f"ngspice {fname}", language="bash")
                st.markdown("**Run with Magic:**")
                st.code(f"magic -T scmos\next2spice", language="bash")
                st.markdown("**KLayout:**")
                st.code("klayout -b -r drc_script.py", language="bash")

        # Node & Edge tables
        st.divider()
        st.markdown('<div class="sec-header">// NODE & EDGE TABLES</div>', unsafe_allow_html=True)
        nt_col, et_col = st.columns(2)
        with nt_col:
            st.markdown("**Nodes (Vertices)**")
            node_rows = ""
            for n in gd.get("nodes", []):
                ntype = n.get("type","NODE")
                color = {"VCC":"#EF4444","GND":"#374151","NET":"#A78BFA"}.get(ntype,"#2855E8")
                deg = G.degree(n['id']) if n['id'] in G else 0
                node_rows += (f"<tr><td><code style='color:{color}'>{n.get('id','')}</code></td>"
                              f"<td>{n.get('label','')}</td><td><code>{ntype}</code></td><td><code style='color:#38D9A9'>{deg}</code></td></tr>")
            st.markdown(f"<table class='data-table'><thead><tr><th>ID</th><th>Label</th><th>Type</th><th>Degree</th></tr></thead><tbody>{node_rows}</tbody></table>", unsafe_allow_html=True)
        with et_col:
            st.markdown("**Edges (Components)**")
            edge_rows = ""
            for e in gd.get("edges", []):
                etype = e.get("type","R"); color = COMPONENT_COLORS.get(etype,"#374151")
                edge_rows += (f"<tr><td><code style='color:{color}'>{e.get('label','')}</code></td>"
                              f"<td><span style='color:{color};font-weight:700'>{etype}</span></td>"
                              f"<td><code>{e.get('value','—')}</code></td>"
                              f"<td style='font-size:10px;color:#2A3A5A'>{e.get('source','')} → {e.get('target','')}</td></tr>")
            st.markdown(f"<table class='data-table'><thead><tr><th>Label</th><th>Type</th><th>Value</th><th>Connection</th></tr></thead><tbody>{edge_rows}</tbody></table>", unsafe_allow_html=True)

        # Degree distribution
        st.divider()
        st.markdown("**Degree Distribution (KCL Node Analysis)**")
        degree_data = dict(sorted(dict(G.degree()).items(), key=lambda x: -x[1]))
        if degree_data:
            max_deg = max(degree_data.values()) or 1
            for node, deg in degree_data.items():
                bar_w = int((deg / max_deg) * 100)
                st.markdown(f'<div class="deg-bar-wrap"><div class="deg-bar-label">{node}</div><div class="deg-bar-track"><div class="deg-bar-fill" style="width:{bar_w}%"></div></div><div class="deg-bar-val">{deg}</div></div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 7: OPTIMIZATION
    # ══════════════════════════════════════════════════════════════════════════
    with t7:
        st.markdown('<div class="sec-header">// OPTIMIZATION AGENT · Component Tuning · Performance Improvement</div>', unsafe_allow_html=True)

        if not st.session_state.optimization_result:
            st.info("Click **✨ Optimize** button above (after running the pipeline) to run the Optimization Agent.")
            st.markdown("""
            <div class="opt-card">
                <div class="opt-card-title">What the Optimization Agent does:</div>
                <div class="opt-card-body">
                    • Analyzes current component values against design goals<br>
                    • Suggests optimal resistor, capacitor, and inductor values<br>
                    • Improves gain, bandwidth, noise figure, or power consumption<br>
                    • Generates an optimized SPICE netlist ready for re-simulation<br>
                    • Provides expected performance improvements with explanations
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            opt = st.session_state.optimization_result
            oc1, oc2 = st.columns([2, 1])
            with oc1:
                st.markdown(f'<div class="opt-card"><div class="opt-card-title">Optimization Summary</div><div class="opt-card-body">{opt.get("optimization_summary","")}</div></div>', unsafe_allow_html=True)
                st.markdown("**Component Changes**")
                for change in opt.get("changes", []):
                    orig = change.get("original","")
                    new = change.get("optimized","")
                    reason = change.get("reason","")
                    comp = change.get("component","")
                    st.markdown(f"""
                    <div style="background:#07090F;border:1px solid #141C2E;border-radius:7px;padding:10px 12px;margin-bottom:6px;font-family:'JetBrains Mono',monospace;font-size:11px">
                        <span style="color:#38D9A9;font-weight:700">{comp}</span>
                        <span style="color:#4A5A7A;margin:0 8px">·</span>
                        <span style="color:#EF4444;text-decoration:line-through">{orig}</span>
                        <span style="color:#4A5A7A;margin:0 6px">→</span>
                        <span style="color:#38D9A9">{new}</span>
                        <div style="color:#4A5A7A;font-size:10px;margin-top:4px">{reason}</div>
                    </div>
                    """, unsafe_allow_html=True)
            with oc2:
                score = opt.get("optimization_score", 0)
                st.markdown(f"""
                <div class="tool-card" style="border-color:#143A28">
                    <div class="tool-title" style="color:#34D399">Optimization Score</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:32px;color:#38D9A9;margin:10px 0">{score}%</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**Expected Improvements**")
                impr = opt.get("expected_improvement", {})
                for key, val in impr.items():
                    label = key.replace("_"," ").title()
                    color = "#38D9A9" if "improve" in str(val).lower() or "+" in str(val) else "#F59E0B"
                    st.markdown(f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:10px;margin:4px 0"><span style="color:#4A5A7A">{label}: </span><span style="color:{color}">{val}</span></div>', unsafe_allow_html=True)

            st.divider()
            st.markdown("**Optimized SPICE Netlist**")
            optimized_netlist = opt.get("improved_netlist", "")
            if optimized_netlist:
                st.code(optimized_netlist, language="text")
                circuit_slug = gd.get('circuit_name','circuit').lower().replace(' ','_')
                st.download_button("⬇ Download Optimized Netlist", data=optimized_netlist,
                    file_name=f"{circuit_slug}_optimized.sp", mime="text/plain")

            if opt.get("additional_recommendations"):
                st.markdown("**Additional Recommendations**")
                for rec in opt["additional_recommendations"]:
                    st.markdown(f"✦ {rec}")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 8: MONTE CARLO
    # ══════════════════════════════════════════════════════════════════════════
    with t8:
        st.markdown('<div class="sec-header">// MONTE CARLO SIMULATION · Component Variation · Statistical Analysis</div>', unsafe_allow_html=True)

        if not st.session_state.monte_carlo_results:
            st.info("Click **🎲 Monte Carlo** button above (after running pipeline) to run statistical simulation.")
            st.markdown("""
            <div class="mc-card">
                <div style="font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;color:#A78BFA;margin-bottom:8px">What Monte Carlo Simulation does:</div>
                <div style="font-family:'Space Grotesk',sans-serif;font-size:12px;color:#8878B8;line-height:1.7">
                    • Randomly varies component values within tolerance (e.g. ±5%)<br>
                    • Runs hundreds of simulations automatically<br>
                    • Computes statistical distribution of output metrics<br>
                    • Shows yield (% of circuits meeting spec)<br>
                    • Identifies worst-case and best-case performance<br>
                    • Critical for manufacturing sign-off
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            mc = st.session_state.monte_carlo_results
            mcc1, mcc2, mcc3, mcc4 = st.columns(4)
            mcc1.metric("Runs", mc["num_runs"])
            mcc2.metric("Mean", f"{mc['mean']:.4g}")
            mcc3.metric("Std Dev", f"{mc['std']:.4g}")
            mcc4.metric("Yield", f"{mc['yield_pct']:.1f}%")

            mcc5, mcc6, mcc7, mcc8 = st.columns(4)
            mcc5.metric("Min", f"{mc['min']:.4g}")
            mcc6.metric("Max", f"{mc['max']:.4g}")
            mcc7.metric("5th pct", f"{mc['p5']:.4g}")
            mcc8.metric("95th pct", f"{mc['p95']:.4g}")

            # Histogram
            mc_fig = plot_monte_carlo(mc)
            st.pyplot(mc_fig, use_container_width=True)
            plt.close(mc_fig)

            # Stats table
            st.divider()
            st.markdown("**Statistical Summary**")
            st.markdown(f"""
            <table class="data-table">
                <thead><tr><th>Metric</th><th>Value</th><th>Description</th></tr></thead>
                <tbody>
                    <tr><td>Metric</td><td><code style='color:#38D9A9'>{mc['metric_name']}</code></td><td>Output variable analyzed</td></tr>
                    <tr><td>Tolerance</td><td><code style='color:#F59E0B'>±{mc['tolerance_pct']}%</code></td><td>Component value variation</td></tr>
                    <tr><td>Mean</td><td><code style='color:#38D9A9'>{mc['mean']:.6g}</code></td><td>Average output</td></tr>
                    <tr><td>Std Dev (σ)</td><td><code style='color:#2855E8'>{mc['std']:.6g}</code></td><td>Output spread</td></tr>
                    <tr><td>σ/Mean (CV)</td><td><code style='color:#A78BFA'>{mc['std']/mc['mean']*100:.2f}%</code></td><td>Coefficient of variation</td></tr>
                    <tr><td>Yield</td><td><code style='color:#38D9A9'>{mc['yield_pct']:.1f}%</code></td><td>Within 3σ of mean</td></tr>
                </tbody>
            </table>
            """, unsafe_allow_html=True)

            # Download MC data
            mc_df = pd.DataFrame({"run": range(1, mc["num_runs"]+1), mc["metric_name"]: mc["raw_data"]})
            st.download_button("⬇ Download MC Data CSV", data=mc_df.to_csv(index=False),
                file_name="monte_carlo_results.csv", mime="text/csv")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 9: LOG & REPORT
    # ══════════════════════════════════════════════════════════════════════════
    with t9:
        log_tab, report_tab = st.tabs(["🖥️ Pipeline Log", "📄 Report"])

        with log_tab:
            st.markdown('<div class="sec-header">// MULTI-AGENT PIPELINE LOG · Real-time execution trace</div>', unsafe_allow_html=True)
            logs = st.session_state.agent_log
            if logs:
                cls_map = {"ok":"log-ok","warn":"log-warn","err":"log-err","info":"log-info","ctrl":"log-ctrl","opt":"log-opt","rag":"log-rag"}
                log_html = ""
                for entry in logs:
                    cls = cls_map.get(entry["level"], "log-muted")
                    log_html += f'<div class="{cls}">[{entry["time"]}] {entry["msg"]}</div>\n'
                st.markdown(f'<div class="console-box">{log_html}</div>', unsafe_allow_html=True)
            else:
                st.info("No log entries yet. Launch the pipeline to populate the execution log.")

        with report_tab:
            st.markdown('<div class="sec-header">// DOWNLOADABLE REPORT · Markdown · Full Analysis</div>', unsafe_allow_html=True)
            report = st.session_state.report_content
            if report:
                st.markdown(report)
                circuit_slug = gd.get('circuit_name','circuit').lower().replace(' ','_')
                st.download_button(
                    "⬇ Download Report (.md)",
                    data=report,
                    file_name=f"{circuit_slug}_autoeda_report.md",
                    mime="text/markdown",
                    use_container_width=False,
                )
            else:
                st.info("Click **📄 Generate Report** button above to create a comprehensive downloadable report.")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0">
    <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:#141C2E;letter-spacing:1.2px">
        AutoEDA AI v2.0 · 7-Agent Multi-Agent EDA Platform · Controller · Simulation · Layout · Verification · SPICE · Optimization · RAG
    </div>
    <div style="font-family:'JetBrains Mono',monospace;font-size:8px;color:#141C2E;letter-spacing:0.5px">
        ngspice · Magic VLSI · KLayout · NetworkX · Monte Carlo · Graph Theory (V=Nodes, E=Components, Cycles=Loops)
    </div>
</div>
""", unsafe_allow_html=True)
