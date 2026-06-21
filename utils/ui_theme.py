"""
InsightML Studio — Central UI Theme
=====================================
All reusable CSS helpers and component functions in one place.
Import and call inject_css() at the top of every page.
DO NOT modify any backend/ML logic here.
"""

from __future__ import annotations
import streamlit as st

# ── Color tokens ─────────────────────────────────────────────────────────────
_COLORS = {
    "primary":        "#7C3AED",   # purple-600
    "primary_light":  "#A78BFA",   # purple-400
    "primary_dark":   "#5B21B6",   # purple-800
    "secondary":      "#2563EB",   # blue-600
    "accent_pink":    "#EC4899",   # pink-500
    "success":        "#10B981",   # emerald-500
    "warning":        "#F59E0B",   # amber-500
    "danger":         "#EF4444",   # red-500
    "info":           "#06B6D4",   # cyan-500

    # Dark mode
    "dark_bg":        "#0D0F1A",
    "dark_surface":   "#13162B",
    "dark_card":      "#191D35",
    "dark_border":    "#2A2F52",
    "dark_muted":     "#64748B",
    "dark_text":      "#E2E8F0",
    "dark_subtext":   "#94A3B8",

    # Light mode
    "light_bg":       "#F8F9FF",
    "light_surface":  "#FFFFFF",
    "light_card":     "#FFFFFF",
    "light_border":   "#E4E6F0",
    "light_muted":    "#6B7280",
    "light_text":     "#1E1B4B",
    "light_subtext":  "#6B7280",
}

_MASTER_CSS = """
<style>
/* ══════════════════════════════════════════════════════════════════════
   GOOGLE FONTS
══════════════════════════════════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ══════════════════════════════════════════════════════════════════════
   CSS VARIABLES — Dark Mode (default)
══════════════════════════════════════════════════════════════════════ */
:root {
  --primary:        #7C3AED;
  --primary-light:  #C4B5FD;
  --primary-dark:   #5B21B6;
  --secondary:      #60A5FA;
  --accent-pink:    #EC4899;
  --success:        #10B981;
  --warning:        #F59E0B;
  --danger:         #EF4444;
  --info:           #22D3EE;

  --bg:             #070B18;
  --surface:        #111827;
  --card:           #111827;
  --border:         rgba(255,255,255,0.08);
  --muted:          #94A3B8;
  --text:           #F8FAFC;
  --subtext:        #CBD5E1;
  --sidebar-bg:     rgba(9,12,24,0.82);

  --radius-sm:      10px;
  --radius-md:      18px;
  --radius-lg:      24px;
  --radius-xl:      32px;

  --shadow-sm:      0 16px 45px rgba(0,0,0,0.22);
  --shadow-md:      0 28px 80px rgba(0,0,0,0.28);
  --shadow-lg:      0 40px 120px rgba(0,0,0,0.32);
  --glow-primary:   0 0 28px rgba(124,58,237,0.24);
  --glow-blue:      0 0 20px rgba(37,99,235,0.25);

  --font-main:      'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono:      'JetBrains Mono', ui-monospace, monospace;

  --transition:     all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ══════════════════════════════════════════════════════════════════════
   LIGHT MODE OVERRIDES
══════════════════════════════════════════════════════════════════════ */
[data-theme="light"] {
  --bg:             #F8F9FF;
  --surface:        #FFFFFF;
  --card:           #FFFFFF;
  --border:         #E4E6F0;
  --muted:          #6B7280;
  --text:           #1E1B4B;
  --subtext:        #6B7280;
  --sidebar-bg:     rgba(255,255,255,0.96);
  --shadow-sm:      0 2px 8px rgba(124,58,237,0.08);
  --shadow-md:      0 4px 20px rgba(124,58,237,0.12);
  --shadow-lg:      0 8px 40px rgba(124,58,237,0.16);
  --glow-primary:   0 0 20px rgba(124,58,237,0.12);
}

[data-theme="light"] {
  color-scheme: light;
}
[data-theme="light"] [data-testid="stSidebar"] {
  border-right: 1px solid rgba(148,163,184,0.2) !important;
  box-shadow: 0 24px 80px rgba(15,23,42,0.08) !important;
}
[data-theme="light"] .sidebar-logo {
  border-bottom: 1px solid rgba(148,163,184,0.14) !important;
}
[data-theme="light"] [data-testid="stSidebar"] div[data-testid="stButton"] > button {
  background: rgba(15,23,42,0.04) !important;
  border: 1px solid rgba(148,163,184,0.16) !important;
  color: var(--text) !important;
}
[data-theme="light"] [data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
  background: rgba(124,58,237,0.08) !important;
}
[data-theme="light"] [data-testid="stSidebar"] div[data-testid="stButton"] > button[disabled] {
  background: linear-gradient(135deg, rgba(124,58,237,0.14), rgba(106,17,203,0.14)) !important;
  border-color: rgba(124,58,237,0.18) !important;
  box-shadow: 0 12px 32px rgba(124,58,237,0.08) !important;
  color: var(--text) !important;
}
[data-theme="light"] .sidebar-section-title {
  color: var(--muted) !important;
}
[data-theme="light"] .page-header {
  border-color: rgba(226,232,240,0.6) !important;
}

/* ══════════════════════════════════════════════════════════════════════
   BASE LAYOUT
══════════════════════════════════════════════════════════════ */
html, body, [data-testid="stAppViewContainer"] {
  font-family: var(--font-main) !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
  min-height: 100%;
  margin: 0 !important;
  padding: 0 !important;
}

body > div {
  margin: 0 !important;
  padding-top: 0 !important;
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > div,
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewContainer"] > header,
header.stAppHeader,
header.stAppToolbar,
header.stAppHeader[data-testid="stHeader"],
header.stAppToolbar[data-testid="stToolbar"] {
  margin-top: 0 !important;
  padding-top: 0 !important;
}

[data-testid="stAppViewContainer"] {
  background: radial-gradient(circle at top left, rgba(124,58,237,0.12), transparent 26%),
              radial-gradient(circle at bottom right, rgba(6,182,212,0.08), transparent 26%),
              var(--bg) !important;
}


  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}

body > section,
body > div > section,
body > div > div > section,
[data-testid="stAppViewContainer"] > section,
[data-testid="stAppViewContainer"] > div > section {
  margin-top: 0 !important;
  padding-top: 0 !important;
}

.block-container {
  padding: 1.5rem 2.25rem 3rem !important;
  max-width: 1400px !important;
}

#MainMenu, footer, [data-testid="stToolbar"],
header.stAppHeader,
header.stAppToolbar,
header[role="banner"],
header[data-testid="stHeader"],
header[data-testid="stToolbar"],
.stAppToolbar,
div[data-testid="stHeader"],
div[data-testid="stToolbar"] {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* Hide Streamlit native page navigation/search in the sidebar */
[data-testid="stSidebarNav"],
[data-testid="stSidebarNav"] *,
[data-testid="stSidebar"] [data-testid="stTextInput"] {
  display: none !important;
}

/* Remove extra top padding so content begins immediately below the browser */
.main .block-container,
.block-container,
[data-testid="stAppViewContainer"],
.stAppHeader,
.stAppToolbar,
.stMain,
.stAppViewContainer,
.stApp {
  padding-top: 0 !important;
  margin-top: 0 !important;
}

header.stAppHeader,
header.stAppHeader *,
.stAppHeader,
.stAppHeader *,
.stAppToolbar,
.stAppToolbar *,
header[data-testid="stHeader"],
header[data-testid="stToolbar"],
div[data-testid="stHeader"],
div[data-testid="stToolbar"] {
  display: none !important;
  visibility: hidden !important;
  height: 0 !important;
  min-height: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow: hidden !important;
}


main, .main {
  padding-top: 0 !important;
  margin-top: 0 !important;
}


/* ══════════════════════════════════════════════════════════════════════
   SIDEBAR
══════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  width: 280px !important;
  min-width: 280px !important;
  max-width: 280px !important;
  background: var(--sidebar-bg) !important;
  border-right: 1px solid rgba(255,255,255,0.08) !important;
  backdrop-filter: blur(22px);
  box-shadow: 0 40px 140px rgba(0,0,0,0.24) !important;
  border-radius: 0 32px 32px 0 !important;
  transition: width 0.35s ease, transform 0.35s ease, opacity 0.35s ease !important;
  position: relative !important;
  overflow: hidden !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 0 !important;
}
[data-testid="stSidebar"] * {
  color: var(--text) !important;
}

/* Sidebar logo area */
.sidebar-logo {
  padding: 2.25rem 1.75rem 1.5rem;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  margin-bottom: 1.75rem;
  display: grid;
  gap: 1rem;
  min-height: 220px;
  background: rgba(255,255,255,0.05);
  border-radius: 28px;
}
.sidebar-logo-brand {
  display: flex;
  align-items: center;
  gap: 1.25rem;
}
.sidebar-logo-icon {
  width: 72px;
  height: 72px;
  border-radius: 22px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, rgba(124,58,237,0.35), rgba(37,99,235,0.25));
  font-size: 2.3rem;
}
.sidebar-logo-title {
  font-size: 1.9rem;
  font-weight: 900;
  letter-spacing: -0.05em;
  line-height: 1.05;
  background: linear-gradient(135deg, var(--primary-light), var(--info));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.sidebar-logo-sub {
  font-size: 0.95rem;
  color: var(--subtext) !important;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  margin-top: 0.2rem;
}
.sidebar-logo-meta {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  font-size: 0.95rem;
  color: var(--subtext);
}
.sidebar-status-dot {
  width: 14px;
  height: 14px;
  border-radius: 999px;
  background: linear-gradient(135deg, #A78BFA, #38BDF8);
  box-shadow: 0 0 24px rgba(124,58,237,0.35);
}
.sidebar-logo-version {
  font-weight: 900;
  letter-spacing: 0.12em;
}

.sidebar-card {
  margin: 0 1rem 1.75rem;
  padding: 1.35rem 1rem 1.3rem;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 28px;
  box-shadow: 0 16px 48px rgba(9,12,24,0.22);
}
.sidebar-card-title {
  font-size: 0.75rem;
  font-weight: 800;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.14em;
  margin-bottom: 0.9rem;
}
.sidebar-card-body {
  font-size: 0.95rem;
  line-height: 1.8;
  color: var(--text);
}
.sidebar-card-body strong {
  color: var(--primary-light);
}

.sidebar-section-title {
  padding: 1rem 1.75rem 0.55rem !important;
  font-size: 0.72rem !important;
  font-weight: 700 !important;
  color: var(--muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.12em !important;
  margin-bottom: 0.35rem !important;
}

[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
  margin: 0 1rem 1.2rem !important;
  padding: 1rem !important;
  border: 1px solid rgba(255,255,255,0.16) !important;
  border-radius: 24px !important;
  background: rgba(255,255,255,0.03) !important;
  box-shadow: inset 0 0 0 rgba(255,255,255,0.04) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"]:hover {
  background: rgba(124,58,237,0.08) !important;
  border-color: rgba(124,58,237,0.22) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div {
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] .stMarkdown > div {
  background: transparent !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button {
  width: 100% !important;
  text-align: left !important;
  justify-content: flex-start !important;
  gap: 0.75rem !important;
  margin-bottom: 1rem !important;
  padding: 1rem 1rem !important;
  background: rgba(255,255,255,0.05) !important;
  color: var(--text) !important;
  border: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: 18px !important;
  box-shadow: inset 0 0 0 rgba(255,255,255,0) !important;
  position: relative !important;
  transition: var(--transition) !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.01em !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 22px 40px rgba(124,58,237,0.12) !important;
  background: rgba(124,58,237,0.12) !important;
  border-color: rgba(124,58,237,0.22) !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button[disabled] {
  background: linear-gradient(135deg, rgba(124,58,237,0.95), rgba(106,17,203,0.95)) !important;
  color: #ffffff !important;
  border-color: rgba(124,58,237,0.55) !important;
  box-shadow: 0 20px 40px rgba(124,58,237,0.28) !important;
}
[data-testid="stSidebar"] div[data-testid="stButton"] > button[disabled]::before {
  content: "";
  position: absolute !important;
  left: 0; top: 0;
  width: 4px; height: 100%;
  border-radius: 0 0 0 18px;
  background: linear-gradient(180deg, var(--primary), var(--info));
}

.sidebar-footer {
  padding: 1.25rem 1.75rem 1.35rem;
  border-top: 1px solid rgba(255,255,255,0.08);
  margin-top: auto;
}
.sidebar-footer-text {
  font-size: 0.72rem;
  color: var(--muted) !important;
  line-height: 1.6;
}
.sidebar-version-badge {
  display: inline-block;
  background: rgba(124,58,237,0.16);
  border: 1px solid rgba(124,58,237,0.35);
  color: var(--primary-light) !important;
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.06em;
}

/* ══════════════════════════════════════════════════════════════════════
   PAGE HEADER
══════════════════════════════════════════════════════════════════════ */
.page-header {
  background: linear-gradient(135deg, rgba(124,58,237,0.18) 0%,
              rgba(37,99,235,0.12) 50%, rgba(6,182,212,0.08) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  padding: 2rem 2.5rem 1.75rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
  box-shadow: var(--shadow-md), var(--glow-primary);
}
.page-header::before {
  content: "";
  position: absolute;
  top: -60px; right: -60px;
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(124,58,237,0.25) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}
.page-header::after {
  content: "";
  position: absolute;
  bottom: -40px; left: 30%;
  width: 150px; height: 150px;
  background: radial-gradient(circle, rgba(6,182,212,0.15) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}
.page-header-icon {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  display: block;
  animation: float 3s ease-in-out infinite;
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-4px); }
}
.page-header-title {
  font-size: 2rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--primary-light), var(--info), #F0ABFC);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0;
  line-height: 1.1;
  letter-spacing: -0.03em;
}
.page-header-subtitle {
  color: var(--subtext);
  font-size: 0.95rem;
  margin-top: 0.5rem;
  font-weight: 400;
  line-height: 1.5;
}
.page-header-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  background: rgba(124,58,237,0.15);
  border: 1px solid rgba(124,58,237,0.35);
  color: var(--primary-light);
  -webkit-text-fill-color: var(--primary-light);
  border-radius: 20px;
  padding: 3px 12px;
  font-size: 0.72rem;
  font-weight: 700;
  margin-top: 1rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}

/* ══════════════════════════════════════════════════════════════════════
   KPI / METRIC CARDS
══════════════════════════════════════════════════════════════════════ */
.kpi-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.25rem 1.5rem;
  position: relative;
  overflow: hidden;
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}
.kpi-card:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-md), var(--glow-primary);
  border-color: rgba(124,58,237,0.4);
}
.kpi-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.kpi-card.purple::before { background: linear-gradient(90deg, var(--primary), var(--primary-light)); }
.kpi-card.blue::before   { background: linear-gradient(90deg, var(--secondary), var(--info)); }
.kpi-card.green::before  { background: linear-gradient(90deg, var(--success), #34D399); }
.kpi-card.orange::before { background: linear-gradient(90deg, var(--warning), #FBD96A); }
.kpi-card.pink::before   { background: linear-gradient(90deg, var(--accent-pink), #F9A8D4); }
.kpi-card.red::before    { background: linear-gradient(90deg, var(--danger), #FCA5A5); }

.kpi-icon {
  font-size: 1.75rem;
  margin-bottom: 0.5rem;
  display: block;
}
.kpi-label {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--subtext);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.4rem;
}
.kpi-value {
  font-size: 1.9rem;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.03em;
  line-height: 1;
}
.kpi-delta {
  font-size: 0.78rem;
  font-weight: 500;
  margin-top: 0.4rem;
  color: var(--subtext);
}
.kpi-delta.up   { color: var(--success); }
.kpi-delta.down { color: var(--danger); }

/* Override Streamlit's native metric widget */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 1rem 1.25rem !important;
  box-shadow: var(--shadow-sm) !important;
  transition: var(--transition) !important;
}
[data-testid="metric-container"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow-md), var(--glow-primary) !important;
  border-color: rgba(124,58,237,0.4) !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
  color: var(--subtext) !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
  font-size: 1.8rem !important;
  font-weight: 800 !important;
  color: var(--text) !important;
  letter-spacing: -0.03em !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
  font-size: 0.8rem !important;
}

/* ══════════════════════════════════════════════════════════════════════
   SECTION CONTAINERS
══════════════════════════════════════════════════════════════════════ */
.section-container {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.5rem 1.75rem;
  margin-bottom: 1.5rem;
  box-shadow: var(--shadow-sm);
  position: relative;
  overflow: hidden;
}
.section-title {
  font-size: 1.05rem;
  font-weight: 700;
  color: var(--text);
  margin-bottom: 1rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
}
.section-title-accent {
  display: inline-block;
  width: 4px;
  height: 1.2em;
  background: linear-gradient(180deg, var(--primary), var(--info));
  border-radius: 2px;
  margin-right: 0.25rem;
}

/* ══════════════════════════════════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════════════════════════════ */
div[data-testid="stButton"] > button {
  font-family: var(--font-main) !important;
  font-weight: 600 !important;
  border-radius: var(--radius-md) !important;
  border: none !important;
  transition: var(--transition) !important;
  letter-spacing: 0.01em !important;
  padding: 0.6rem 1.5rem !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
  background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
  color: #fff !important;
  box-shadow: 0 4px 15px rgba(124,58,237,0.4) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
  background: linear-gradient(135deg, var(--primary-dark), var(--primary)) !important;
  box-shadow: 0 6px 25px rgba(124,58,237,0.55) !important;
  transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stButton"] > button:not([kind]) {
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}
div[data-testid="stButton"] > button:not([kind="primary"]):hover {
  background: rgba(124,58,237,0.12) !important;
  border-color: var(--primary) !important;
  color: var(--primary-light) !important;
  transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button:active {
  transform: translateY(0px) !important;
}

/* Download buttons */
div[data-testid="stDownloadButton"] > button {
  font-family: var(--font-main) !important;
  font-weight: 600 !important;
  border-radius: var(--radius-md) !important;
  background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(37,99,235,0.12)) !important;
  border: 1px solid var(--border) !important;
  color: var(--primary-light) !important;
  transition: var(--transition) !important;
  padding: 0.6rem 1.5rem !important;
}
div[data-testid="stDownloadButton"] > button:hover {
  background: linear-gradient(135deg, rgba(124,58,237,0.3), rgba(37,99,235,0.25)) !important;
  border-color: var(--primary) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 15px rgba(124,58,237,0.3) !important;
}

/* ══════════════════════════════════════════════════════════════════════
   TABS
══════════════════════════════════════════════════════════════════════ */
[data-testid="stTabs"] [data-testid="stTab"] {
  font-family: var(--font-main) !important;
  font-weight: 500 !important;
  font-size: 0.9rem !important;
  color: var(--subtext) !important;
  border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
  transition: var(--transition) !important;
  padding: 0.6rem 1rem !important;
}
[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] {
  color: var(--primary-light) !important;
  font-weight: 700 !important;
}
[data-testid="stTabs"] [data-testid="stTabPanel"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 0 var(--radius-lg) var(--radius-lg) var(--radius-lg);
  padding: 1.5rem !important;
}

/* ══════════════════════════════════════════════════════════════════════
   EXPANDERS
══════════════════════════════════════════════════════════════════════ */
details[data-testid="stExpander"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
  margin-bottom: 1rem !important;
  box-shadow: var(--shadow-sm) !important;
  transition: var(--transition) !important;
}
details[data-testid="stExpander"]:hover {
  border-color: rgba(124,58,237,0.35) !important;
}
details[data-testid="stExpander"] summary {
  color: var(--text) !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  padding: 1rem 1.25rem !important;
  transition: var(--transition) !important;
}
details[data-testid="stExpander"] summary:hover {
  color: var(--primary-light) !important;
}
details[data-testid="stExpander"] > div {
  padding: 0 1.25rem 1.25rem !important;
}

/* ══════════════════════════════════════════════════════════════════════
   FORMS & INPUTS
══════════════════════════════════════════════════════════════════════ */
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input,
[data-testid="stSelectbox"] select,
[data-testid="stTextArea"] textarea {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text) !important;
  font-family: var(--font-main) !important;
  transition: var(--transition) !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(124,58,237,0.15) !important;
  outline: none !important;
}
[data-testid="stSelectbox"] > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text) !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] {
  color: var(--primary-light) !important;
}
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] {
  color: var(--primary-light) !important;
}
div[data-testid="stSlider"] > div > div > div {
  background: linear-gradient(90deg, var(--primary), var(--primary-light)) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius-lg) !important;
  transition: var(--transition) !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--primary) !important;
  background: rgba(124,58,237,0.06) !important;
}
[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] {
  color: var(--subtext) !important;
}

/* ══════════════════════════════════════════════════════════════════════
   DATAFRAMES / TABLES
══════════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
  border: 1px solid var(--border) !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="stDataFrame"] table {
  font-family: var(--font-main) !important;
  font-size: 0.875rem !important;
}
[data-testid="stDataFrame"] thead th {
  background: var(--surface) !important;
  color: var(--subtext) !important;
  font-weight: 700 !important;
  font-size: 0.75rem !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0.75rem 1rem !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) td {
  background: rgba(124,58,237,0.04) !important;
}
[data-testid="stDataFrame"] tbody tr:hover td {
  background: rgba(124,58,237,0.1) !important;
}
[data-testid="stDataFrame"] tbody td {
  color: var(--text) !important;
  border-bottom: 1px solid rgba(42,47,82,0.5) !important;
  padding: 0.65rem 1rem !important;
}

/* ══════════════════════════════════════════════════════════════════════
   PROGRESS BARS
══════════════════════════════════════════════════════════════════════ */
[data-testid="stProgressBar"] > div {
  background: var(--border) !important;
  border-radius: 20px !important;
  overflow: hidden !important;
}
[data-testid="stProgressBar"] > div > div {
  background: linear-gradient(90deg, var(--primary), var(--primary-light), var(--info)) !important;
  border-radius: 20px !important;
  box-shadow: 0 0 10px rgba(124,58,237,0.5) !important;
  transition: width 0.5s ease !important;
  animation: shimmer 2s infinite !important;
  background-size: 200% 100% !important;
}
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}

/* ══════════════════════════════════════════════════════════════════════
   ALERTS / MESSAGES
══════════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"][data-baseweb="notification"] {
  border-radius: var(--radius-md) !important;
  border-width: 1px !important;
  font-family: var(--font-main) !important;
}
[data-testid="stAlert"][kind="success"] {
  background: rgba(16,185,129,0.12) !important;
  border-color: rgba(16,185,129,0.4) !important;
  color: #6EE7B7 !important;
}
[data-testid="stAlert"][kind="warning"] {
  background: rgba(245,158,11,0.12) !important;
  border-color: rgba(245,158,11,0.4) !important;
  color: #FCD34D !important;
}
[data-testid="stAlert"][kind="error"] {
  background: rgba(239,68,68,0.12) !important;
  border-color: rgba(239,68,68,0.4) !important;
  color: #FCA5A5 !important;
}
[data-testid="stAlert"][kind="info"] {
  background: rgba(6,182,212,0.12) !important;
  border-color: rgba(6,182,212,0.4) !important;
  color: #67E8F9 !important;
}

/* ══════════════════════════════════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
══════════════════════════════════════════════════════════════════════ */

/* Badges */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  border-radius: 20px;
  padding: 3px 12px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.badge-purple { background: rgba(124,58,237,0.18); border: 1px solid rgba(124,58,237,0.45); color: #C4B5FD; }
.badge-blue   { background: rgba(37,99,235,0.18);  border: 1px solid rgba(37,99,235,0.45);  color: #93C5FD; }
.badge-green  { background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.45); color: #6EE7B7; }
.badge-orange { background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.45); color: #FCD34D; }
.badge-pink   { background: rgba(236,72,153,0.15); border: 1px solid rgba(236,72,153,0.45); color: #F9A8D4; }
.badge-red    { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.45);  color: #FCA5A5; }
.badge-cyan   { background: rgba(6,182,212,0.15);  border: 1px solid rgba(6,182,212,0.45);  color: #67E8F9; }

/* Divider */
.premium-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--border), transparent);
  margin: 1.5rem 0;
  border: none;
}

/* Observation card */
.obs-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--primary);
  border-radius: 0 var(--radius-md) var(--radius-md) 0;
  padding: 0.85rem 1.25rem;
  margin-bottom: 0.75rem;
  font-size: 0.9rem;
  color: var(--text);
  line-height: 1.6;
  transition: var(--transition);
}
.obs-card:hover {
  border-left-color: var(--primary-light);
  background: rgba(124,58,237,0.06);
}

/* Model rank badge */
.rank-1 { background: linear-gradient(135deg,#FFD700,#FFA500); color:#1A1A1A; border-radius:6px; padding:2px 10px; font-weight:800; font-size:0.78rem; }
.rank-n { background: var(--surface); border:1px solid var(--border); color: var(--subtext); border-radius:6px; padding:2px 10px; font-weight:600; font-size:0.78rem; }

/* Result card */
.result-card {
  background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(37,99,235,0.10));
  border: 1px solid rgba(124,58,237,0.35);
  border-radius: var(--radius-lg);
  padding: 1.5rem 2rem;
  margin-top: 1rem;
  box-shadow: var(--glow-primary);
}
.result-card-title {
  color: var(--primary-light);
  font-size: 1rem;
  font-weight: 700;
  margin-bottom: 0.75rem;
  display: flex;
  align-items: center;
  gap: 0.5rem;
}
.result-card-body {
  color: var(--text);
  font-size: 0.95rem;
  line-height: 1.7;
}

/* Gauge bar */
.gauge-container {
  background: var(--surface);
  border-radius: 20px;
  height: 12px;
  overflow: hidden;
  margin-top: 0.5rem;
  border: 1px solid var(--border);
}
.gauge-fill {
  height: 100%;
  border-radius: 20px;
  transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 0 8px currentColor;
}
.gauge-green  { background: linear-gradient(90deg, #10B981, #34D399); box-shadow: 0 0 8px rgba(16,185,129,0.6); }
.gauge-orange { background: linear-gradient(90deg, #F59E0B, #FCD34D); box-shadow: 0 0 8px rgba(245,158,11,0.6); }
.gauge-red    { background: linear-gradient(90deg, #EF4444, #FCA5A5); box-shadow: 0 0 8px rgba(239,68,68,0.6); }

/* Download card */
.download-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.25rem 1.5rem;
  text-align: center;
  transition: var(--transition);
  box-shadow: var(--shadow-sm);
}
.download-card:hover {
  border-color: rgba(124,58,237,0.4);
  box-shadow: var(--shadow-md), var(--glow-primary);
  transform: translateY(-2px);
}
.download-card-icon {
  font-size: 2.25rem;
  display: block;
  margin-bottom: 0.5rem;
}
.download-card-label {
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.25rem;
}
.download-card-sub {
  font-size: 0.75rem;
  color: var(--subtext);
  margin-bottom: 0.75rem;
}

/* Spinner / loading override */
[data-testid="stSpinner"] svg {
  color: var(--primary-light) !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label {
  color: var(--text) !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
}
[data-testid="stCheckbox"] span[data-testid="stCheckboxBase"] {
  border-color: var(--border) !important;
  border-radius: 4px !important;
}

/* Multiselect */
[data-testid="stMultiSelect"] > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
  background: rgba(124,58,237,0.25) !important;
  border: 1px solid rgba(124,58,237,0.45) !important;
  border-radius: 6px !important;
  color: var(--primary-light) !important;
}

/* Radio */
[data-testid="stRadio"] label {
  color: var(--text) !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
}

/* Horizontal rule */
hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--border), transparent) !important;
  margin: 1.5rem 0 !important;
}

/* Headings */
h1, h2, h3, h4, h5 {
  font-family: var(--font-main) !important;
  color: var(--text) !important;
  letter-spacing: -0.02em !important;
}
h2 { font-size: 1.6rem !important; font-weight: 800 !important; }
h3 { font-size: 1.2rem !important; font-weight: 700 !important; }
h4 { font-size: 1rem !important;   font-weight: 600 !important; }

/* Caption / small text */
[data-testid="stCaptionContainer"] p {
  color: var(--subtext) !important;
  font-size: 0.82rem !important;
}

/* Code blocks */
code {
  font-family: var(--font-mono) !important;
  background: rgba(124,58,237,0.12) !important;
  color: var(--primary-light) !important;
  border-radius: 4px !important;
  padding: 2px 6px !important;
  font-size: 0.85em !important;
}

/* Spinner text */
[data-testid="stText"] {
  color: var(--subtext) !important;
  font-size: 0.9rem !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* Animation utilities */
@keyframes fade-in {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes slide-up {
  from { opacity: 0; transform: translateY(24px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fade-in  { animation: fade-in  0.5s ease forwards; }
.slide-up { animation: slide-up 0.6s ease forwards; }

/* Responsive tweaks */
@media (max-width: 768px) {
  .block-container { padding: 1rem !important; }
  .page-header { padding: 1.25rem 1.5rem; }
  .page-header-title { font-size: 1.5rem !important; }
}
</style>
"""

# Light mode CSS variable overrides injected via a tiny JS snippet
_LIGHT_MODE_JS = """
<script>
(function() {
  const isDark = localStorage.getItem('insightml_theme') !== 'light';
  if (!isDark) {
    document.documentElement.setAttribute('data-theme', 'light');
  }
})();
</script>
"""


def inject_css() -> None:
    """Inject master CSS + light/dark mode into the current page.
    Call this at the very top of every page's render() function.
    """
    st.markdown(_MASTER_CSS, unsafe_allow_html=True)

    # Sync CSS variables with current theme choice
    theme = st.session_state.get("theme", "dark")
    if theme == "light":
        st.markdown(
            """<style>
            :root { color-scheme: light; }
            [data-testid='stAppViewContainer'], html, body { background: var(--bg) !important; color: var(--text) !important; }
            [data-testid='stSidebar'] { background: var(--sidebar-bg) !important; }
            </style>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<script>
            document.documentElement.setAttribute('data-theme', 'light');
            </script>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<style>
            :root { color-scheme: dark; }
            [data-testid='stAppViewContainer'], html, body { background: var(--bg) !important; color: var(--text) !important; }
            [data-testid='stSidebar'] { background: var(--sidebar-bg) !important; }
            </style>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """<script>
            document.documentElement.removeAttribute('data-theme');
            </script>""",
            unsafe_allow_html=True,
        )


def theme_toggle(sidebar_obj=None) -> None:
    """Render a dark/light mode toggle in the sidebar (or current scope)."""
    scope = sidebar_obj if sidebar_obj else st.sidebar
    current = st.session_state.get("theme", "dark")
    label = "☀️ Light Mode" if current == "dark" else "🌙 Dark Mode"
    if scope.button(label, key="__theme_toggle__", use_container_width=True):
        st.session_state["theme"] = "light" if current == "dark" else "dark"
        st.rerun()


def page_header(
    title: str,
    icon: str = "🧠",
    subtitle: str = "",
    badge: str = "",
) -> None:
    """Render a premium gradient page header."""
    badge_html = f'<span class="page-header-badge">{badge}</span>' if badge else ""
    st.markdown(
        f"""<div class="page-header fade-in">
  <span class="page-header-icon">{icon}</span>
  <div class="page-header-title">{title}</div>
  {"<div class='page-header-subtitle'>" + subtitle + "</div>" if subtitle else ""}
  {badge_html}
</div>""",
        unsafe_allow_html=True,
    )


def kpi_row(metrics: list[dict]) -> None:
    """
    Render a row of KPI cards using st.columns.

    Each dict should have keys:
      label, value, icon (optional), color (optional: purple/blue/green/orange/pink/red), delta (optional)
    """
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        icon   = m.get("icon", "📊")
        color  = m.get("color", "purple")
        delta  = m.get("delta", "")
        delta_cls = ""
        if delta:
            delta_cls = "up" if str(delta).startswith("+") else "down" if str(delta).startswith("-") else ""
        delta_html = f'<div class="kpi-delta {delta_cls}">{delta}</div>' if delta else ""
        col.markdown(
            f"""<div class="kpi-card {color}">
  <span class="kpi-icon">{icon}</span>
  <div class="kpi-label">{m['label']}</div>
  <div class="kpi-value">{m['value']}</div>
  {delta_html}
</div>""",
            unsafe_allow_html=True,
        )


def section_header(title: str, icon: str = "") -> None:
    """Render a styled section title with an accent bar."""
    prefix = f"{icon} " if icon else ""
    st.markdown(
        f"""<div class="section-title">
  <span class="section-title-accent"></span>{prefix}{title}
</div>""",
        unsafe_allow_html=True,
    )


def obs_card(text: str) -> None:
    """Render a single styled observation card."""
    st.markdown(f'<div class="obs-card">{text}</div>', unsafe_allow_html=True)


def premium_divider() -> None:
    """Render a styled divider line."""
    st.markdown('<hr class="premium-divider">', unsafe_allow_html=True)


def badge(text: str, kind: str = "purple") -> str:
    """Return a badge HTML string. kind: purple|blue|green|orange|pink|red|cyan"""
    return f'<span class="badge badge-{kind}">{text}</span>'


def result_card(title: str, body: str) -> None:
    """Render a premium result card."""
    st.markdown(
        f"""<div class="result-card">
  <div class="result-card-title">{title}</div>
  <div class="result-card-body">{body}</div>
</div>""",
        unsafe_allow_html=True,
    )


def gauge_bar(value: float, label: str = "") -> None:
    """
    Render an animated gradient gauge bar.
    value: float 0.0–1.0
    """
    pct = min(max(value, 0.0), 1.0) * 100
    color_cls = "gauge-green" if value < 0.45 else "gauge-orange" if value < 0.75 else "gauge-red"
    label_html = f"<div style='font-size:0.82rem;color:var(--subtext);margin-bottom:4px;'>{label}: <strong style='color:var(--text);'>{value:.1%}</strong></div>" if label else ""
    st.markdown(
        f"""{label_html}<div class="gauge-container">
  <div class="gauge-fill {color_cls}" style="width:{pct:.1f}%"></div>
</div>""",
        unsafe_allow_html=True,
    )


def sidebar_logo(*args, version: str = "1.0.0", **kwargs) -> None:
    """Render the InsightML Studio logo block in the sidebar.

    Accepts either a positional version argument or keyword `version` for
    compatibility with legacy imports and runtime reloads.
    """
    if args:
        version = str(args[0])

    st.sidebar.markdown(
        f"""<div class="sidebar-logo">
  <div class="sidebar-logo-brand">
    <span class="sidebar-logo-icon">🧠</span>
    <div>
      <div class="sidebar-logo-title">InsightML Studio</div>
      <div class="sidebar-logo-sub">AutoML Platform</div>
    </div>
  </div>
  <div class="sidebar-logo-meta">
    <span class="sidebar-status-dot"></span>
    <span class="sidebar-logo-version">v{version}</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def sidebar_footer() -> None:
    """Render branded footer at the bottom of the sidebar."""
    st.sidebar.markdown(
        """<div class="sidebar-footer">
  <div class="sidebar-version-badge">v1.0.0</div>
  <div class="sidebar-footer-text">
    Intelligent Automated ML<br>& Explainable Analytics
  </div>
</div>""",
        unsafe_allow_html=True,
    )
