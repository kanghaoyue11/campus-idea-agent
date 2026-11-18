from __future__ import annotations

import json
import os

import streamlit as st

from core import APP_NAME, MAX_RETRIES, LLMClient, run_orchestration, to_markdown


st.set_page_config(
    page_title=f"{APP_NAME} · 让创意先被验证",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_CSS = """
<style>
:root {
    --ink: #17171a;
    --ink-secondary: #4f535a;
    --ink-tertiary: #73777f;
    --accent: #0071e3;
    --accent-hover: #147ceb;
    --surface: rgba(255, 255, 255, 0.62);
    --surface-strong: rgba(255, 255, 255, 0.78);
    --canvas: #f4f6fa;
    --field: rgba(255, 255, 255, 0.58);
    --line: rgba(255, 255, 255, 0.66);
    --success: #1f7a38;
    --warning: #8a5b00;
    --radius-lg: 24px;
    --radius-md: 16px;
    --radius-sm: 12px;
    --glass-shadow: 0 18px 55px rgba(31, 35, 45, 0.075), 0 2px 8px rgba(31, 35, 45, 0.035);
    --glass-highlight: inset 0 1px 0 rgba(255, 255, 255, 0.86),
        inset 0 -1px 0 rgba(255, 255, 255, 0.24);
}

html {
    scroll-behavior: smooth;
}

html, body, [class*="css"], .stApp {
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text",
        "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "Noto Sans CJK SC", sans-serif;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
    text-rendering: optimizeLegibility;
}

body {
    background: var(--canvas);
    overflow-x: hidden;
}

.stApp {
    position: relative;
    min-height: 100vh;
    background: var(--canvas);
    isolation: isolate;
}

.stApp::before {
    content: "";
    position: fixed;
    inset: -18% -12%;
    z-index: 0;
    pointer-events: none;
    background:
        radial-gradient(circle at 14% 18%, rgba(84, 170, 255, 0.28), transparent 30%),
        radial-gradient(circle at 84% 13%, rgba(196, 145, 255, 0.24), transparent 31%),
        radial-gradient(circle at 48% 52%, rgba(132, 220, 207, 0.16), transparent 34%),
        radial-gradient(circle at 76% 84%, rgba(255, 188, 164, 0.20), transparent 32%),
        radial-gradient(circle at 22% 92%, rgba(144, 182, 255, 0.16), transparent 27%),
        linear-gradient(180deg, #f2f7fd 0%, #eef3f9 42%, #f7f4f9 100%);
    filter: blur(32px) saturate(1.16);
    transform: translate3d(-1%, -1%, 0) scale(1);
    animation: ambient-drift 32s ease-in-out infinite alternate;
    will-change: transform;
}

[data-testid="stAppViewContainer"] {
    position: relative;
    z-index: 1;
    background: transparent;
}

[data-testid="stHeader"] {
    background: transparent;
    backdrop-filter: none;
    -webkit-backdrop-filter: none;
}

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
    visibility: hidden;
}

.block-container {
    position: relative;
    z-index: 1;
    max-width: 1180px;
    padding: 1.35rem 2rem 3rem;
}

::selection {
    color: var(--ink);
    background: rgba(119, 180, 255, 0.24);
}

.site-header {
    position: fixed;
    top: 0.85rem;
    left: 50%;
    z-index: 40;
    width: min(calc(100vw - 2rem), 1116px);
    min-height: 52px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin: 0;
    padding: 0.5rem 0.65rem;
    border: 1px solid rgba(255, 255, 255, 0.72);
    border-radius: 18px;
    background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.74), rgba(245, 249, 255, 0.48));
    backdrop-filter: blur(26px) saturate(1.55);
    -webkit-backdrop-filter: blur(26px) saturate(1.55);
    box-shadow: var(--glass-highlight), 0 14px 38px rgba(31, 44, 67, 0.075);
    transform: translateX(-50%);
    overflow: hidden;
}

.site-header::before {
    content: "";
    position: absolute;
    top: 0;
    left: 20%;
    right: 20%;
    height: 1px;
    background: linear-gradient(90deg, transparent,
        rgba(89, 157, 255, 0.46), rgba(176, 139, 255, 0.34),
        rgba(255, 178, 188, 0.28), transparent);
    opacity: 0.72;
}

.brand {
    display: inline-flex;
    align-items: center;
    gap: 0.7rem;
    color: var(--ink);
    font-size: 0.95rem;
    font-weight: 650;
    letter-spacing: -0.01em;
    white-space: nowrap;
}

.brand-mark {
    position: relative;
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border: 1px solid rgba(255, 255, 255, 0.38);
    border-radius: 11px;
    color: #ffffff;
    background: linear-gradient(145deg, #202025 0%, #34343a 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.20),
        0 5px 14px rgba(24, 27, 34, 0.13);
    overflow: hidden;
}

.brand-mark::after {
    content: "";
    position: absolute;
    top: -40%;
    left: -30%;
    width: 160%;
    height: 70%;
    background: linear-gradient(100deg, transparent,
        rgba(128, 177, 255, 0.28), rgba(220, 171, 255, 0.20), transparent);
    transform: rotate(-12deg);
}

.local-badge, .result-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.43rem 0.76rem;
    border: 1px solid rgba(255, 255, 255, 0.68);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.52);
    color: var(--ink-secondary);
    font-size: 0.76rem;
    font-weight: 550;
    white-space: nowrap;
    backdrop-filter: blur(14px) saturate(1.2);
    -webkit-backdrop-filter: blur(14px) saturate(1.2);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72);
}

.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #2f9e64;
    box-shadow: 0 0 0 3px rgba(47, 158, 100, 0.11);
}

.hero {
    position: relative;
    max-width: 860px;
    margin: 4.9rem auto 2.15rem;
    text-align: center;
    animation: rise-in 560ms cubic-bezier(.2, .72, .2, 1) both;
}

.hero::before {
    content: "";
    position: absolute;
    top: -3rem;
    left: 50%;
    z-index: -1;
    width: min(44rem, 86vw);
    height: 16rem;
    background:
        radial-gradient(ellipse at 24% 45%, rgba(92, 182, 255, 0.17), transparent 42%),
        radial-gradient(ellipse at 50% 34%, rgba(176, 139, 255, 0.14), transparent 46%),
        radial-gradient(ellipse at 74% 48%, rgba(255, 184, 189, 0.12), transparent 42%);
    filter: blur(20px);
    opacity: 0.92;
    transform: translateX(-50%);
    pointer-events: none;
}

.hero::after {
    content: "";
    display: block;
    width: min(190px, 46vw);
    height: 2px;
    margin: 1.55rem auto 0;
    border-radius: 999px;
    background: linear-gradient(90deg, transparent,
        rgba(77, 146, 255, 0.54), rgba(164, 137, 255, 0.46),
        rgba(255, 180, 189, 0.38), transparent);
    opacity: 0.72;
}

.eyebrow {
    margin-bottom: 1.05rem;
    color: var(--accent);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.hero h1 {
    margin: 0;
    color: var(--ink);
    font-size: clamp(2.55rem, 5.1vw, 4.75rem);
    font-weight: 720;
    line-height: 1.09;
    letter-spacing: -0.052em;
}

.hero h1 span {
    color: transparent;
    background: linear-gradient(103deg, #0874e5 0%, #2478ec 34%, #5968e8 68%, #8e5ed5 100%);
    -webkit-background-clip: text;
    background-clip: text;
}

.hero .hero-description {
    max-width: 610px;
    margin: 1.2rem auto 0;
    color: var(--ink-secondary);
    font-size: clamp(1rem, 1.6vw, 1.16rem);
    line-height: 1.85;
    letter-spacing: 0.01em;
}

.flow-strip {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.1rem;
    width: max-content;
    max-width: 100%;
    margin: 0 auto 2.55rem;
    padding: 0.58rem 0.95rem;
    border: 1px solid rgba(255, 255, 255, 0.76);
    border-radius: 999px;
    background: linear-gradient(105deg,
        rgba(255, 255, 255, 0.66), rgba(240, 247, 255, 0.42) 54%,
        rgba(251, 244, 255, 0.46));
    color: var(--ink-secondary);
    backdrop-filter: blur(20px) saturate(1.45);
    -webkit-backdrop-filter: blur(20px) saturate(1.45);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.86),
        inset 0 -1px 0 rgba(255, 255, 255, 0.24),
        0 12px 32px rgba(31, 44, 67, 0.055);
    overflow-x: auto;
    scrollbar-width: none;
    animation: rise-in 560ms 90ms cubic-bezier(.2, .72, .2, 1) both;
}

.flow-strip::-webkit-scrollbar {
    display: none;
}

.flow-item {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.83rem;
    white-space: nowrap;
}

.flow-number {
    color: var(--accent);
    font-size: 0.69rem;
    font-weight: 700;
    letter-spacing: 0.06em;
}

.flow-arrow {
    color: #a8acb4;
    font-size: 0.9rem;
}

[data-testid="stHorizontalBlock"] {
    gap: 1.25rem;
}

[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .glass-card-marker) {
    position: relative;
    border: 1px solid rgba(255, 255, 255, 0.80) !important;
    border-radius: var(--radius-lg) !important;
    background:
        linear-gradient(145deg, rgba(255, 255, 255, 0.66),
            rgba(246, 250, 255, 0.38) 56%, rgba(252, 246, 255, 0.42)) !important;
    backdrop-filter: blur(28px) saturate(1.55);
    -webkit-backdrop-filter: blur(28px) saturate(1.55);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.94),
        inset 0 -1px 0 rgba(255, 255, 255, 0.30),
        0 24px 64px rgba(31, 44, 67, 0.10),
        0 4px 16px rgba(31, 44, 67, 0.045);
    overflow: hidden;
    animation: glass-in 620ms 120ms cubic-bezier(.2, .72, .2, 1) both;
    transition: border-color 220ms ease, box-shadow 220ms ease;
}

[data-testid="stVerticalBlockBorderWrapper"]::before,
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .glass-card-marker)::before {
    content: "";
    position: absolute;
    inset: 0;
    z-index: 0;
    background: linear-gradient(125deg,
        rgba(255, 255, 255, 0.78) 0%, rgba(255, 255, 255, 0.18) 28%,
        transparent 56%, rgba(255, 255, 255, 0.42) 100%);
    opacity: 0.72;
    pointer-events: none;
}

[data-testid="stVerticalBlockBorderWrapper"]::after,
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .glass-card-marker)::after {
    content: "";
    position: absolute;
    top: 0;
    left: 24px;
    right: 24px;
    z-index: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent,
        rgba(89, 157, 255, 0.48), rgba(177, 139, 255, 0.38),
        rgba(255, 171, 184, 0.30), transparent);
    opacity: 0.82;
    pointer-events: none;
}

[data-testid="stVerticalBlockBorderWrapper"]:hover,
[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .glass-card-marker):hover {
    border-color: rgba(255, 255, 255, 0.86) !important;
    box-shadow: var(--glass-highlight), 0 22px 62px rgba(31, 35, 45, 0.085),
        0 3px 10px rgba(31, 35, 45, 0.04);
}

[data-testid="stVerticalBlockBorderWrapper"] > div {
    position: relative;
    z-index: 1;
    padding: 1.55rem 1.65rem;
}

[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .glass-card-marker) {
    padding: 1.55rem 1.65rem;
}

[data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .glass-card-marker)
    > [data-testid="stElementContainer"] {
    position: relative;
    z-index: 1;
}

.glass-card-marker {
    display: none;
}

[data-testid="stElementContainer"]:has(> [data-testid="stMarkdown"] .glass-card-marker) {
    display: none;
}

.section-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    margin: 0 0 1.4rem;
}

.section-icon {
    flex: 0 0 auto;
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    border: 1px solid rgba(255, 255, 255, 0.62);
    border-radius: 11px;
    color: var(--accent);
    background: linear-gradient(135deg,
        rgba(232, 243, 255, 0.88), rgba(247, 241, 255, 0.72));
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.82),
        0 4px 10px rgba(31, 35, 45, 0.045);
}

.section-heading .section-title {
    margin: 0;
    color: var(--ink);
    font-size: 1.05rem;
    font-weight: 650;
    line-height: 1.35;
    letter-spacing: -0.015em;
}

.section-heading .section-subtitle {
    margin: 0.3rem 0 0;
    color: var(--ink-tertiary);
    font-size: 0.78rem;
    line-height: 1.6;
}

[data-testid="stWidgetLabel"] p {
    color: #36383d;
    font-size: 0.84rem;
    font-weight: 620;
    line-height: 1.5;
}

[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    border: 1px solid rgba(255, 255, 255, 0.68);
    border-radius: 13px;
    background-color: rgba(255, 255, 255, 0.56);
    color: var(--ink);
    font-size: 0.92rem;
    line-height: 1.7;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.72),
        inset 0 1px 3px rgba(31, 35, 45, 0.025);
    backdrop-filter: blur(14px) saturate(1.2);
    -webkit-backdrop-filter: blur(14px) saturate(1.2);
    transition: border-color 190ms ease, box-shadow 190ms ease,
        background-color 190ms ease;
}

[data-testid="stTextArea"] textarea {
    padding: 0.85rem 0.95rem;
}

[data-testid="stTextInput"] input {
    min-height: 46px;
    padding: 0.65rem 0.9rem;
}

[data-testid="stTextArea"] textarea::placeholder,
[data-testid="stTextInput"] input::placeholder {
    color: #7d828a;
    opacity: 1;
}

[data-testid="stTextArea"] textarea:hover,
[data-testid="stTextInput"] input:hover {
    border-color: rgba(255, 255, 255, 0.90);
    background-color: rgba(255, 255, 255, 0.68);
}

[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: rgba(0, 113, 227, 0.72);
    background-color: rgba(255, 255, 255, 0.86);
    box-shadow: 0 0 0 4px rgba(0, 113, 227, 0.11),
        inset 0 1px 0 rgba(255, 255, 255, 0.86);
    outline: none;
}

[data-testid="stTextArea"] textarea:focus-visible,
[data-testid="stTextInput"] input:focus-visible,
.stButton button:focus-visible,
.stDownloadButton button:focus-visible,
[data-testid="stRadio"] input:focus-visible + div {
    outline: 3px solid rgba(0, 113, 227, 0.30);
    outline-offset: 2px;
}

[data-testid="stRadio"] > div[role="radiogroup"] {
    width: 100%;
    gap: 0.35rem;
    padding: 0.3rem;
    border: 1px solid rgba(255, 255, 255, 0.64);
    border-radius: 13px;
    background: rgba(255, 255, 255, 0.42);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.62);
    backdrop-filter: blur(14px) saturate(1.25);
    -webkit-backdrop-filter: blur(14px) saturate(1.25);
}

[data-testid="stRadio"] > div[role="radiogroup"] > label {
    flex: 1 1 0;
    min-width: 0;
    min-height: 42px;
    justify-content: center;
    margin: 0;
    padding: 0.5rem 0.7rem;
    border-radius: 10px;
    cursor: pointer;
    transition: background-color 190ms ease, box-shadow 190ms ease, color 190ms ease;
}

[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.42);
}

[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
    background: rgba(255, 255, 255, 0.88);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.9),
        0 2px 8px rgba(31, 35, 45, 0.075);
}

[data-testid="stRadio"] > div[role="radiogroup"] > label > div:first-child {
    display: none;
}

[data-testid="stRadio"] > div[role="radiogroup"] > label p {
    width: 100%;
    color: var(--ink-secondary);
    font-size: 0.78rem;
    font-weight: 570;
    text-align: center;
    white-space: normal;
    line-height: 1.45;
}

[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) p {
    color: var(--ink);
}

.settings-divider {
    height: 1px;
    margin: 1.25rem 0;
    background: linear-gradient(90deg, transparent,
        rgba(255, 255, 255, 0.78), rgba(31, 35, 45, 0.055), transparent);
}

.settings-label {
    margin-bottom: 0.8rem;
    color: var(--ink-secondary);
    font-size: 0.74rem;
    font-weight: 650;
    letter-spacing: 0.035em;
}

.key-status {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    margin-top: 0.95rem;
    padding: 0.74rem 0.82rem;
    border: 1px solid rgba(255, 255, 255, 0.62);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.44);
    color: var(--ink-secondary);
    font-size: 0.74rem;
    line-height: 1.55;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.66);
}

.key-status.ready {
    color: #236f39;
    background: rgba(239, 250, 242, 0.66);
}

.key-status svg {
    flex: 0 0 auto;
    margin-top: 2px;
}

.privacy-note {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    margin: 0.1rem 0 1rem;
    color: var(--ink-tertiary);
    font-size: 0.73rem;
    line-height: 1.5;
}

.stButton button,
.stDownloadButton button {
    position: relative;
    min-height: 44px;
    border-radius: 13px;
    font-size: 0.86rem;
    font-weight: 620;
    letter-spacing: 0.01em;
    cursor: pointer;
    overflow: hidden;
    transition: background-color 190ms ease, border-color 190ms ease,
        box-shadow 190ms ease, color 190ms ease, transform 140ms ease;
}

.stButton button[kind="primary"] {
    min-height: 52px;
    border: 1px solid rgba(255, 255, 255, 0.42);
    background: linear-gradient(180deg, #1680f0 0%, #006ee6 100%);
    color: #ffffff;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.34),
        inset 0 -1px 0 rgba(0, 42, 92, 0.16),
        0 11px 28px rgba(0, 113, 227, 0.24),
        0 2px 6px rgba(0, 113, 227, 0.16);
}

.stButton button[kind="primary"]::before {
    content: "";
    position: absolute;
    inset: 0;
    background: linear-gradient(105deg, transparent 18%,
        rgba(255, 255, 255, 0.22) 38%, rgba(184, 214, 255, 0.18) 48%,
        rgba(255, 188, 205, 0.10) 55%, transparent 70%);
    opacity: 0;
    transform: translateX(-105%);
    transition: transform 620ms cubic-bezier(.2, .72, .2, 1), opacity 220ms ease;
    pointer-events: none;
}

.stButton button[kind="primary"]::after {
    content: "";
    position: absolute;
    top: 0;
    left: 10%;
    right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent,
        rgba(255, 255, 255, 0.68), rgba(207, 226, 255, 0.54),
        rgba(255, 190, 211, 0.36), transparent);
    opacity: 0.78;
    pointer-events: none;
}

.stButton button[kind="primary"]:hover {
    border-color: rgba(255, 255, 255, 0.58);
    background: linear-gradient(180deg, #2289f5 0%, #0b75e8 100%);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.38),
        0 14px 34px rgba(0, 113, 227, 0.28),
        0 3px 8px rgba(0, 113, 227, 0.18);
}

.stButton button[kind="primary"]:hover::before {
    opacity: 1;
    transform: translateX(105%);
}

.stButton button[kind="primary"]:active {
    background: linear-gradient(180deg, #0d74dc 0%, #0063ce 100%);
    transform: translateY(1px);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.28),
        0 7px 20px rgba(0, 113, 227, 0.20);
}

.stDownloadButton button,
.stButton button[kind="secondary"] {
    border: 1px solid rgba(255, 255, 255, 0.70);
    background: rgba(255, 255, 255, 0.56);
    color: #3a3d43;
    backdrop-filter: blur(14px) saturate(1.25);
    -webkit-backdrop-filter: blur(14px) saturate(1.25);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.78),
        0 5px 16px rgba(31, 35, 45, 0.045);
}

.stDownloadButton button:hover,
.stButton button[kind="secondary"]:hover {
    border-color: rgba(255, 255, 255, 0.92);
    background: rgba(255, 255, 255, 0.74);
    color: var(--ink);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.88),
        0 8px 22px rgba(31, 35, 45, 0.065);
}

.stDownloadButton button:active,
.stButton button[kind="secondary"]:active {
    transform: translateY(1px);
}

.button-caption {
    margin-top: 0.65rem;
    color: var(--ink-tertiary);
    font-size: 0.7rem;
    text-align: center;
}

.runtime-note {
    margin-top: 1.3rem;
    padding-top: 1.15rem;
    border-top: 1px solid rgba(255, 255, 255, 0.52);
}

.runtime-note-title {
    margin-bottom: 0.75rem;
    color: var(--ink-secondary);
    font-size: 0.73rem;
    font-weight: 650;
}

.runtime-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-top: 0.55rem;
    color: var(--ink-secondary);
    font-size: 0.75rem;
}

.runtime-check {
    width: 18px;
    height: 18px;
    display: grid;
    place-items: center;
    border: 1px solid rgba(255, 255, 255, 0.56);
    border-radius: 50%;
    color: var(--accent);
    background: rgba(235, 245, 255, 0.72);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.76);
}

.results-section {
    margin-top: 2.9rem;
    scroll-margin-top: 6.2rem;
    animation: rise-in 460ms cubic-bezier(.2, .72, .2, 1) both;
}

.results-heading-row {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1.15rem;
}

.results-kicker {
    margin-bottom: 0.4rem;
    color: var(--accent);
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.13em;
}

.results-heading-row .results-title {
    margin: 0;
    color: var(--ink);
    font-size: 1.8rem;
    font-weight: 680;
    letter-spacing: -0.035em;
}

.result-time {
    color: var(--ink-tertiary);
    font-size: 0.73rem;
    white-space: nowrap;
}

.result-banner {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1rem;
    padding: 0.86rem 1rem;
    border: 1px solid rgba(255, 255, 255, 0.68);
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.54);
    backdrop-filter: blur(18px) saturate(1.32);
    -webkit-backdrop-filter: blur(18px) saturate(1.32);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.78),
        0 9px 26px rgba(31, 35, 45, 0.045);
    overflow: hidden;
}

.result-banner::before {
    content: "";
    position: absolute;
    top: 0;
    left: 14%;
    right: 14%;
    height: 1px;
    background: linear-gradient(90deg, transparent,
        rgba(88, 156, 255, 0.35), rgba(181, 142, 255, 0.27),
        rgba(255, 182, 190, 0.22), transparent);
}

.result-banner-text {
    color: var(--ink-secondary);
    font-size: 0.78rem;
    line-height: 1.55;
}

.result-banner strong {
    color: var(--ink);
    font-weight: 600;
}

.result-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}

.result-metric {
    padding: 1rem 1.1rem;
    border: 1px solid rgba(255, 255, 255, 0.68);
    border-radius: 15px;
    background: rgba(255, 255, 255, 0.50);
    backdrop-filter: blur(18px) saturate(1.30);
    -webkit-backdrop-filter: blur(18px) saturate(1.30);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.76),
        0 8px 24px rgba(31, 35, 45, 0.040);
}

.result-metric-label {
    margin-bottom: 0.35rem;
    color: var(--ink-tertiary);
    font-size: 0.69rem;
}

.result-metric-value {
    color: var(--ink);
    font-size: 1rem;
    font-weight: 650;
}

.result-metric-value.success {
    color: var(--success);
}

.result-metric-value.warning {
    color: var(--warning);
}

[data-testid="stTabs"] [data-baseweb="tab-list"],
[data-testid="stTabs"] [role="tablist"] {
    display: flex;
    gap: 0.25rem;
    margin-bottom: 1rem;
    padding: 0.3rem;
    border: 1px solid rgba(255, 255, 255, 0.76);
    border-radius: 14px;
    background: linear-gradient(110deg,
        rgba(230, 242, 255, 0.58), rgba(255, 255, 255, 0.40) 48%,
        rgba(250, 238, 255, 0.46));
    backdrop-filter: blur(18px) saturate(1.28);
    -webkit-backdrop-filter: blur(18px) saturate(1.28);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.78),
        0 8px 22px rgba(31, 44, 67, 0.040);
    overflow-x: auto;
    scrollbar-width: none;
}

[data-testid="stTabs"] [data-baseweb="tab-list"]::-webkit-scrollbar,
[data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar {
    display: none;
}

[data-testid="stTabs"] [data-baseweb="tab"],
[data-testid="stTabs"] [data-testid="stTab"] {
    flex: 1 0 auto;
    min-height: 40px;
    padding: 0 1rem;
    border: none;
    border-radius: 10px;
    background: transparent;
    color: var(--ink-secondary);
    font-size: 0.8rem;
    font-weight: 570;
    cursor: pointer;
    transition: background-color 180ms ease, color 180ms ease, box-shadow 180ms ease;
}

[data-testid="stTabs"] [data-testid="stTab"] {
    position: relative;
    justify-content: center;
}

[data-testid="stTabs"] [data-baseweb="tab"]:hover,
[data-testid="stTabs"] [data-testid="stTab"]:hover {
    color: var(--ink);
    background: rgba(255, 255, 255, 0.48);
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.96),
        rgba(250, 253, 255, 0.84)) !important;
    color: var(--ink) !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.92),
        0 3px 10px rgba(31, 35, 45, 0.075);
}

[data-testid="stTabs"] [data-testid="stTab"] p {
    margin: 0;
    color: inherit;
    font-size: inherit;
    font-weight: inherit;
    line-height: 1.4;
}

[data-testid="stTabs"] [data-testid="stTab"][aria-selected="true"] p {
    color: var(--ink);
}

[data-testid="stTabs"] .react-aria-SelectionIndicator,
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
}

[data-testid="stTabs"] [data-testid="stMarkdownContainer"] h1 {
    margin: 0.2rem 0 1.25rem;
    color: var(--ink);
    font-size: 1.65rem;
    font-weight: 680;
    line-height: 1.35;
    letter-spacing: -0.03em;
}

[data-testid="stTabs"] [data-testid="stMarkdownContainer"] h2 {
    margin: 1.8rem 0 0.75rem;
    padding: 0;
    border: none;
    color: var(--ink);
    font-size: 1.12rem;
    font-weight: 650;
    line-height: 1.45;
    letter-spacing: -0.015em;
}

[data-testid="stTabs"] [data-testid="stMarkdownContainer"] h3 {
    color: var(--ink);
    font-size: 1rem;
    font-weight: 650;
}

[data-testid="stTabs"] [data-testid="stMarkdownContainer"] p,
[data-testid="stTabs"] [data-testid="stMarkdownContainer"] li {
    color: #3c3f45;
    font-size: 0.92rem;
    line-height: 1.9;
}

[data-testid="stTabs"] [data-testid="stMarkdownContainer"] strong {
    color: var(--ink);
}

[data-testid="stExpander"] {
    border: 1px solid rgba(255, 255, 255, 0.66);
    border-radius: 13px !important;
    background: rgba(255, 255, 255, 0.46);
    backdrop-filter: blur(14px) saturate(1.2);
    -webkit-backdrop-filter: blur(14px) saturate(1.2);
    overflow: hidden;
}

[data-testid="stExpander"] summary {
    color: var(--ink-secondary);
    font-size: 0.8rem;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.62);
    border-radius: 13px;
    background: rgba(255, 255, 255, 0.52);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.70);
    overflow: hidden;
}

.review-summary {
    margin-bottom: 1rem;
    color: #3f4147;
    font-size: 0.92rem;
    line-height: 1.8;
}

.knowledge-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0.8rem 0 0.3rem;
}

.knowledge-chip {
    padding: 0.45rem 0.7rem;
    border: 1px solid rgba(255, 255, 255, 0.68);
    border-radius: 999px;
    color: var(--ink-secondary);
    background: rgba(255, 255, 255, 0.48);
    font-size: 0.74rem;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.70);
}

.download-row {
    margin-top: 1rem;
}

[data-testid="stAlert"] {
    border: 1px solid rgba(255, 255, 255, 0.68);
    border-radius: 13px;
    font-size: 0.84rem;
    backdrop-filter: blur(14px) saturate(1.2);
    -webkit-backdrop-filter: blur(14px) saturate(1.2);
}

[data-testid="stSpinner"] {
    padding: 0.75rem 1rem;
    border: 1px solid rgba(255, 255, 255, 0.68);
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.58);
    backdrop-filter: blur(18px) saturate(1.3);
    -webkit-backdrop-filter: blur(18px) saturate(1.3);
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.76),
        0 8px 24px rgba(31, 35, 45, 0.055);
}

[data-testid="stSpinner"] svg {
    animation: spinner-rotate 850ms linear infinite;
}

.site-footer {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 2rem;
    margin-top: 3rem;
    padding-top: 1.3rem;
    border-top: 1px solid rgba(255, 255, 255, 0.58);
    color: var(--ink-tertiary);
    font-size: 0.7rem;
    line-height: 1.75;
}

.footer-brand {
    color: var(--ink-secondary);
    font-weight: 600;
    white-space: nowrap;
}

@keyframes ambient-drift {
    from { transform: translate3d(-1.2%, -1.1%, 0) scale(1); }
    to { transform: translate3d(1.1%, 1.3%, 0) scale(1.035); }
}

@keyframes rise-in {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes glass-in {
    from {
        opacity: 0;
        transform: translateY(12px) scale(0.996);
        filter: blur(4px);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
        filter: blur(0);
    }
}

@keyframes spinner-rotate {
    to { transform: rotate(360deg); }
}

@supports not ((backdrop-filter: blur(1px)) or (-webkit-backdrop-filter: blur(1px))) {
    .site-header,
    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .glass-card-marker),
    .flow-strip,
    .result-banner,
    .result-metric,
    .key-status,
    [data-testid="stTabs"] [role="tablist"] {
        background: rgba(255, 255, 255, 0.93) !important;
    }

    [data-testid="stTextArea"] textarea,
    [data-testid="stTextInput"] input,
    .stDownloadButton button,
    .stButton button[kind="secondary"] {
        background: rgba(255, 255, 255, 0.96) !important;
    }
}

@media (max-width: 760px) {
    .block-container {
        padding: 0.95rem 1rem 2rem;
    }

    .site-header {
        top: 0.55rem;
        min-height: 48px;
        padding: 0.45rem 0.55rem;
        border-radius: 16px;
    }

    .brand {
        gap: 0.55rem;
        font-size: 0.84rem;
    }

    .brand-mark {
        width: 31px;
        height: 31px;
        border-radius: 10px;
    }

    .local-badge {
        padding: 0.37rem 0.6rem;
        font-size: 0.68rem;
    }

    .hero {
        margin-top: 4.25rem;
        margin-bottom: 1.8rem;
    }

    .hero h1 {
        font-size: clamp(2rem, 9.1vw, 3rem);
        line-height: 1.14;
        letter-spacing: -0.042em;
    }

    .hero .hero-description {
        margin-top: 1rem;
        font-size: 0.94rem;
        line-height: 1.78;
    }

    .hero::after {
        margin-top: 1.25rem;
    }

    .flow-strip {
        align-items: flex-start;
        gap: 0.65rem;
        margin-bottom: 2rem;
        padding: 0.52rem 0.72rem;
    }

    .flow-item {
        flex-direction: column;
        gap: 0.25rem;
        font-size: 0.72rem;
    }

    .flow-arrow {
        margin-top: 0.55rem;
    }

    [data-testid="stVerticalBlockBorderWrapper"],
    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .glass-card-marker) {
        border-radius: 20px !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 1.25rem 1.1rem;
    }

    [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] .glass-card-marker) {
        padding: 1.25rem 1.1rem;
    }

    .section-heading {
        margin-bottom: 1.25rem;
    }

    .section-icon {
        width: 34px;
        height: 34px;
    }

    .results-section {
        margin-top: 2.35rem;
    }

    .results-heading-row .results-title {
        font-size: 1.55rem;
    }

    .results-heading-row {
        align-items: flex-start;
        flex-direction: column;
        gap: 0.45rem;
    }

    .result-banner {
        align-items: flex-start;
        flex-direction: column;
        gap: 0.65rem;
    }

    .result-metrics {
        grid-template-columns: 1fr;
        gap: 0.55rem;
    }

    .result-metric {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.8rem 0.9rem;
    }

    .result-metric-label {
        margin: 0;
    }

    [data-testid="stTabs"] [data-baseweb="tab"] {
        padding: 0 0.85rem;
        font-size: 0.75rem;
    }

    [data-testid="stTabs"] [role="tablist"] {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        overflow: visible;
    }

    [data-testid="stTabs"] [data-testid="stTab"] {
        flex: none;
        justify-content: center;
        padding: 0 0.65rem;
    }

    [data-testid="stTabs"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.35rem;
        line-height: 1.45;
    }

    .site-footer {
        flex-direction: column;
        gap: 0.65rem;
        margin-top: 2.5rem;
    }
}

@media (max-width: 390px) {
    .local-badge {
        border: none;
        background: transparent;
        padding-right: 0;
        box-shadow: none;
        backdrop-filter: none;
        -webkit-backdrop-filter: none;
    }

    .local-badge .status-dot {
        display: none;
    }

    .flow-strip {
        gap: 0.45rem;
    }

    .flow-item {
        font-size: 0.68rem;
    }
}

@media (prefers-reduced-motion: reduce) {
    html {
        scroll-behavior: auto !important;
    }

    .stApp::before {
        animation: none !important;
        transform: none !important;
    }

    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
"""

ICONS = {
    "spark": '<path d="m12 3-1.4 4.2a5 5 0 0 1-3.2 3.2L3 12l4.4 1.6a5 5 0 0 1 3.2 3.2L12 21l1.4-4.2a5 5 0 0 1 3.2-3.2L21 12l-4.4-1.6a5 5 0 0 1-3.2-3.2L12 3Z"/>',
    "edit": '<path d="M12 20h9"/><path d="M16.5 3.5a2.1 2.1 0 0 1 3 3L8 18l-4 1 1-4Z"/>',
    "sliders": '<path d="M4 21v-7"/><path d="M4 10V3"/><path d="M12 21v-9"/><path d="M12 8V3"/><path d="M20 21v-5"/><path d="M20 12V3"/><path d="M1 14h6"/><path d="M9 8h6"/><path d="M17 16h6"/>',
    "lock": '<rect width="16" height="12" x="4" y="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-8 9-4.5-1.5-8-4-8-9V5l8-3 8 3Z"/><path d="m9 12 2 2 4-4"/>',
    "check": '<path d="m5 12 4 4L19 6"/>',
}


def icon(name: str, size: int = 18, stroke_width: float = 1.8) -> str:
    """Return a small, consistent inline SVG icon for static UI chrome."""
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        f'stroke="currentColor" stroke-width="{stroke_width}" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{ICONS[name]}</svg>'
    )


def glass_card_marker() -> None:
    """标记 Streamlit 原生带边框容器，供跨版本的玻璃卡片样式定位。"""
    st.markdown('<span class="glass-card-marker" aria-hidden="true"></span>', unsafe_allow_html=True)


def section_heading(icon_name: str, title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="section-heading"><span class="section-icon">{icon(icon_name)}</span>'
        f'<div><h2 class="section-title">{title}</h2>'
        f'<p class="section-subtitle">{subtitle}</p></div></div>',
        unsafe_allow_html=True,
    )


st.markdown(APP_CSS, unsafe_allow_html=True)

st.markdown(
    f'''<header class="site-header">
        <div class="brand"><span class="brand-mark">{icon("spark", 17)}</span>校园创意验证</div>
        <div class="local-badge"><span class="status-dot"></span>本地运行 · 隐私优先</div>
    </header>''',
    unsafe_allow_html=True,
)

st.markdown(
    '''<section class="hero">
        <div class="eyebrow">From problem to proof</div>
        <h1>把一个校园问题，<br><span>变成可验证的方案。</span></h1>
        <p class="hero-description">描述你观察到的真实需求，多 Agent 将协作生成验证计划、项目方案与独立检查结果。先小规模验证，再决定下一步。</p>
    </section>
    <div class="flow-strip" aria-label="使用流程">
        <span class="flow-item"><span class="flow-number">01</span>输入校园问题</span>
        <span class="flow-arrow">→</span>
        <span class="flow-item"><span class="flow-number">02</span>生成验证计划</span>
        <span class="flow-arrow">→</span>
        <span class="flow-item"><span class="flow-number">03</span>查看完整方案</span>
    </div>''',
    unsafe_allow_html=True,
)

input_column, settings_column = st.columns([1.45, 1], gap="large")
with input_column:
    with st.container(border=True):
        glass_card_marker()
        section_heading("edit", "描述一个真实需求", "越具体的问题，越容易得到可执行的验证计划。")
        problem = st.text_area(
            "遇到的具体问题",
            placeholder="例如：新生常常不知道去哪补办校园卡，需要在多个网页和群聊里找信息。",
            height=116,
            key="problem_input",
        )
        audience = st.text_input(
            "主要服务谁",
            placeholder="例如：刚入学、还不熟悉办事流程的新生",
            key="audience_input",
        )
        evidence = st.text_area(
            "你已经观察或收集到的证据（可选）",
            placeholder="例如：问过 3 位同学，其中 2 位最近一周找过相关信息。没有证据可留空。",
            height=88,
            key="evidence_input",
        )
        st.markdown(
            f'<div class="privacy-note">{icon("shield", 14)}请勿输入身份证号、联系方式等敏感信息。</div>',
            unsafe_allow_html=True,
        )
        run_clicked = st.button("生成验证方案", type="primary", use_container_width=True)
        st.markdown(
            '<div class="button-caption">运行规划、方案与独立验证 Agent</div>',
            unsafe_allow_html=True,
        )

with settings_column:
    with st.container(border=True):
        glass_card_marker()
        section_heading("sliders", "运行设置", "先用模拟模式体验，或接入兼容的真实模型。")
        mode = st.radio(
            "运行方式",
            ["模拟模式（不调用真实模型）", "真实模型 API"],
            horizontal=True,
            label_visibility="collapsed",
            help="模拟模式使用本地固定逻辑，不调用真实模型。",
            key="run_mode",
        )
        is_simulation = mode.startswith("模拟")
        st.markdown('<div class="settings-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="settings-label">模型与接口</div>', unsafe_allow_html=True)
        base_url = st.text_input(
            "兼容接口地址",
            value=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            key="base_url_input",
            help="使用 OpenAI Chat Completions 兼容接口，应用会补充 /chat/completions 路径。",
        )
        model = st.text_input(
            "模型名称",
            value=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
            key="model_input",
        )
        has_api_key = bool(os.getenv("OPENAI_API_KEY", ""))
        if is_simulation:
            key_message = "模拟模式无需 API Key，不会发起模型网络请求。"
            key_class = ""
        elif has_api_key:
            key_message = "已从 OPENAI_API_KEY 环境变量读取密钥，页面不会显示或保存它。"
            key_class = " ready"
        else:
            key_message = "未检测到 OPENAI_API_KEY。请在启动应用前设置环境变量。"
            key_class = ""
        st.markdown(
            f'<div class="key-status{key_class}">{icon("lock", 14)}<span>{key_message}</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'''<div class="runtime-note">
                <div class="runtime-note-title">每次运行都包含</div>
                <div class="runtime-row"><span class="runtime-check">{icon("check", 11, 2.2)}</span>规划与方案协作</div>
                <div class="runtime-row"><span class="runtime-check">{icon("check", 11, 2.2)}</span>独立验证，最多返工 {MAX_RETRIES} 次</div>
                <div class="runtime-row"><span class="runtime-check">{icon("check", 11, 2.2)}</span>完整日志与报告下载</div>
            </div>''',
            unsafe_allow_html=True,
        )

if run_clicked:
    if not problem.strip() or not audience.strip():
        st.error("请先填写具体问题和目标用户，再生成验证方案。")
    else:
        client = LLMClient(mode, base_url, model, os.getenv("OPENAI_API_KEY", ""))
        with st.spinner("Agent 正在协作：生成计划、撰写方案并独立检查…"):
            try:
                st.session_state["last_result"] = run_orchestration(
                    problem.strip(), audience.strip(), evidence.strip(), client
                )
            except Exception as exc:
                st.error(f"运行失败：{exc}")

result = st.session_state.get("last_result")
if result:
    review = result["review"]
    is_simulation_result = result["mode"].startswith("模拟")
    review_passed = bool(review.get("passed"))
    knowledge_sources = result.get("knowledge_sources", [])
    created_at = result.get("created_at", "").replace("T", " ")[:19]
    mode_label = "模拟运行" if is_simulation_result else "真实模型 API"
    banner_text = (
        "本次结果由本地固定逻辑生成，用于体验完整流程，不代表真实模型能力。"
        if is_simulation_result
        else "本次结果由配置的真实模型生成，请在实际使用前人工复核。"
    )
    st.markdown(
        f'''<section class="results-section">
            <div class="results-heading-row">
                <div><div class="results-kicker">YOUR VALIDATION RESULT</div>
                <h2 class="results-title">验证结果已就绪</h2></div>
                <div class="result-time">{created_at}</div>
            </div>
            <div class="result-banner">
                <div class="result-banner-text"><strong>{mode_label}</strong> · {banner_text}</div>
                <span class="result-badge">{"本地确定性输出" if is_simulation_result else "已调用模型接口"}</span>
            </div>
            <div class="result-metrics">
                <div class="result-metric"><div class="result-metric-label">独立验证</div>
                    <div class="result-metric-value {"success" if review_passed else "warning"}">{"检查通过" if review_passed else "需要复核"}</div></div>
                <div class="result-metric"><div class="result-metric-label">Self-Loop 修订</div>
                    <div class="result-metric-value">{result["attempts"]} / {MAX_RETRIES} 次</div></div>
                <div class="result-metric"><div class="result-metric-label">匹配本地指南</div>
                    <div class="result-metric-value">{len(knowledge_sources)} 条</div></div>
            </div>
        </section>''',
        unsafe_allow_html=True,
    )

    solution_tab, plan_tab, review_tab, logs_tab = st.tabs(
        ["最终方案", "验证计划", "质量检查", "运行记录"]
    )
    with solution_tab:
        with st.container(border=True):
            glass_card_marker()
            st.markdown(result["solution"])
        st.markdown('<div class="download-row"></div>', unsafe_allow_html=True)
        markdown_download, json_download = st.columns(2)
        with markdown_download:
            st.download_button(
                "下载 Markdown 报告",
                to_markdown(result),
                file_name="campus_idea_agent_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with json_download:
            st.download_button(
                "下载 JSON 运行数据",
                json.dumps(result, ensure_ascii=False, indent=2),
                file_name="campus_idea_agent_run.json",
                mime="application/json",
                use_container_width=True,
            )

    with plan_tab:
        with st.container(border=True):
            glass_card_marker()
            st.markdown(result["plan"])
        st.markdown("##### 本地知识参考")
        st.caption("通过关键词匹配本地指南，不联网；这些指南只提供方法参考，不构成用户证据。")
        if knowledge_sources:
            chips = "".join(
                f'<span class="knowledge-chip">{item["title"]}</span>' for item in knowledge_sources
            )
            st.markdown(f'<div class="knowledge-list">{chips}</div>', unsafe_allow_html=True)
        else:
            st.write("本次没有匹配到指南。")

    with review_tab:
        with st.container(border=True):
            glass_card_marker()
            st.markdown("#### 独立验证报告")
            if review_passed:
                st.success("方案结构与证据表达检查通过。")
            else:
                st.warning("方案仍需人工复核或进一步修订。")
            st.write(review.get("summary", "本次未提供检查摘要。"))
            issues = review.get("issues", [])
            if issues:
                st.markdown("**需要关注的问题**")
                for issue in issues:
                    st.markdown(f"- {issue}")
            else:
                st.write("未发现需要额外列出的问题。")
            with st.expander("查看结构化检查数据"):
                st.json(review, expanded=False)

    with logs_tab:
        st.markdown("##### 可观测运行日志")
        st.dataframe(result["logs"], use_container_width=True, hide_index=True)
        st.markdown("##### 动态 Agent 派发")
        st.dataframe(result.get("scheduled_agents", []), use_container_width=True, hide_index=True)

st.markdown(
    '''<footer class="site-footer">
        <div><span class="footer-brand">校园创意验证 Agent</span><br>把假设留在方案里，把答案交给真实用户。</div>
        <div>方案中的目标值是待验证指标，不是已经取得的成果。<br>真实模型输出需人工复核，模拟输出不代表模型能力。</div>
    </footer>''',
    unsafe_allow_html=True,
)
