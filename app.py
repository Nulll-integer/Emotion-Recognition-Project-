# app.py
# Engagement Register — Real-Time Emotion Recognition
# Run: streamlit run app.py

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import cv2
import time
import datetime
from collections import Counter
from utils.predictor import (
    predict_emotion, detect_and_predict, EMOTION_EMOJI, ENGAGEMENT_MAP
)

st.set_page_config(page_title="Engagement Register", page_icon="📋", layout="wide")

# ─── FLAT COLOR PALETTE — overrides utils.predictor's neon set ──
COLOR = {
    'happy'   : '#2B5D3E',
    'neutral' : '#5B5A52',
    'sad'     : '#335C82',
    'surprise': '#A87B16',
    'fear'    : '#6B4C7A',
    'angry'   : '#A8362B',
}
INK, INK_SOFT, PAPER_LINE = '#1C1B17', '#6B6859', '#E4E1D5'

# ─── STYLE ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@600;700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

.stApp { background:#FBFAF6; color:#1C1B17; }
#MainMenu, footer, header { visibility:hidden; }
.block-container { padding:18px 30px !important; max-width:1280px !important; }
* { font-family:'Inter',sans-serif; }

.eyebrow { font:600 11px/1 'Inter'; letter-spacing:1.5px; text-transform:uppercase; color:#6B6859; }
.hero-title { font:700 32px/1.15 'Source Serif 4'; color:#1C1B17; margin:6px 0 4px 0; }
.hero-sub { font:400 14px/1.4 'Inter'; color:#6B6859; }
.rule { border-top:1px solid #E4E1D5; margin:14px 0 18px 0; }

.section-label { font:600 11px/1 'Inter'; letter-spacing:1.5px; text-transform:uppercase;
    color:#6B6859; border-bottom:1px solid #E4E1D5; padding-bottom:8px; margin-bottom:14px; }

.feed-meta { font:500 12px/1 'IBM Plex Mono'; color:#6B6859; margin-top:8px; }

.reading-row { display:flex; align-items:center; gap:14px; padding:10px 0; }
.reading-emoji { font-size:2.6em; line-height:1; }
.reading-name { font:700 22px/1 'Source Serif 4'; text-transform:capitalize; }
.reading-conf { font:600 14px/1 'IBM Plex Mono'; color:#6B6859; margin-left:auto; }

.stamp-wrap { display:flex; align-items:center; gap:18px; padding:14px 0; }
.stamp { width:84px; height:84px; flex-shrink:0; border:2px solid; outline:1px solid;
    outline-offset:4px; border-radius:6px; display:flex; flex-direction:column;
    align-items:center; justify-content:center; transform:rotate(-3deg); }
.stamp-num { font:700 26px/1 'IBM Plex Mono'; }
.stamp-den { font:500 10px/1 'Inter'; letter-spacing:1px; }
.eng-label { font:600 15px/1.3 'Inter'; }
.eng-desc { font:400 13px/1.4 'Inter'; color:#6B6859; margin-top:4px; }

.prob-row, .log-row { display:flex; align-items:center; gap:10px; padding:7px 0;
    border-bottom:1px solid #E4E1D5; }
.prob-row:last-child, .log-row:last-child { border-bottom:none; }
.row-emo { width:80px; font:500 13px/1 'Inter'; flex-shrink:0; }
.row-track { flex-grow:1; height:6px; background:#E4E1D5; border-radius:2px; overflow:hidden; }
.row-fill { height:100%; border-radius:2px; }
.row-num { width:46px; text-align:right; font:600 12px/1 'IBM Plex Mono'; color:#1C1B17; flex-shrink:0; }
.log-swatch { width:8px; height:8px; border-radius:1px; flex-shrink:0; }

.status-chip { font:500 12px/1 'IBM Plex Mono'; padding:6px 0; }
.dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:6px; }

.idle-box { padding:48px 24px; text-align:center; border:1px solid #E4E1D5; border-radius:6px; }
.idle-box p { color:#6B6859; font-size:13px; max-width:360px; margin:6px auto 0; }

.stButton > button { border-radius:4px !important; font:600 13px/1 'Inter' !important;
    letter-spacing:0.3px !important; box-shadow:none !important; }
.stButton > button[kind="primary"] { background:#2B5D3E !important; border:1px solid #2B5D3E !important; color:white !important; }
.stButton > button[kind="secondary"] { background:transparent !important; border:1px solid #1C1B17 !important; color:#1C1B17 !important; }
div[data-testid="stImage"] img { border:1px solid #E4E1D5 !important; border-radius:4px !important; box-shadow:none !important; }
</style>
""", unsafe_allow_html=True)

today = datetime.date.today().strftime('%B %d, %Y')

# ─── HEADER ──────────────────────────────────────────────────
st.markdown(f"""
<div class="hero-title">Engagement Register</div>
<div class="hero-sub">Real-time facial expression reading for online classroom engagement </div>
<div class="rule"></div>
""", unsafe_allow_html=True)

# ─── STATE ───────────────────────────────────────────────────
for k, v in [('running', False), ('history', []), ('frames', 0)]:
    if k not in st.session_state:
        st.session_state[k] = v

c1, c2, c3, c4 = st.columns([2, 2, 2, 6])
with c1:
    start = st.button("Start session", type="primary", use_container_width=True)
with c2:
    stop = st.button("End session", type="secondary", use_container_width=True)
with c3:
    rate_label = st.select_slider("Update rate", options=["Fast", "Standard", "Slow"], value="Standard")
    rate = {"Fast": 0.2, "Standard": 0.5, "Slow": 1.0}[rate_label]
with c4:
    status_ph = st.empty()

if start:
    st.session_state.running, st.session_state.history, st.session_state.frames = True, [], 0
if stop:
    st.session_state.running = False

cam_col, res_col = st.columns([3, 2], gap="large")
with cam_col:
    feed_label_ph = st.empty()
    cam_ph = st.empty()
    feed_meta_ph = st.empty()
with res_col:
    reading_ph = st.empty()
    stamp_ph = st.empty()
    prob_ph = st.empty()
    log_ph = st.empty()


def render_reading(emotion, conf):
    color = COLOR.get(emotion, INK)
    reading_ph.markdown(f"""
    <div class="section-label">Current reading</div>
    <div class="reading-row">
        <div class="reading-emoji">{EMOTION_EMOJI.get(emotion,'—')}</div>
        <div class="reading-name" style="color:{color};">{emotion}</div>
        <div class="reading-conf">{conf:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)


def render_stamp(emotion):
    label, _, score = ENGAGEMENT_MAP.get(emotion, ('Reading…', '', 50))
    color = '#2B5D3E' if score >= 70 else ('#A87B16' if score >= 40 else '#A8362B')
    stamp_ph.markdown(f"""
    <div class="section-label">Engagement</div>
    <div class="stamp-wrap">
        <div class="stamp" style="border-color:{color}; outline-color:{color};">
            <div class="stamp-num" style="color:{color};">{score}</div>
            <div class="stamp-den">/ 100</div>
        </div>
        <div>
            <div class="eng-label" style="color:{color};">{label}</div>
            <div class="eng-desc">Derived from the dominant facial expression in the current frame.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_probs(scores):
    rows = "".join(f"""
    <div class="prob-row">
        <div class="row-emo">{e}</div>
        <div class="row-track"><div class="row-fill" style="width:{min(s,100):.1f}%;background:{COLOR.get(e,INK)};"></div></div>
        <div class="row-num">{s:.0f}%</div>
    </div>""" for e, s in sorted(scores.items(), key=lambda x: x[1], reverse=True))
    prob_ph.markdown(f'<div class="section-label">Probability breakdown</div>{rows}', unsafe_allow_html=True)


def render_log(history):
    if len(history) < 2:
        log_ph.markdown('<div class="section-label">Session log</div>'
                         '<div style="color:#6B6859;font-size:13px;">Collecting readings…</div>',
                         unsafe_allow_html=True)
        return
    counts, total = Counter(history), len(history)
    rows = "".join(f"""
    <div class="log-row">
        <div class="log-swatch" style="background:{COLOR.get(e,INK)};"></div>
        <div class="row-emo">{e}</div>
        <div class="row-track"><div class="row-fill" style="width:{c/total*100:.1f}%;background:{COLOR.get(e,INK)};"></div></div>
        <div class="row-num">×{c}</div>
    </div>""" for e, c in counts.most_common())
    log_ph.markdown(f'<div class="section-label">Session log · {total} readings</div>{rows}', unsafe_allow_html=True)


def show_idle():
    feed_label_ph.markdown('<div class="section-label">Session feed</div>', unsafe_allow_html=True)
    cam_ph.markdown("""
    <div class="idle-box">
        <div style="font:700 16px 'Source Serif 4'; color:#1C1B17;">No session running</div>
        <p>Start a session to begin reading facial expressions from the camera.</p>
    </div>""", unsafe_allow_html=True)
    feed_meta_ph.empty()
    status_ph.markdown('<div class="status-chip"><span class="dot" style="background:#E4E1D5;"></span>Idle</div>', unsafe_allow_html=True)
    render_reading('—', 0.0)
    render_stamp('—')
    prob_ph.markdown('<div class="section-label">Probability breakdown</div>'
                      '<div style="color:#6B6859;font-size:13px;">No data yet.</div>', unsafe_allow_html=True)
    render_log([])


if not st.session_state.running:
    show_idle()
else:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Camera unavailable. Close any other app using the camera, then start the session again.")
        st.session_state.running = False
        show_idle()
    else:
        status_ph.markdown('<div class="status-chip"><span class="dot" style="background:#2B5D3E;"></span>Live</div>', unsafe_allow_html=True)
        last_emo, last_conf, last_scores = 'neutral', 0.0, {e: 0.0 for e in EMOTION_EMOJI}

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.error("Lost the camera feed. End the session and start again.")
                break
            st.session_state.frames += 1

            annotated, results = detect_and_predict(frame)
            if results:
                last_emo, last_conf, last_scores = results[0]['emotion'], results[0]['confidence'], results[0]['scores']
                st.session_state.history.append(last_emo)
                if len(st.session_state.history) > 100:
                    st.session_state.history.pop(0)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                last_emo, last_conf, last_scores = predict_emotion(gray)

            feed_label_ph.markdown('<div class="section-label">Session feed</div>', unsafe_allow_html=True)
            cam_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            face_note = f"{len(results)} face detected" if results else "no face detected — reading full frame"
            feed_meta_ph.markdown(f'<div class="feed-meta">Frame {st.session_state.frames} · {face_note}</div>', unsafe_allow_html=True)

            render_reading(last_emo, last_conf)
            render_stamp(last_emo)
            render_probs(last_scores)
            render_log(st.session_state.history)

            time.sleep(rate)
            if not st.session_state.running:
                break

        cap.release()
        status_ph.markdown('<div class="status-chip"><span class="dot" style="background:#E4E1D5;"></span>Idle</div>', unsafe_allow_html=True)
        show_idle()

st.markdown('<div class="rule"></div><div style="color:#6B6859;font-size:11px;">'
            'CNN · FER-2013 · 67.09% test accuracy</div>', unsafe_allow_html=True)
        