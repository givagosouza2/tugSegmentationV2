import io
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

from scipy.signal import butter, filtfilt, detrend

# Plotly (para clique no gráfico)
import plotly.graph_objects as go

# Componente para capturar clique
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False

import sys, numpy, pandas
import scipy

print("PY:", sys.version)
print("NUMPY:", numpy.__version__)
print("PANDAS:", pandas.__version__)
print("SCIPY:", scipy.__version__)

# =========================
# Config
# =========================
st.set_page_config(page_title="TUG Gyro Segmentation", page_icon="🌀", layout="wide")

BASE_DIR = Path(__file__).resolve().parent

EVENTS = [
    ("t0_start", "Início do sinal"),
    ("t1_turn3m_start", "Início do componente de Giro em 3 m"),
    ("t2_turn3m_peak", "Pico do componente de Giro em 3 m"),
    ("t3_turn3m_end", "Final do componente de Giro em 3 m"),
    ("t4_turnchair_start", "Início do componente de Giro na frente da cadeira"),
    ("t5_turnchair_peak", "Pico do componente de Giro na frente da cadeira"),
    ("t6_turnchair_end", "Final do componente de Giro na frente da cadeira"),
    ("t7_end", "Final da atividade"),
]

DIFF_EVENT_OPTIONS = ["Nenhum"] + [label for _, label in EVENTS]

FS_TARGET = 100.0     # Hz
FC = 1.5              # Hz (lowpass)
FILTER_ORDER = 4
ACCEPT_TOL_SEC = 0.15 # tolerância treino (100 ms)

FILES_BASE = [
        'Pct 01_GYR.txt',
        'Pct 02_GYR.txt',
        'Pct 03_GYR.txt',
        'Pct 04_GYR.txt',
        'Pct 05_GYR.txt',
        'Pct 06_GYR.txt',
        'Pct 07_GYR.txt',
        'Pct 08_GYR.txt',
        'Pct 09_GYR.txt',
        'Pct 10_GYR.txt',
        'Pct 11_GYR.txt',
        'Pct 12_GYR.txt',
        'Pct 13_GYR.txt',
        'Pct 14_GYR.txt',
        'Pct 15_GYR.txt',
        'Pct 16_GYR.txt',
        'Pct 17_GYR.txt',
        'Pct 18_GYR.txt',
        'Pct 19_GYR.txt',
        'Pct 20_GYR.txt',
        'Pct 21_GYR.txt',
        'Pct 22_GYR.txt',
        'Pct 23_GYR.txt',
        'Pct 24_GYR.txt',
        'Pct 25_GYR.txt',
        'Pct 26_GYR.txt',
        'Pct 27_GYR.txt',
        'Pct 28_GYR.txt',
        'Pct 29_GYR.txt',
        'Pct 30_GYR.txt',
        'Pct 31_GYR.txt',
        'Pct 32_GYR.txt',
        'Pct 33_GYR.txt',
        'Pct 34_GYR.txt',
        'Pct 35_GYR.txt',
        'Pct 36_GYR.txt',
        'Pct 37_GYR.txt',
        'Pct 38_GYR.txt',
        'Pct 39_GYR.txt',
        'Pct 40_GYR.txt',
        'Pct 41_GYR.txt',
        'Pct 42_GYR.txt',
        'Pct 43_GYR.txt',
        'Pct 44_GYR.txt',
        'Pct 45_GYR.txt',
        'Pct 46_GYR.txt',
        'Pct 47_GYR.txt',
        'Pct 48_GYR.txt',
        'Pct 49_GYR.txt',
        'Pct 50_GYR.txt'
    ]

# =========================
# Leitura e pré-processamento
# =========================
def _read_semicolon_txt(file_like) -> pd.DataFrame:
    df = pd.read_csv(file_like, sep=";", engine="python")
    df.columns = [c.strip() for c in df.columns]
    return df

def _ensure_time_seconds(df: pd.DataFrame, time_col: str) -> np.ndarray:
    t = df[time_col].astype(float).to_numpy()
    if np.nanmax(t) > 200:  # heurística: ms
        t = t / 1000.0
    return t

def preprocess_gyro_norm(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    fs_target: float = FS_TARGET,
    fc: float = FC,
    order: int = FILTER_ORDER,
) -> Tuple[np.ndarray, np.ndarray]:
    x_d = detrend(x.astype(float), type="linear")
    y_d = detrend(y.astype(float), type="linear")
    z_d = detrend(z.astype(float), type="linear")

    idx = np.argsort(t)
    t_s = t[idx]
    x_d, y_d, z_d = x_d[idx], y_d[idx], z_d[idx]

    keep = np.concatenate(([True], np.diff(t_s) > 0))
    t_s = t_s[keep]
    x_d, y_d, z_d = x_d[keep], y_d[keep], z_d[keep]

    t0, t1 = float(t_s[0]), float(t_s[-1])
    t_u = np.arange(t0, t1, 1.0 / fs_target)
    x_i = np.interp(t_u, t_s, x_d)
    y_i = np.interp(t_u, t_s, y_d)
    z_i = np.interp(t_u, t_s, z_d)

    norm = np.sqrt(x_i**2 + y_i**2 + z_i**2)

    nyq = 0.5 * fs_target
    wn = fc / nyq
    b, a = butter(order, wn, btype="low", analog=False)
    norm_f = filtfilt(b, a, norm)

    return t_u, norm_f


# =========================
# Plotly + clique
# =========================
def make_plotly_fig(t: np.ndarray, norm: np.ndarray, event_times: Dict[str, float], title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=norm, line_color="black", mode="lines", name="||ω||"))

    # Eventos (linhas verticais)
    y_max = float(np.nanmax(norm)) if len(norm) else 1.0
    for key, label in EVENTS:
        v = event_times.get(key, None)
        if v is None:
            continue
        fig.add_vline(x=float(v), line_dash="dash", line_width=1, line_color="gray")
        fig.add_annotation(
            x=float(v) + 0.1,
            y=y_max,
            text=label,
            showarrow=False,
            textangle=-90,
            yanchor="top",
            font=dict(color="red", size=12),
        )

    fig.update_layout(
        title=title,
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Tempo (s)",
        yaxis_title="||ω|| (norma) (u.a.)",
    )
    return fig


# =========================
# Randomização por avaliador
# =========================
def _seed_from_text(text: str) -> int:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit

def _ensure_random_order_for_evaluator(evaluator_id: str) -> None:
    """
    Gera (uma vez) uma ordem específica para o avaliador e guarda no session_state.
    Se o avaliador mudar, refaz a ordem e reseta os registros/anotações.
    """
    ss = st.session_state
    evaluator_id = (evaluator_id or "").strip()

    if not evaluator_id:
        return

    # Se mudou o avaliador, reseta ordem e dados do estudo
    if ss.get("evaluator_id") != evaluator_id:
        ss["evaluator_id"] = evaluator_id
        ss["files_order"] = []
        ss["order_seed"] = None

        # reseta o que depende do conjunto/ordem
        ss["uploaded_records"] = []
        ss["record_index"] = 0
        ss["annotations_by_name"] = {}
        ss["cursor_time"] = None
        if "temp_event_times" in ss:
            del ss["temp_event_times"]

    # Se ainda não existe uma ordem, cria
    if not ss.get("files_order"):
        seed = _seed_from_text(evaluator_id)
        rng = np.random.default_rng(seed)
        order = FILES_BASE.copy()
        rng.shuffle(order)
        ss["files_order"] = order
        ss["order_seed"] = seed


# =========================
# State
# =========================
def init_state():
    ss = st.session_state
    ss.setdefault("consent", None)  # True/False/None
    ss.setdefault("experience", None)
    ss.setdefault("video_url", "")
    ss.setdefault("training_done", False)

    ss.setdefault("uploaded_records", [])  # list dicts {name, t_u, norm_f}
    ss.setdefault("record_index", 0)

    ss.setdefault("annotations_by_name", {})  # record_name -> ann dict
    ss.setdefault("started_at", time.strftime("%Y-%m-%d %H:%M:%S"))

    # Randomização / avaliador
    ss.setdefault("evaluator_id", "")
    ss.setdefault("files_order", [])
    ss.setdefault("order_seed", None)

    # Para modo clique/cursor:
    ss.setdefault("cursor_time", None)         # último tempo clicado
    ss.setdefault("cursor_event_key", EVENTS[0][0])  # evento selecionado para receber o cursor

init_state()
ss = st.session_state

def can_proceed_main() -> bool:
    return ss.get("consent") is True and ss.get("experience") is not None


# =========================
# Gabarito treino (preencher depois)
# =========================
EXPERT_TRAINING_EVENTS: Dict[str, float] = {
    # Exemplo:
    "t0_start": 3.07,
    "t1_turn3m_start": 6.34,
    "t2_turn3m_peak": 7.48,
    "t3_turn3m_end": 8.13,
    "t4_turnchair_start": 9.85,
    "t5_turnchair_peak": 10.77,
    "t6_turnchair_end": 11.42,
    "t7_end": 12.34
}


# =========================
# App
# =========================
st.title("🌀 Plataforma de Segmentação do Giroscópio no TUG (V2)")

tabs = st.tabs([
    "1) Consentimento",
    "2) Autoavaliação",
    "3) Vídeo Educativo",
    "4) Treinamento",
    "5) Segmentação (Estudo)",
])

# -------------------------
# TAB 1
# -------------------------
with tabs[0]:
    st.header("Apresentação & Consentimento")
    st.write(
        """
        Você irá **segmentar sinais de velocidade angular** do teste Timed Up and Go (TUG) marcando eventos temporais.
        O sistema irá registrar suas marcações, dificuldade percebida e eventos mais difíceis.
        """
    )
    consent_text = st.text_area(
        "Termo de Consentimento (modelo - personalize):",
        value=(
            "Declaro que fui informado(a) sobre os objetivos e procedimentos do estudo.\n"
            "Minha participação é voluntária e posso desistir a qualquer momento.\n"
            "Autorizo o uso das minhas marcações (anonimizadas) para fins de pesquisa."
        ),
        height=140,
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Aceito participar", type="primary"):
            ss.consent = True
    with c2:
        if st.button("⛔ Não quero participar"):
            ss.consent = False

    if ss.consent is True:
        st.success("Consentimento aceito.")
    elif ss.consent is False:
        st.error("Você recusou participar. O estudo será encerrado.")
        st.stop()

# -------------------------
# TAB 2
# -------------------------
with tabs[1]:
    st.header("Autoavaliação")
    if ss.consent is not True:
        st.info("Aceite o consentimento na Aba 1.")
        st.stop()

    ss.experience = st.radio(
        "Como você se avalia na tarefa de segmentação?",
        options=["Experiente", "Intermediário", "Inexperiente"],
        index=None,
        horizontal=True,
    )

# -------------------------
# TAB 3
# -------------------------
with tabs[2]:
    st.header("Vídeo Educativo")
    if ss.consent is not True:
        st.info("Aceite o consentimento na Aba 1.")
        st.stop()

    ss.video_url = "https://youtu.be/JiW_Q_KkX0M"
    uploaded_video = []
    if ss.video_url and ss.video_url.strip():
        st.video(ss.video_url)
    else:
        st.info("Nenhum vídeo disponível.")

    st.info("Assista quando quiser. Você pode voltar aqui a qualquer momento.")

# -------------------------
# TAB 4 - Treino
# -------------------------
with tabs[3]:
    st.header("Treinamento com comparação ao experimento")
    if not can_proceed_main():
        st.info("Você precisa aceitar o consentimento e preencher a autoavaliação.")
        st.stop()

    train_file = "Pct 21_GYR.txt"
    if train_file is None:
        st.warning("Envie o arquivo de treino para continuar.")
        st.stop()

    train_path = BASE_DIR / train_file
    if not train_path.exists():
        st.error(f"Arquivo de treino não encontrado no diretório principal: {train_path}")
        st.stop()

    df = _read_semicolon_txt(train_path)
    time_col = "DURACAO" if "DURACAO" in df.columns else df.columns[0]

    # padrão esperado
    x_col, y_col, z_col = "AVL EIXO X", "AVL EIXO Y", "AVL EIXO Z"
    if not all(c in df.columns for c in [x_col, y_col, z_col]):
        st.error("Não encontrei as colunas AVL EIXO X/Y/Z no arquivo de treino.")
        st.stop()

    t = _ensure_time_seconds(df, time_col)
    t_u, norm_f = preprocess_gyro_norm(t, df[x_col].values, df[y_col].values, df[z_col].values)

    # marcação por sliders (treino simples)
    tmin, tmax = float(t_u[0]), float(t_u[-1])

    cols = st.columns(3)
    user_events = {}

    for i, (key, label) in enumerate(EVENTS):
        with cols[i % 3]:
            user_events[key] = st.slider(
                label,
                min_value=tmin,
                max_value=tmax,
                value=tmin,
                step=0.01,
                key=f"train_{key}",
            )

    fig = make_plotly_fig(t_u, norm_f, user_events)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    if not EXPERT_TRAINING_EVENTS:
        st.warning(
            "⚠️ O gabarito do treino ainda não foi configurado (EXPERT_TRAINING_EVENTS vazio).\n"
            "Quando você preencher os tempos do experimento para este arquivo de treino, a comparação ficará automática."
        )
        ss.training_done = False
    else:
        diffs = {k: abs(float(user_events[k]) - float(EXPERT_TRAINING_EVENTS[k])) for k, _ in EVENTS}
        ok_all = all(d <= ACCEPT_TOL_SEC for d in diffs.values())

        if ok_all:
            st.success(f"✅ Treino aceitável (tolerância ±{ACCEPT_TOL_SEC:.2f}s).")
            ss.training_done = True
        else:
            st.error(f"⛔ Treino ainda não aceitável (tolerância ±{ACCEPT_TOL_SEC:.2f}s).")
            ss.training_done = False

        st.dataframe(pd.DataFrame([
            {
                "Evento": label,
                "Seu tempo (s)": round(float(user_events[k]), 3),
                "Gabarito (s)": round(float(EXPERT_TRAINING_EVENTS[k]), 3),
                "Erro abs (s)": round(float(diffs[k]), 3),
                "Aceitável?": "Sim" if diffs[k] <= ACCEPT_TOL_SEC else "Não"
            }
            for k, label in EVENTS
        ]), use_container_width=True)

# -------------------------
# TAB 5 - Estudo
# -------------------------
with tabs[4]:
    st.header("Segmentação (Estudo)")

    if not can_proceed_main():
        st.info("Você precisa aceitar o consentimento e preencher a autoavaliação.")
        st.stop()

    # -------- Identidade do avaliador + randomização
    st.subheader("0) Identificação do avaliador")
    evaluator_id = st.text_input(
        "Identidade do avaliador (ex.: AVAL_01, JOAO_02)",
        value=ss.get("evaluator_id", ""),
        max_chars=40
    ).strip()

    if not evaluator_id:
        st.warning("Informe a identidade do avaliador para gerar a sequência específica.")
        st.stop()

    _ensure_random_order_for_evaluator(evaluator_id)

    st.caption(f"Seed (registrável): {ss.order_seed}")
    st.caption("Sequência de apresentação (específica deste avaliador):")
    st.code("\n".join(ss.files_order))

    # -------- Registros (arquivos locais)
    st.subheader("1) Registros do estudo (arquivos locais)")
    files = ss.files_order  # <- ordem aleatorizada por avaliador (persistente)

    if files:
        current_names = files
        cached_names = [r["name"] for r in ss.uploaded_records] if ss.uploaded_records else []

        if current_names != cached_names:
            ss.uploaded_records = []
            ss.record_index = 0
            ss.annotations_by_name = {}
            ss.cursor_time = None
            if "temp_event_times" in ss:
                del ss["temp_event_times"]

            for f in files:
                path = BASE_DIR / f
                if not path.exists():
                    st.error(f"Arquivo não encontrado no diretório principal: {path}")
                    st.stop()

                df = _read_semicolon_txt(path)
                time_col = "DURACAO" if "DURACAO" in df.columns else df.columns[0]

                # tenta padrão; fallback pega últimas 3 colunas
                x_col, y_col, z_col = "AVL EIXO X", "AVL EIXO Y", "AVL EIXO Z"
                if not all(c in df.columns for c in [x_col, y_col, z_col]):
                    if df.shape[1] >= 4:
                        x_col, y_col, z_col = df.columns[-3], df.columns[-2], df.columns[-1]
                    else:
                        st.error(f"{f}: não consegui inferir colunas X/Y/Z.")
                        st.stop()

                t = _ensure_time_seconds(df, time_col)
                t_u, norm_f = preprocess_gyro_norm(t, df[x_col].values, df[y_col].values, df[z_col].values)

                ss.uploaded_records.append({"name": f, "t_u": t_u, "norm_f": norm_f})

    if not ss.uploaded_records:
        st.info("Envie os arquivos para começar.")
        st.stop()

    n = len(ss.uploaded_records)

    # -------- Revisão geral
    st.subheader("2) Revisão geral (o que já foi feito)")
    review_rows = []
    for rec in ss.uploaded_records:
        name = rec["name"]
        ann = ss.annotations_by_name.get(name)
        review_rows.append({
            "Registro": name,
            "Salvo?": "Sim" if ann else "Não",
            "Dificuldade": ann["difficulty_1to7"] if ann else None,
            "Eventos difíceis": "|".join(ann["hardest_events"]) if ann else None,
        })
    df_review = pd.DataFrame(review_rows)
    st.dataframe(df_review, use_container_width=True)

    # Ir para o primeiro pendente
    pendings = [i for i, rec in enumerate(ss.uploaded_records) if rec["name"] not in ss.annotations_by_name]
    col_jump1, col_jump2 = st.columns([1, 2])
    with col_jump1:
        if st.button("➡️ Ir para o primeiro pendente", disabled=(len(pendings) == 0)):
            ss.record_index = pendings[0]
            ss.cursor_time = None
            st.rerun()
    with col_jump2:
        st.caption(f"Pendente(s): {len(pendings)} / {n}")

    st.divider()

    # -------- Registro atual
    i = int(ss.record_index)
    rec = ss.uploaded_records[i]
    t_u = rec["t_u"]
    norm_f = rec["norm_f"]
    tmin, tmax = float(t_u[0]), float(t_u[-1])

    st.subheader(f"3) Registro {i+1}/{n}: {rec['name']}")

    # Recupera marcação existente (se houver)
    existing = ss.annotations_by_name.get(rec["name"])
    event_times = {k: None for k, _ in EVENTS}
    if existing:
        event_times.update(existing["events_sec"])

    # Defaults se estiver vazio
    if all(v is None for v in event_times.values()):
        for idx_ev, (k, _) in enumerate(EVENTS):
            event_times[k] = float(min(tmax, tmin + 0.5 * idx_ev))

    # -------- Clique no gráfico
    st.markdown("### Marcação por clique no gráfico")

    if PLOTLY_EVENTS_AVAILABLE:
        st.caption("Clique no gráfico para escolher um tempo. Depois selecione o evento e atribua.")
        fig = make_plotly_fig(t_u, norm_f, event_times, title="Clique no sinal para definir o cursor")
        clicked = plotly_events(fig, click_event=True, hover_event=False, select_event=False, override_height=380)

        if clicked:
            ss.cursor_time = float(clicked[0]["x"])

        cA, cB, cC = st.columns([2, 2, 2])
        with cA:
            st.write(f"**Cursor:** {ss.cursor_time:.3f} s" if ss.cursor_time is not None else "**Cursor:** (clique no gráfico)")
        with cB:
            ss.cursor_event_key = st.selectbox(
                "Evento para receber o cursor",
                options=[k for k, _ in EVENTS],
                format_func=lambda k: dict(EVENTS)[k],
                index=[k for k, _ in EVENTS].index(ss.cursor_event_key) if ss.cursor_event_key in [k for k, _ in EVENTS] else 0
            )
        with cC:
            if st.button("📌 Atribuir ao evento", disabled=(ss.cursor_time is None)):
                event_times[ss.cursor_event_key] = float(ss.cursor_time)
                ss.setdefault("temp_event_times", {})
                ss.temp_event_times = event_times
                st.success("Atribuído.")
    else:
        st.warning(
            "Clique no gráfico indisponível porque `streamlit-plotly-events` não está instalado.\n"
            "Instale com: pip install streamlit-plotly-events"
        )

    # Se houve atribuições, recarrega do temp
    if "temp_event_times" in ss and isinstance(ss.temp_event_times, dict):
        event_times = ss.temp_event_times

    # -------- Ajuste fino por sliders
    st.markdown("### Ajuste fino (sliders)")
    cols = st.columns(3)
    for idx_ev, (k, label) in enumerate(EVENTS):
        if idx_ev == 0 or idx_ev == 1 or idx_ev == 2:
            with cols[0]:
                event_times[k] = st.slider(
                    label,
                    tmin, tmax,
                    float(event_times[k]) if event_times[k] is not None else tmin,
                    0.01,
                    key=f"slider_{rec['name']}_{k}"
                    )
        elif idx_ev == 3 or idx_ev == 4 or idx_ev == 5:
            with cols[1]:
                event_times[k] = st.slider(
                    label,
                    tmin, tmax,
                    float(event_times[k]) if event_times[k] is not None else tmin,
                    0.01,
                    key=f"slider_{rec['name']}_{k}"
                    )
        else:
            with cols[2]:
                event_times[k] = st.slider(
                    label,
                    tmin, tmax,
                    float(event_times[k]) if event_times[k] is not None else tmin,
                    0.01,
                    key=f"slider_{rec['name']}_{k}"
                    )

    # checagem de ordem
    times_list = [event_times[k] for k, _ in EVENTS]
    #if any(np.diff(times_list) < 0):
    #    st.warning("⚠️ Alguns eventos ficaram fora de ordem temporal (um evento está antes do anterior).")

    # gráfico final com eventos
    st.plotly_chart(make_plotly_fig(t_u, norm_f, event_times, title="Sinal processado (norma)"), use_container_width=True)

    # -------- Dificuldade
    st.markdown("### Dificuldade percebida")
    default_diff = int(existing["difficulty_1to7"]) if existing else 3
    difficulty = st.slider("Quão difícil foi segmentar este registro?", 1, 7, default_diff, 1)

    default_hard = existing["hardest_events"] if existing else ["Nenhum"]
    hardest_events = st.multiselect(
        "Quais eventos foram mais difíceis de identificar?",
        options=DIFF_EVENT_OPTIONS,
        default=default_hard
    )
    if "Nenhum" in hardest_events and len(hardest_events) > 1:
        st.error("Se selecionar 'Nenhum', não selecione outros eventos.")
        st.stop()

    # -------- Botões: salvar/navegar/exportar
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("💾 Salvar marcação", type="primary"):
            ann = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "started_at": ss.started_at,
                "evaluator_experience": ss.experience,
                "evaluator_id": ss.evaluator_id,
                "order_seed": ss.order_seed,
                "files_order": list(ss.files_order),
                "record_index": i,
                "record_name": rec["name"],
                "events_sec": {k: float(event_times[k]) for k, _ in EVENTS},
                "difficulty_1to7": int(difficulty),
                "hardest_events": hardest_events,
            }
            ss.annotations_by_name[rec["name"]] = ann
            st.success("Salvo.")

    with col2:
        if st.button("⬅️ Anterior", disabled=(i == 0)):
            ss.record_index = max(0, i - 1)
            ss.cursor_time = None
            st.rerun()

    with col3:
        if st.button("➡️ Próximo", disabled=(i == n - 1)):
            ss.record_index = min(n - 1, i + 1)
            ss.cursor_time = None
            st.rerun()

    with col4:
        st.markdown("### Exportar relatório")
        if len(ss.annotations_by_name) == 0:
            st.info("Nenhuma marcação salva ainda.")
        else:
            # CSV
            rows = []
            for name, a in ss.annotations_by_name.items():
                row = {
                    "timestamp": a["timestamp"],
                    "started_at": a["started_at"],
                    "evaluator_experience": a["evaluator_experience"],
                    "evaluator_id": a.get("evaluator_id", ""),
                    "order_seed": a.get("order_seed", ""),
                    "files_order": "|".join(a.get("files_order", [])),
                    "record_name": a["record_name"],
                    "difficulty_1to7": a["difficulty_1to7"],
                    "hardest_events": "|".join(a["hardest_events"]),
                }
                for k, _ in EVENTS:
                    row[k] = a["events_sec"][k]
                rows.append(row)

            df_out = pd.DataFrame(rows).sort_values(["record_name"])
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")

            st.download_button(
                "⬇️ Baixar CSV",
                data=csv_bytes,
                file_name="tug_segmentation_report.csv",
                mime="text/csv",
            )

            # JSON
            json_bytes = json.dumps(list(ss.annotations_by_name.values()), ensure_ascii=False, indent=2).encode("utf-8")
            st.download_button(
                "⬇️ Baixar JSON",
                data=json_bytes,
                file_name="tug_segmentation_report.json",
                mime="application/json",
            )

            st.caption(f"{len(ss.annotations_by_name)} registro(s) salvos de {n}.")
