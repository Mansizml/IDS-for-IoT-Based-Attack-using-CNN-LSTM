"""
IoT Intrusion Detection Dashboard  —  Hybrid 1D-CNN + LSTM
==========================================================
Model-agnostic: works with BOTH the UNSW-NB15 model (36 features) and the
Bot-IoT model (20 features). The expected feature count is auto-detected from
whichever model you load, so one app serves both.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py

Model formats (auto-detected):
    * .keras   -> tf.keras (most reliable; recommended)
    * .tflite  -> full TensorFlow (needs the Flex delegate: the LSTM uses
                  Select-TF ops, so the lightweight tflite-runtime CANNOT run it)

Data input (auto-detected):
    A) Preprocessed CSV  -> N model-ready feature columns (+ optional 'label').
                            Generic: works for any model. Faithful to your report.
    B) preprocess.json + raw CSV -> exact transform for EITHER dataset.
    C) Raw UNSW CSV with no artifacts -> approximate (built-in UNSW profile only).
"""

import json
import time
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc,
)

# --------------------------------------------------------------------------
# Built-in UNSW-NB15 profile (used only for the approximate raw-CSV fallback
# when no preprocess.json is supplied). Bot-IoT raw mode requires artifacts,
# because its selected features depend on your specific training run.
# --------------------------------------------------------------------------
UNSW_FEATURES = [
    'dbytes', 'sttl', 'dttl', 'dloss', 'Sload', 'Dload', 'Spkts', 'Dpkts',
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'tcprtt', 'synack',
    'ackdat', 'ct_state_ttl', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm',
    'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'proto_tcp', 'proto_udp', 'proto_unas', 'state_CON', 'state_FIN',
    'state_INT', 'service_-', 'service_dns', 'service_ftp-data',
    'service_pop3', 'service_ssh',
]
UNSW_CATEGORICAL = ['proto', 'state', 'service']
UNSW_LEAKY = ['srcip', 'dstip', 'sport', 'dsport', 'stime', 'ltime', 'attack_cat']

st.set_page_config(page_title='IoT IDS Dashboard', page_icon='🛡️', layout='wide')


# ==========================================================================
# Model loading  ->  (predict_fn, kind, n_features)
# ==========================================================================
@st.cache_resource(show_spinner='Loading model...')
def load_model(model_bytes: bytes, filename: str):
    suffix = '.keras' if filename.lower().endswith('.keras') else '.tflite'
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(model_bytes); tmp.flush(); tmp.close()

    if suffix == '.keras':
        import tensorflow as tf
        model = tf.keras.models.load_model(tmp.name)
        n_feat = int(model.input_shape[1])
        def predict_fn(x3d):
            return model.predict(x3d, verbose=0).ravel()
        return predict_fn, 'keras', n_feat

    # .tflite — needs full TensorFlow for the Flex (Select-TF) LSTM ops.
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=tmp.name)
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    n_feat = int(inp['shape'][1])
    interp.allocate_tensors()

    def predict_fn(x3d):
        try:
            interp.resize_tensor_input(inp['index'], [len(x3d), n_feat, 1])
            interp.allocate_tensors()
            interp.set_tensor(inp['index'], x3d.astype(np.float32))
            interp.invoke()
            return interp.get_tensor(out['index']).ravel()
        except Exception:
            probs = np.empty(len(x3d), dtype=np.float32)
            for i, row in enumerate(x3d):
                interp.set_tensor(inp['index'], row[None].astype(np.float32))
                interp.invoke()
                probs[i] = interp.get_tensor(out['index']).ravel()[0]
            return probs

    return predict_fn, 'tflite', n_feat


# ==========================================================================
# Helpers
# ==========================================================================
def split_label(df):
    cols = [str(c) for c in df.columns]
    lower = [c.lower() for c in cols]
    if 'label' in lower:
        name = cols[lower.index('label')]
        lab = pd.to_numeric(df[name], errors='coerce').fillna(0).astype(int).values
        lab = (lab > 0).astype(int)
        return df.drop(columns=[name]), lab
    return df, None


def is_preprocessed(df, n_feat):
    body, _ = split_label(df)
    return body.shape[1] == n_feat


def prep_preprocessed(df, n_feat):
    """Path A: already N model-ready features (+ optional label)."""
    body, label = split_label(df)
    X = body.apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
    return X.reshape(-1, n_feat, 1), label, 'exact'


def prep_with_artifacts(df, art):
    """Path B: faithful transform for ANY dataset using exported preprocess.json."""
    df = df.copy(); df.columns = [str(c) for c in df.columns]
    df, label = split_label(df)

    df = df.drop(columns=[c for c in art['leaky_cols'] if c in df.columns],
                 errors='ignore')
    for c, v in art.get('numeric_medians', {}).items():
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(v)
    cats = [c for c in art.get('categorical_cols', []) if c in df.columns]
    for c in cats:
        mode = art.get('cat_modes', {}).get(c, '-')
        df[c] = (df[c].astype(str).str.strip()
                 .replace({'': '-', 'nan': '-'}).fillna(mode))
    df = pd.get_dummies(df, columns=cats)

    df = df.reindex(columns=art['ohe_vocab'], fill_value=0)
    df = df.apply(pd.to_numeric, errors='coerce').astype('float64').fillna(0.0)

    mn = np.asarray(art['scaler_min'], dtype='float64')
    mx = np.asarray(art['scaler_max'], dtype='float64')
    denom = (mx - mn); denom[denom == 0] = 1.0
    scaled = pd.DataFrame((df.values - mn) / denom, columns=art['ohe_vocab'])

    sel_names = art['selected_feature_names']
    X = scaled.reindex(columns=sel_names, fill_value=0.0).values.astype(np.float32)
    return X.reshape(-1, len(sel_names), 1), label, 'exact'


def prep_raw_unsw(df, n_feat):
    """Path C: approximate UNSW transform (no fitted scaler embedded)."""
    df = df.copy(); df.columns = [str(c) for c in df.columns]
    df, label = split_label(df)
    df = df.drop(columns=[c for c in UNSW_LEAKY if c in df.columns], errors='ignore')
    cats = [c for c in UNSW_CATEGORICAL if c in df.columns]
    for c in cats:
        df[c] = df[c].astype(str).str.strip().replace({'': '-', 'nan': '-'})
    df = pd.get_dummies(df, columns=cats)
    df = df.reindex(columns=UNSW_FEATURES, fill_value=0)
    df = df.apply(pd.to_numeric, errors='coerce').astype('float64').fillna(0.0)
    mn, mx = df.min(), df.max()
    rng = (mx - mn).replace(0, 1)
    X = ((df - mn) / rng).values.astype(np.float32)
    return X.reshape(-1, n_feat, 1), label, 'approx'


def sniff_dataset(df):
    cols = {c.lower() for c in df.columns}
    if {'sttl', 'ct_state_ttl'} & cols or 'attack_cat' in cols:
        return 'UNSW-NB15'
    if {'saddr', 'pkseqid', 'subcategory'} & cols or ('attack' in cols and 'proto' in cols):
        return 'Bot-IoT'
    return 'unknown'


# ==========================================================================
# UI
# ==========================================================================
st.title('🛡️ IoT Intrusion Detection Dashboard')
st.caption('Hybrid 1D-CNN + LSTM · model-agnostic (UNSW-NB15 & Bot-IoT)')

with st.sidebar:
    st.header('1 · Model')
    mfile = st.file_uploader('Model (.keras or .tflite)', type=['keras', 'tflite'])
    threshold = st.slider('Attack threshold', 0.05, 0.95, 0.50, 0.05,
                          help='Flows with P(attack) ≥ threshold are flagged.')
    st.header('2 · Flows')
    dfile = st.file_uploader('Flow CSV', type=['csv'])
    st.header('3 · (optional) Preprocess artifacts')
    afile = st.file_uploader('preprocess.json — enables exact raw-CSV scoring '
                             'for either dataset', type=['json'])
    st.divider()
    st.caption('Easiest path: upload the preprocessed feature CSV exported from '
               'your notebook — it matches your report exactly and works for '
               'any model.')

if mfile is None:
    st.info('⬅️ Upload a model (.keras recommended, or .tflite) to begin.')
    st.stop()

predict_fn, kind, n_feat = load_model(mfile.getvalue(), mfile.name)
guess = 'UNSW-NB15' if n_feat == 36 else ('Bot-IoT' if n_feat == 20 else f'{n_feat}-feature')
st.sidebar.success(f'Model loaded ({kind}) · expects {n_feat} features → {guess}.')

if dfile is None:
    st.info(f'⬅️ Upload a flow CSV. This model expects **{n_feat} features**.')
    st.stop()

raw_df = pd.read_csv(dfile)
artifacts = json.load(afile) if afile is not None else None

# ---- Route to the right preprocessing path ----
if is_preprocessed(raw_df, n_feat):
    X3d, label, mode = prep_preprocessed(raw_df, n_feat)
elif artifacts is not None:
    X3d, label, mode = prep_with_artifacts(raw_df, artifacts)
elif n_feat == 36 and sniff_dataset(raw_df) == 'UNSW-NB15':
    X3d, label, mode = prep_raw_unsw(raw_df, n_feat)
else:
    ds = sniff_dataset(raw_df)
    st.error(f'This looks like a raw **{ds}** CSV but no `preprocess.json` was '
             f'provided, so it can\'t be transformed to the {n_feat} features '
             f'this model needs. Either upload the preprocessed feature CSV, or '
             f'add the `preprocess.json` exported from your notebook.')
    st.stop()

if X3d.shape[1] != n_feat:
    st.error(f'Feature mismatch: data has {X3d.shape[1]} features but the model '
             f'expects {n_feat}. Check that the CSV/artifacts match this model.')
    st.stop()

if mode == 'approx':
    st.warning('Raw approximate mode: predictions use an estimated scaler, so '
               'numbers may differ from your report. Upload the preprocessed '
               'CSV or `preprocess.json` for exact results.')

# ---- Inference (timed) ----
t0 = time.perf_counter()
probs = predict_fn(X3d)
elapsed = time.perf_counter() - t0
preds = (probs >= threshold).astype(int)
latency_ms = elapsed / max(len(X3d), 1) * 1000.0

n = len(preds); n_attack = int(preds.sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric('Flows scored', f'{n:,}')
c2.metric('Attacks flagged', f'{n_attack:,}')
c3.metric('Attack rate', f'{(n_attack / max(n,1)) * 100:.1f}%')
c4.metric('Latency', f'{latency_ms:.2f} ms/flow')

tab_alerts, tab_eval, tab_live = st.tabs(['🚨 Alerts', '📊 Evaluation', '▶️ Live demo'])

with tab_alerts:
    out = raw_df.copy()
    out.insert(0, 'P(attack)', np.round(probs, 4))
    out.insert(1, 'Verdict', np.where(preds == 1, '🚨 ATTACK', '✓ normal'))
    alerts = out[preds == 1].sort_values('P(attack)', ascending=False)
    st.subheader(f'{len(alerts):,} flagged flows')
    st.dataframe(alerts.head(500), use_container_width=True, height=420)
    st.download_button('Download alert log (CSV)',
                       alerts.to_csv(index=False).encode(),
                       file_name='ids_alerts.csv', mime='text/csv')

with tab_eval:
    if label is None:
        st.info('No `label` column found, so ground-truth metrics are '
                'unavailable. Add a `label` column (0=Normal, 1=Attack) to see '
                'the confusion matrix and scores.')
    else:
        if mode == 'exact':
            st.caption('Tip: on Bot-IoT, attacks dominate — read **MCC** and '
                       '**Normal-class recall**, not raw accuracy.')
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric('Accuracy',  f"{accuracy_score(label, preds):.4f}")
        m2.metric('Precision', f"{precision_score(label, preds, zero_division=0):.4f}")
        m3.metric('Recall',    f"{recall_score(label, preds, zero_division=0):.4f}")
        m4.metric('F1',        f"{f1_score(label, preds, zero_division=0):.4f}")
        m5.metric('MCC',       f"{matthews_corrcoef(label, preds):.4f}")
        try:
            fpr, tpr, _ = roc_curve(label, probs); roc_auc = auc(fpr, tpr)
        except Exception:
            fpr, tpr, roc_auc = [0, 1], [0, 1], float('nan')
        m6.metric('AUC', f'{roc_auc:.4f}')

        g1, g2 = st.columns(2)
        with g1:
            cm = confusion_matrix(label, preds)
            fig, ax = plt.subplots(figsize=(4.2, 3.6))
            im = ax.imshow(cm, cmap='Blues')
            for (i, j), v in np.ndenumerate(cm):
                ax.text(j, i, f'{v:,}', ha='center', va='center',
                        color='white' if v > cm.max() / 2 else 'black')
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Normal', 'Attack'])
            ax.set_yticklabels(['Normal', 'Attack'])
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
            ax.set_title('Confusion matrix')
            fig.colorbar(im, fraction=0.046); fig.tight_layout()
            st.pyplot(fig)
        with g2:
            fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
            ax2.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
            ax2.plot([0, 1], [0, 1], 'k--', lw=1)
            ax2.set_xlabel('False Positive Rate'); ax2.set_ylabel('True Positive Rate')
            ax2.set_title('ROC curve'); ax2.legend(loc='lower right')
            ax2.grid(alpha=0.3); fig2.tight_layout()
            st.pyplot(fig2)

with tab_live:
    st.caption('Replays the uploaded flows to mimic a real-time gateway feed.')
    speed = st.select_slider('Replay speed', ['slow', 'medium', 'fast'], 'medium')
    if st.button('▶️ Start replay'):
        delay = {'slow': 0.25, 'medium': 0.08, 'fast': 0.0}[speed]
        chunk = max(1, n // 50)
        prog = st.progress(0.0); kpi = st.empty(); log = st.empty()
        attacks = 0; rows = []
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            attacks += int(preds[start:end].sum())
            for idx in range(start, end):
                if preds[idx] == 1:
                    rows.append({'flow #': idx, 'P(attack)': round(float(probs[idx]), 4)})
            with kpi.container():
                k1, k2, k3 = st.columns(3)
                k1.metric('Processed', f'{end:,}/{n:,}')
                k2.metric('Attacks', f'{attacks:,}')
                k3.metric('Attack rate', f'{attacks / max(end,1) * 100:.1f}%')
            if rows:
                log.dataframe(pd.DataFrame(rows[-15:][::-1]),
                              use_container_width=True, height=300)
            prog.progress(end / n)
            if delay:
                time.sleep(delay)
        st.success(f'Replay complete — {attacks:,} attacks flagged in {n:,} flows.')
