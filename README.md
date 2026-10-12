# Intrusion Detection System for IoT-Based Attacks using Deep Learning (CNN-LSTM)

Final-year project: a lightweight hybrid 1D-CNN + LSTM network intrusion detection system aimed at IoT edge gateways, evaluated on the UNSW-NB15 and Bot-IoT datasets.

## Contents

| File | Description |
|------|-------------|
| `unsw_nb15_v1.ipynb` | Training and evaluation on UNSW-NB15 (36 features) |
| `Bot_IoT_CNN_LSTM_v1.ipynb` | Training and evaluation on Bot-IoT (20 features) |
| `app.py` | Streamlit dashboard (NIDS Console) for scoring flows with a trained model |
| `requirements.txt` | Python dependencies for the dashboard |

## Approach

- Hybrid 1D-CNN + LSTM, with standalone CNN and LSTM baselines for comparison
- Metrics: Accuracy, Precision, Recall, F1 and MCC (primary, given class imbalance)
- Edge-readiness: model size, trainable parameters, per-flow inference latency, TensorFlow Lite conversion

## Running the dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a trained model (`.keras` recommended, or `.tflite`) and a flow CSV in the sidebar. The expected feature count is detected from the model, so one app serves both datasets. A `.tflite` model needs full TensorFlow because the LSTM uses Select-TF ops.

Datasets and trained models are not included in this repository. Download UNSW-NB15 and Bot-IoT from their official sources and run the notebooks (designed for Google Colab) to reproduce the models.
