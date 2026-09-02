import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

WINDOW_LEN = 150  # 5 segundos a 30fps
N_FEATURES = 22   # 5 AUs (1,4,6,12,15) * 4 classes = 20 + eye_openness + hr_bpm

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model_lstm_stress.keras")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"Modelo carregado: {MODEL_PATH}, input_shape={model.input_shape}")

app = FastAPI(title="LSTM repouso-vs-stress")

# Permite que o browser aceda a esta API localmente ou remotamente
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JanelaFeatures(BaseModel):
    # 150 instantes, 22 valores cada — já relativos à baseline pessoal de T1
    janela: list[list[float]]

    @field_validator("janela")
    @classmethod
    def validar_forma(cls, v):
        if len(v) != WINDOW_LEN:
            raise ValueError(f"esperava {WINDOW_LEN} instantes, recebi {len(v)}")
        for i, linha in enumerate(v):
            if len(linha) != N_FEATURES:
                raise ValueError(f"instante {i}: esperava {N_FEATURES} valores, recebi {len(linha)}")
        return v

@app.get("/")
def health():
    return {"status": "ok", "input_shape_esperado": [WINDOW_LEN, N_FEATURES]}

@app.post("/predict")
def predict(payload: JanelaFeatures):
    x = np.array(payload.janela, dtype="float32")[np.newaxis, :, :]
    proba = float(model.predict(x, verbose=0)[0, 0])
    return {
        "probabilidade_stress": proba,
        "classe": "stress" if proba >= 0.5 else "repouso",
    }
