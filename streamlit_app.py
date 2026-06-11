# ============================================================
#  Tablero BI + Ciencia de Datos — Predicción de Enfermedad Cardiaca
#  Dataset: Heart Failure Prediction (fedesoriano, Kaggle)
# ============================================================
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, recall_score,
                             precision_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ------------------------------------------------------------
# Configuración general y paleta
# ------------------------------------------------------------
st.set_page_config(
    page_title="Riesgo Cardiaco | BI & Data Science",
    page_icon="🫀",
    layout="wide",
)

C_SANO = "#2A9D8F"      # teal  -> HeartDisease = 0
C_ENFERMO = "#E63946"   # rojo  -> HeartDisease = 1
C_INK = "#1D3557"
PALETA_TARGET = {"Sano": C_SANO, "Enfermedad": C_ENFERMO}

NUM_COLS = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
CAT_NOMINAL = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina"]
CAT_ORDINAL = ["ST_Slope"]
BIN_COLS = ["FastingBS"]

ETIQUETAS = {
    "Age": "Edad (años)", "Sex": "Sexo", "ChestPainType": "Tipo de dolor de pecho",
    "RestingBP": "Presión arterial en reposo (mmHg)", "Cholesterol": "Colesterol (mg/dl)",
    "FastingBS": "Glucosa en ayunas > 120 mg/dl", "RestingECG": "ECG en reposo",
    "MaxHR": "Frecuencia cardiaca máxima", "ExerciseAngina": "Angina por ejercicio",
    "Oldpeak": "Oldpeak (depresión ST)", "ST_Slope": "Pendiente del segmento ST",
}

# ------------------------------------------------------------
# Carga de datos: Kaggle (kagglehub) con respaldo local
# ------------------------------------------------------------
@st.cache_data(show_spinner="Descargando dataset desde Kaggle…")
def cargar_datos() -> pd.DataFrame:
    try:
        import kagglehub
        path = kagglehub.dataset_download("fedesoriano/heart-failure-prediction")
        df = pd.read_csv(os.path.join(path, "heart.csv"))
    except Exception:
        # Respaldo: copia local incluida en el repo
        df = pd.read_csv("heart.csv")
    return df


@st.cache_data
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Ceros fisiológicamente imposibles -> NaN -> imputación por mediana."""
    d = df.copy()
    d["Cholesterol"] = d["Cholesterol"].replace(0, np.nan)
    d["RestingBP"] = d["RestingBP"].replace(0, np.nan)
    for col in ["Cholesterol", "RestingBP"]:
        d[col] = d[col].fillna(d[col].median())
    d["Target"] = d["HeartDisease"].map({0: "Sano", 1: "Enfermedad"})
    return d


df_raw = cargar_datos()
df = limpiar_datos(df_raw)

# ------------------------------------------------------------
# Entrenamiento de modelos (cacheado)
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando modelos…")
def entrenar_modelos(data: pd.DataFrame):
    X = data[NUM_COLS + CAT_NOMINAL + CAT_ORDINAL + BIN_COLS]
    y = data["HeartDisease"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("nom", OneHotEncoder(handle_unknown="ignore"), CAT_NOMINAL),
        ("ord", OrdinalEncoder(categories=[["Up", "Flat", "Down"]]), CAT_ORDINAL),
        ("bin", "passthrough", BIN_COLS),
    ])

    modelos = {
        "Regresión Logística": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(
                C=0.5, class_weight="balanced",
                solver="liblinear", max_iter=2000, random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=3,
                max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1)),
        ]),
    }

    resultados = {}
    for nombre, pipe in modelos.items():
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = pipe.predict(X_te)
        resultados[nombre] = {
            "pipe": pipe,
            "recall": recall_score(y_te, pred),
            "precision": precision_score(y_te, pred),
            "f1": f1_score(y_te, pred),
            "auc": roc_auc_score(y_te, proba),
            "cm": confusion_matrix(y_te, pred),
            "roc": roc_curve(y_te, proba),
        }
    return resultados, X_te, y_te


resultados, X_te, y_te = entrenar_modelos(df)

# ------------------------------------------------------------
# Encabezado
# ------------------------------------------------------------
st.title("🫀 Tablero de riesgo cardiaco")
st.markdown(
    f"**918 pacientes · 11 variables clínicas · objetivo: detectar enfermedad cardiaca a tiempo.** "
    f"Tasa de enfermedad en la muestra: "
    f"<span style='color:{C_ENFERMO};font-weight:700'>{df['HeartDisease'].mean():.0%}</span>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Filtros globales (sidebar) — elementos interactivos
# ------------------------------------------------------------
st.sidebar.header("🎚️ Filtros del tablero")
rango_edad = st.sidebar.slider(
    "Rango de edad", int(df.Age.min()), int(df.Age.max()), (28, 77)
)
sexo_sel = st.sidebar.multiselect("Sexo", ["M", "F"], default=["M", "F"])
dolor_sel = st.sidebar.multiselect(
    "Tipo de dolor de pecho",
    sorted(df.ChestPainType.unique()),
    default=sorted(df.ChestPainType.unique()),
    help="ASY = asintomático, ATA = angina atípica, NAP = dolor no anginoso, TA = angina típica",
)

df_f = df[
    df.Age.between(*rango_edad)
    & df.Sex.isin(sexo_sel if sexo_sel else ["M", "F"])
    & df.ChestPainType.isin(dolor_sel if dolor_sel else df.ChestPainType.unique())
]
st.sidebar.metric("Pacientes en el filtro", f"{len(df_f)} / {len(df)}")
st.sidebar.caption("Los filtros afectan las pestañas **Datos** y **Exploración**.")

tab_datos, tab_eda, tab_modelo, tab_pred = st.tabs(
    ["📋 Los datos", "🔍 Exploración", "🧠 Modelo de ML", "🩺 Predictor de riesgo"]
)

# ============================================================
# TAB 1 — LOS DATOS (interpretación: tipos, rangos, estadísticas)
# ============================================================
with tab_datos:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pacientes", len(df_f))
    c2.metric("Variables", df.shape[1] - 2)
    c3.metric("Con enfermedad", f"{df_f['HeartDisease'].mean():.0%}" if len(df_f) else "—")
    c4.metric("Valores nulos reales", "0 (tras limpieza)")

    st.subheader("¿Cómo están conformados los datos?")
    st.markdown(
        """
Cada fila es **un paciente** y cada columna una **medición clínica**. La variable objetivo es
`HeartDisease` (1 = enfermedad, 0 = sano). Las variables se dividen por tipo, y de ese tipo
depende su limpieza, su codificación y la gráfica adecuada:
        """
    )
    tipos = pd.DataFrame({
        "Variable": list(ETIQUETAS.keys()),
        "Tipo de dato": [
            "Numérica discreta", "Categórica nominal (binaria)", "Categórica nominal",
            "Numérica continua", "Numérica continua", "Categórica binaria (bandera)",
            "Categórica nominal", "Numérica continua", "Categórica nominal (binaria)",
            "Numérica continua", "Categórica ordinal (Up < Flat < Down)",
        ],
        "Rango / categorías": [
            "28 – 77 años", "M (725) · F (193)", "ASY · NAP · ATA · TA",
            "80 – 200 mmHg", "85 – 603 mg/dl", "0 / 1",
            "Normal · LVH · ST", "60 – 202 lpm", "N · Y",
            "-2.6 – 6.2", "Up · Flat · Down",
        ],
    })
    st.dataframe(tipos, width="stretch", hide_index=True)

    st.subheader("Estadísticas descriptivas (con filtros aplicados)")
    if len(df_f):
        st.dataframe(df_f[NUM_COLS].describe().round(2), width="stretch")

    st.subheader("🧹 Limpieza aplicada: nulos disfrazados de cero")
    st.markdown(
        f"""
El dataset original **no tiene `NaN`**, pero sí valores imposibles: **172 pacientes (18.7%)
con `Cholesterol = 0`** y **1 con `RestingBP = 0`**. Un colesterol de cero es fisiológicamente
imposible: son **datos faltantes codificados como cero**. El 88% de esos registros tiene
enfermedad, así que dejarlos crearía un patrón falso que el modelo aprendería.
**Tratamiento:** se convierten a `NaN` y se imputan con la **mediana** (robusta a outliers).
        """
    )
    ver_efecto = st.toggle("Ver el efecto de la limpieza en Cholesterol", value=True)
    if ver_efecto:
        comp = pd.concat([
            df_raw[["Cholesterol"]].assign(Versión="Original (con ceros)"),
            df[["Cholesterol"]].assign(Versión="Limpio (mediana imputada)"),
        ])
        fig = px.histogram(
            comp, x="Cholesterol", color="Versión", barmode="overlay", nbins=60,
            color_discrete_sequence=["#A8A8A8", C_INK],
            title="Distribución de Cholesterol: antes vs. después de la limpieza",
        )
        fig.add_annotation(x=0, y=160, text="172 ceros imposibles", showarrow=True, arrowhead=2)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Histograma porque Cholesterol es numérica continua: muestra la forma de la "
            "distribución y evidencia el pico artificial en cero que desaparece tras imputar."
        )

# ============================================================
# TAB 2 — EXPLORACIÓN (gráficas según tipo de dato)
# ============================================================
with tab_eda:
    if len(df_f) < 10:
        st.warning("Muy pocos pacientes con los filtros actuales. Amplía el filtro en la barra lateral.")
    st.subheader("Explora una variable contra el diagnóstico")
    st.caption(
        "La gráfica se elige automáticamente según el tipo de dato: histograma + caja para "
        "numéricas, barras de proporción para categóricas."
    )

    var = st.selectbox(
        "Variable a explorar",
        NUM_COLS + CAT_NOMINAL + CAT_ORDINAL + BIN_COLS,
        format_func=lambda v: ETIQUETAS[v],
    )

    if var in NUM_COLS:
        col_a, col_b = st.columns(2)
        with col_a:
            fig = px.histogram(
                df_f, x=var, color="Target", barmode="overlay", nbins=40,
                color_discrete_map=PALETA_TARGET,
                title=f"Distribución de {ETIQUETAS[var]}",
            )
            st.plotly_chart(fig, width="stretch")
        with col_b:
            fig = px.box(
                df_f, x="Target", y=var, color="Target",
                color_discrete_map=PALETA_TARGET, points="outliers",
                title=f"{ETIQUETAS[var]} por diagnóstico",
            )
            st.plotly_chart(fig, width="stretch")
        st.caption(
            "**Justificación:** al ser una variable numérica continua, el histograma muestra la "
            "forma y el sesgo de la distribución, y el boxplot compara medianas y outliers entre "
            "sanos y enfermos. Si las cajas se separan, la variable discrimina."
        )
    else:
        prop = (
            df_f.groupby([var, "Target"]).size().reset_index(name="n")
        )
        prop["pct"] = prop["n"] / prop.groupby(var)["n"].transform("sum")
        fig = px.bar(
            prop, x=var, y="pct", color="Target", barmode="stack",
            color_discrete_map=PALETA_TARGET, text_auto=".0%",
            title=f"Tasa de enfermedad por {ETIQUETAS[var]}",
            labels={"pct": "Proporción de pacientes"},
        )
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "**Justificación:** al ser categórica, se usan barras apiladas al 100%: permiten "
            "comparar la tasa de enfermedad entre categorías sin que el tamaño de cada grupo "
            "distorsione la lectura. Un histograma aquí sería incorrecto."
        )

    st.divider()
    col_h, col_s = st.columns(2)
    with col_h:
        st.subheader("Correlación entre variables numéricas")
        corr = df_f[NUM_COLS + ["HeartDisease"]].corr().round(2)
        fig = px.imshow(
            corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
            title="Mapa de calor de correlaciones",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Heatmap porque resume muchas correlaciones numéricas en una sola vista: "
            "Oldpeak (+0.40) y MaxHR (−0.40) son las más asociadas al diagnóstico."
        )
    with col_s:
        st.subheader("Relación edad vs. frecuencia cardiaca máxima")
        fig = px.scatter(
            df_f, x="Age", y="MaxHR", color="Target",
            color_discrete_map=PALETA_TARGET, opacity=0.65,
            trendline="ols" if len(df_f) > 20 else None,
            title="Edad vs. MaxHR por diagnóstico",
            labels=ETIQUETAS,
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Dispersión porque cruza dos numéricas continuas: los pacientes enfermos se "
            "concentran en mayor edad y menor frecuencia cardiaca máxima alcanzada."
        )

    st.success(
        "**Hallazgos clave:** (1) el dolor de pecho **asintomático (ASY)** concentra la mayor tasa "
        "de enfermedad — el síntoma 'silencioso' es el más peligroso; (2) una pendiente ST **Flat/Down** "
        "y un **Oldpeak alto** son fuertes señales de riesgo; (3) a menor **MaxHR** alcanzada, mayor "
        "probabilidad de enfermedad; (4) los hombres presentan una tasa de enfermedad mayor que las mujeres."
    )

# ============================================================
# TAB 3 — MODELO DE ML
# ============================================================
with tab_modelo:
    st.subheader("Definición del problema")
    st.markdown(
        """
**Tipo de problema:** clasificación binaria supervisada — predecir `HeartDisease` (0/1).
**Variables más importantes:** `ST_Slope`, `ChestPainType`, `Oldpeak`, `MaxHR`, `ExerciseAngina`, `Age`.
**Propuesta de valor:** una herramienta de **tamizaje** que prioriza pacientes de alto riesgo para
estudios especializados, usando solo mediciones de bajo costo. No sustituye el diagnóstico médico.

**Métrica prioritaria: Recall (sensibilidad).** En medicina, un falso negativo (decirle "sano" a un
enfermo) cuesta mucho más que un falso positivo. Por eso ambos modelos usan
`class_weight="balanced"` y se evalúan con Recall, F1 y ROC-AUC, no solo con accuracy.
        """
    )

    modelo_sel = st.radio(
        "Modelo a inspeccionar", list(resultados.keys()), horizontal=True
    )
    res = resultados[modelo_sel]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Recall (sensibilidad)", f"{res['recall']:.1%}")
    m2.metric("Precisión", f"{res['precision']:.1%}")
    m3.metric("F1-score", f"{res['f1']:.1%}")
    m4.metric("ROC-AUC", f"{res['auc']:.3f}")

    col_cm, col_roc = st.columns(2)
    with col_cm:
        cm = res["cm"]
        fig = px.imshow(
            cm, text_auto=True, color_continuous_scale="Reds",
            x=["Pred: sano", "Pred: enfermo"], y=["Real: sano", "Real: enfermo"],
            title="Matriz de confusión (conjunto de prueba, 20%)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"De {cm[1].sum()} pacientes realmente enfermos, el modelo detecta {cm[1,1]} "
            f"y deja pasar solo {cm[1,0]} falsos negativos — el error que más nos importa minimizar."
        )
    with col_roc:
        fpr, tpr, _ = res["roc"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=modelo_sel, line=dict(color=C_ENFERMO, width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Azar", line=dict(dash="dash", color="#999")))
        fig.update_layout(
            title=f"Curva ROC — AUC = {res['auc']:.3f}",
            xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos",
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Justificación del modelo e hiperparámetros (sin valores por defecto)")
    if modelo_sel == "Regresión Logística":
        st.markdown(
            """
**¿Por qué?** Es el estándar en problemas médicos por su interpretabilidad: sus coeficientes se
leen como factores de riesgo. Sirve como línea base honesta. Requiere escalado
(`StandardScaler`) y codificación One-Hot, ambos dentro de un `Pipeline` para evitar fuga de datos.

| Hiperparámetro | Valor usado | Default | Por qué |
|---|---|---|---|
| `C` | **0.5** | 1.0 | Más regularización para evitar sobreajuste con 918 muestras |
| `class_weight` | **balanced** | None | Penaliza más los falsos negativos (prioridad clínica) |
| `solver` | **liblinear** | lbfgs | Eficiente en datasets pequeños con L2 |
| `max_iter` | **2000** | 100 | Garantiza convergencia |
            """
        )
    else:
        st.markdown(
            """
**¿Por qué?** Captura relaciones no lineales e interacciones entre variables (p. ej. edad × MaxHR),
es robusto a outliers, no requiere escalado y entrega importancia de variables — clave para
explicar los hallazgos al área médica.

| Hiperparámetro | Valor usado | Default | Por qué |
|---|---|---|---|
| `n_estimators` | **300** | 100 | Más árboles = predicción más estable |
| `max_depth` | **8** | None (ilimitado) | Limita la profundidad para no memorizar el train |
| `min_samples_leaf` | **3** | 1 | Hojas con ≥3 pacientes reducen el sobreajuste |
| `max_features` | **sqrt** | sqrt* | Decorrelaciona los árboles (*fijado explícitamente) |
| `class_weight` | **balanced** | None | Prioriza el recall clínico |
            """
        )
        st.subheader("¿Qué variables pesan más en la predicción?")
        pipe = res["pipe"]
        nombres = pipe.named_steps["pre"].get_feature_names_out()
        imp = pd.DataFrame({
            "Variable": [n.split("__")[1] for n in nombres],
            "Importancia": pipe.named_steps["clf"].feature_importances_,
        }).sort_values("Importancia", ascending=True).tail(10)
        fig = px.bar(
            imp, x="Importancia", y="Variable", orientation="h",
            color_discrete_sequence=[C_INK],
            title="Top 10 variables por importancia (Random Forest)",
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Barras horizontales porque comparan magnitudes entre categorías y permiten leer "
            "los nombres completos. Coincide con el EDA: ST_Slope, Oldpeak y MaxHR dominan."
        )

# ============================================================
# TAB 4 — PREDICTOR DE RIESGO (propuesta de valor interactiva)
# ============================================================
with tab_pred:
    st.subheader("Evalúa el riesgo de un paciente nuevo")
    st.caption(
        "Propuesta de valor: tamizaje inmediato con datos de consulta general. "
        "Resultado orientativo; no sustituye una valoración médica."
    )
    # Opciones en español sencillo -> valor que entiende el modelo
    OPC_SEXO = {"Hombre": "M", "Mujer": "F"}
    OPC_DOLOR = {
        "No siente dolor en el pecho": "ASY",
        "Dolor que no parece del corazón": "NAP",
        "Dolor de pecho leve o poco típico": "ATA",
        "Dolor de pecho clásico (opresión al esforzarse)": "TA",
    }
    OPC_ECG = {
        "Normal": "Normal",
        "Con alteraciones leves": "ST",
        "Con signos de corazón crecido": "LVH",
    }
    OPC_ANGINA = {"No": "N", "Sí": "Y"}
    OPC_PENDIENTE = {
        "Sube (buena señal)": "Up",
        "Se mantiene plana (señal de alerta)": "Flat",
        "Baja (señal de mayor riesgo)": "Down",
    }

    with st.form("form_pred"):
        f1c, f2c, f3c = st.columns(3)
        with f1c:
            p_age = st.slider(
                "Edad", 20, 90, 54,
                help="Edad de la persona en años cumplidos.",
            )
            p_sex = st.selectbox(
                "Sexo", list(OPC_SEXO),
                help="Sexo biológico de la persona. En los datos, los hombres "
                     "presentan una tasa de enfermedad cardiaca más alta.",
            )
            p_cp = st.selectbox(
                "¿Cómo es el dolor de pecho?", list(OPC_DOLOR),
                help="Describe qué siente la persona en el pecho. Dato importante: "
                     "**no sentir dolor NO significa estar sano**; en este estudio, "
                     "la mayoría de los enfermos no tenía dolor (enfermedad 'silenciosa').",
            )
            p_bp = st.slider(
                "Presión arterial en reposo", 80, 200, 130,
                help="Es el número 'alto' cuando te toman la presión, medido en reposo "
                     "(mmHg). Lo normal ronda 120; arriba de 140 se considera presión alta.",
            )
        with f2c:
            p_chol = st.slider(
                "Colesterol en sangre", 85, 600, 240,
                help="Cantidad de colesterol total en la sangre (mg/dl), medida con un "
                     "análisis de laboratorio. Menos de 200 es deseable; arriba de 240 es alto.",
            )
            p_fbs = st.selectbox(
                "¿Azúcar alta en ayunas?", ["No", "Sí"],
                help="Indica si el nivel de azúcar (glucosa) en la sangre, medido sin haber "
                     "comido, supera 120 mg/dl. Un valor alto puede señalar diabetes o prediabetes.",
            )
            p_ecg = st.selectbox(
                "Resultado del electrocardiograma", list(OPC_ECG),
                help="El electrocardiograma (ECG) registra la actividad eléctrica del corazón "
                     "con sensores en el pecho. Aquí va el resultado que reportó el médico: "
                     "normal, con alteraciones leves, o con signos de que el corazón ha crecido "
                     "por trabajar de más.",
            )
            p_hr = st.slider(
                "Pulso máximo alcanzado en ejercicio", 60, 210, 140,
                help="Las pulsaciones por minuto más altas que alcanzó la persona durante una "
                     "prueba de esfuerzo (caminar/correr en una banda). Un corazón sano suele "
                     "alcanzar pulsos más altos; como referencia, el máximo teórico es 220 menos la edad.",
            )
        with f3c:
            p_ang = st.selectbox(
                "¿Dolor de pecho al hacer ejercicio?", list(OPC_ANGINA),
                help="Indica si a la persona le duele u oprime el pecho cuando hace esfuerzo "
                     "físico (subir escaleras, caminar rápido). Ese dolor se llama 'angina' y "
                     "sugiere que al corazón le falta oxígeno cuando trabaja más.",
            )
            p_old = st.slider(
                "Descenso en el electrocardiograma con esfuerzo", -2.0, 6.5, 1.0, 0.1,
                help="Durante la prueba de esfuerzo, el médico observa si una parte de la señal "
                     "del electrocardiograma 'baja' respecto al reposo (se llama depresión del "
                     "segmento ST u 'Oldpeak'). Entre más grande el número, más señal de que el "
                     "corazón sufre con el esfuerzo. 0 es lo normal.",
            )
            p_slope = st.selectbox(
                "Comportamiento de la señal del corazón al esforzarse", list(OPC_PENDIENTE),
                help="En la prueba de esfuerzo, la señal del electrocardiograma puede subir, "
                     "mantenerse plana o bajar. Que suba es lo esperado en un corazón sano; "
                     "que se quede plana o baje es una de las señales de riesgo más fuertes "
                     "de todo este estudio.",
            )
            modelo_pred = st.selectbox(
                "Modelo de predicción a usar", list(resultados.keys()), index=1,
                help="Algoritmo de machine learning que hará el cálculo. Random Forest es el "
                     "más preciso; Regresión Logística es el más fácil de interpretar.",
            )
        enviado = st.form_submit_button("🫀 Calcular riesgo", width="stretch")

    if enviado:
        paciente = pd.DataFrame([{
            "Age": p_age, "Sex": OPC_SEXO[p_sex], "ChestPainType": OPC_DOLOR[p_cp],
            "RestingBP": p_bp, "Cholesterol": p_chol,
            "FastingBS": 1 if p_fbs == "Sí" else 0,
            "RestingECG": OPC_ECG[p_ecg], "MaxHR": p_hr,
            "ExerciseAngina": OPC_ANGINA[p_ang],
            "Oldpeak": p_old, "ST_Slope": OPC_PENDIENTE[p_slope],
        }])
        proba = resultados[modelo_pred]["pipe"].predict_proba(paciente)[0, 1]

        col_g, col_t = st.columns([1, 1])
        with col_g:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={"suffix": "%"},
                title={"text": "Probabilidad de enfermedad cardiaca"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": C_INK},
                    "steps": [
                        {"range": [0, 30], "color": "#D8F3EE"},
                        {"range": [30, 60], "color": "#FFE8CC"},
                        {"range": [60, 100], "color": "#FFD2D5"},
                    ],
                },
            ))
            fig.update_layout(height=320, margin=dict(t=60, b=10))
            st.plotly_chart(fig, width="stretch")
        with col_t:
            if proba >= 0.6:
                st.error(
                    f"**Riesgo alto ({proba:.0%}).** Recomendación: canalizar a cardiología "
                    "para estudios confirmatorios (prueba de esfuerzo, ecocardiograma)."
                )
            elif proba >= 0.3:
                st.warning(
                    f"**Riesgo moderado ({proba:.0%}).** Recomendación: seguimiento clínico, "
                    "control de factores de riesgo y reevaluación."
                )
            else:
                st.success(
                    f"**Riesgo bajo ({proba:.0%}).** Recomendación: mantener hábitos saludables "
                    "y chequeos periódicos."
                )
            st.caption(
                f"Modelo usado: {modelo_pred} · Recall {resultados[modelo_pred]['recall']:.0%} · "
                f"ROC-AUC {resultados[modelo_pred]['auc']:.3f}"
            )

st.divider()
st.caption(
    "Proyecto de Business Intelligence y Ciencia de Datos · Dataset: Heart Failure Prediction "
    "(fedesoriano, Kaggle) · Modelos: scikit-learn · Visualización: Plotly + Streamlit"
)
