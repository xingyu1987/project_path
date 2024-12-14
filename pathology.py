import streamlit as st
from joblib import load
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np


st.set_page_config(
        page_title="Pathology prediction of lung cancer",
        page_icon="🧊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
        }
    )


st.title('This is a prediction APP for subtype of Lung Cancer')

st.sidebar.header("Features Input")


def user_input_features():
    form = st.sidebar.form("heloo")
    age = form.number_input("Age", min_value=10, max_value=100, step=1, value=20)
    smoking = form.selectbox("Smoking", ('No', 'Yes'))
    Alcohol_Frequency = form.selectbox("Daily Alcohol Consumption",("Never", "Occasionally", "Regular", "Daily"))


    Hb = form.number_input("Hemoglobin", min_value=0, max_value=200, step=1, value=120)
    UA = form.number_input('Uric Acid', min_value=0, max_value=999, step=1, value=50)
    Sodium = form.number_input('Sodium', min_value=0, max_value=999, step=1, value=130)
    Fib = form.number_input('Fibrinogen', min_value=0, max_value=30, step=1, value=1)
    CEA = form.number_input('Carcinoembryonic Antigen', min_value=0, max_value=999, step=1, value=1)
    NSCL = form.number_input('Non Small Cell Lung Cancer Antigen', min_value=0, max_value=999, step=1, value=1)
    NSE = form.number_input('Neuron Specific Enolase', min_value=0, max_value=999, step=1, value=1)

    if smoking == "No":
        smoking = 0
    elif smoking == "Yes":
        smoking = 1
    if Alcohol_Frequency == "Never":
        Alcohol_Frequency = 0
    elif Alcohol_Frequency == "Occasionally":
        Alcohol_Frequency = 1
    elif Alcohol_Frequency == "Regular":
        Alcohol_Frequency = 2
    elif Alcohol_Frequency == "Daily":
        Alcohol_Frequency = 3

    data = {
        "Age": age,
        "Smoking": smoking,
        'Alcohol Intake Frequency': Alcohol_Frequency,
        'Hemoglobin': Hb,
        'Uric Acid': UA,
        'Sodium': Sodium,
        'Fibrinogen': Fib,
        'Carcinoembryonic Antigen': CEA,
        'Non Small Cell Lung Cancer Antigen': NSCL,
        'Neuron Specific Enolase': NSE
    }

    features = pd.DataFrame(data, index=[0])
    features = features.astype("float")

    form.form_submit_button("Submit for Prediction")

    return features


features = user_input_features()


path = r"E:\Study\余明静\project_path\lightGBM.joblib"

model = load(path)





prediction_Adc = round(model.predict_proba(features)[0][0], 5)
prediction_scc = round(model.predict_proba(features)[0][1], 5)
prediction_scclc = round(model.predict_proba(features)[0][2], 5)

st.error('The probability of Adenocarcinoma is about {}'.format(prediction_Adc))
st.warning('The probability of Squamous Cell Carcinoma is about {}'.format(prediction_scc))
st.info('The probability of Small Cell Carcinoma is about {}'.format(prediction_scclc))

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(features)

# 处理shap_values格式
if isinstance(shap_values, np.ndarray):
    if shap_values.shape[-1] == len(explainer.expected_value):  # 如果最后一个维度匹配类别数
        # 保持2D维度，每个类别一个数组
        shap_values = [shap_values[0, :, i:i+1] for i in range(shap_values.shape[-1])]

fig, ax = plt.subplots()
fig = shap.multioutput_decision_plot(
    explainer.expected_value,
    shap_values,
    row_index=0,
    feature_names=features.columns.to_list(),
    legend_labels=["Adenocarcinoma", "Squamous Cell Carcinoma", "Small Cell Carcinoma"],
    legend_location='lower right',
    link='logit',
    )
st.set_option('deprecation.showPyplotGlobalUse', False)
st.pyplot(fig)
