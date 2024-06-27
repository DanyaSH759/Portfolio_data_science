import streamlit as st
from set_params import load_page_setting
import pandas as pd
from io import StringIO

st.set_page_config(
    page_title="Load data",
    page_icon="📈"
)

# загрузка параметров фона стилей и т.д.
load_page_setting()

st.markdown("# Загрузка данных")

st.sidebar.header("Загрузка данных")
st.sidebar.success("Стэк для подбора модели")
st.sidebar.success("Загрузка датасета")

st.markdown(
"""
### Наш стэк в данном проекте для разработки модели:
"""
)

with st.expander("Библиотеки"):
    st.code("""# загрузка библиотек
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from phik.report import plot_correlation_matrix

from sklearn.model_selection import (ShuffleSplit, train_test_split, cross_val_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, confusion_matrix)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from catboost import CatBoostClassifier

import ipywidgets as widgets
from IPython.display import display, clear_output

from mlxtend.plotting import plot_decision_regions
import shap
from lime import lime_tabular

import warnings
warnings.filterwarnings('ignore')""", line_numbers = True)


st.markdown(
"""
### Загружаем данные
"""
)

st.markdown(
"""
Изначальный датасет имеет большое количество пропусков и колонок.
Для перехода к EDA необходимо подготовить данные.
"""
)

# грузим данные
df = pd.read_excel('./diabetes_ds.xlsx', index_col=0)

# Создаем буфер для записи вывода df.info()
buffer = StringIO()
df.info(buf=buffer, verbose = True, show_counts = True)

# Преобразуем буфер в строку и выводим с помощью st.text()
info_str = buffer.getvalue()

with st.expander("Просмотр первичного датасета"):
    st.text(info_str)
    st.dataframe(df.head())
    st.dataframe(df.tail())

st.markdown(
"""
### Корректируем ошибки при загрузке данных и отделяем и делим выборку на тестовую и тренировочную.
"""
)

with st.expander("Деление данных на трейн и тест"):
    st.code(
        """
# удалим ошибочно отображенные строчки при загрузке датасте
data = data.iloc[:-2]

# Удалим сразу строчку с пустым таргетом
data.dropna(subset=['Сахарный диабет (ЕСТЬ\НЕТ)'], inplace=True)

# Отделим наши тренировочные данные от теста. Тест отложим до финального тестирования
data, test = train_test_split(data, test_size = 0.30, shuffle = True, random_state = RANDOM_STATE)

"""
    )

st.markdown(
"""
### Выводы

Загружены библиотеки и датасет. В датасете обнаружено большое кол-во пропусков, сам датасет состоит из 107 колонок и 484 строчек. От датасета была отделена часть для проведения тестирования.

"""
)
