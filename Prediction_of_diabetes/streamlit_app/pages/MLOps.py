import streamlit as st
from set_params import load_page_setting
from base_funct import transform_df_base, DataTransformer
import pandas as pd
from catboost import CatBoostClassifier
import os

st.set_page_config(
    page_title="MLOps",
    page_icon="🔌"
)

# загрузка параметров фона стилей и т.д.
load_page_setting()


st.sidebar.header("Тестирование лучшей модели")
st.sidebar.success("Финальное тестирвоание лучшей модели")
st.sidebar.success("Исследование лучшей модели")

st.markdown("""
# Деплой модели
            
Модель на вход получает excel файл, после чего в зависимости от полученных результатов выдаст вероятность есть ли у пациента диабет или нет

""")

# Обучение модели
df, test = transform_df_base()
transformer_class = DataTransformer()
transformer_class.fit(df, columns_to_exclude=['Натрий', 'Калий', 'Мочевина', 'Общий билирубин'])
transform_test = transformer_class.transform(test)

df = pd.concat([df, transform_test])

x_train_all = df.drop('Сахарный диабет (ЕСТЬ\НЕТ)', axis = 1)
y_train_all = df['Сахарный диабет (ЕСТЬ\НЕТ)']

# лучшие параметры catboost
best_params = {'iterations': 437,
 'depth': 2,
 'learning_rate': 0.059429953553017466,
 'l2_leaf_reg': 2.0210035460360274}

model_best = CatBoostClassifier(**best_params, verbose=0)
model_best.fit(x_train_all, y_train_all)

# деплой модели

col1, col2 = st.columns([1,1])

col1.markdown(
"""
Вот образец для заполнения, скачайте и заполните файл, после чего отправте нам в поле ниже
"""
)

file_path = './streamlit_app/data/Шаблон для заполнения.xlsx'

# Функция для чтения загруженного файла
def load_file(uploaded_file):
    if uploaded_file is not None:
        try:
            # Прочитать загруженный файл в DataFrame
            data_d = pd.read_excel(uploaded_file)
            st.write("Файл успешно загружен!")
            return data_d
        except Exception as e:
            st.error(f"Ошибка при чтении файла: {e}")


# Функция для скачивания существующего файла
def download_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            st.download_button(
                label="Скачать существующий файл",
                data=f,
                file_name=os.path.basename(file_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    except Exception as e:
        st.error(f"Ошибка при скачивании файла: {e}")


# Секция для скачивания файла
st.subheader('Скачать образец для заполнения Excel')
download_file(file_path)

# Секция для загрузки файла
st.subheader('Загрузите файл Excel')
uploaded_file = st.file_uploader("Выберите файл Excel", type=["xlsx"])

data_r = load_file(uploaded_file)

try:
    if uploaded_file is not None:
        st.markdown("""Вероятность сахарного диабета у пациента""")
        st.text(model_best.predict_proba(pd.DataFrame(data_r))[0][1])
except:
    pass





