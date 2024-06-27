import streamlit as st
from set_params import load_page_setting
from PIL import Image


st.set_page_config(
    page_title="ML testing",
    page_icon="📡"
)

# загрузка параметров фона стилей и т.д.
load_page_setting()


st.sidebar.header("Тестирование лучшей модели")
st.sidebar.success("Финальное тестирвоание лучшей модели")
st.sidebar.success("Исследование лучшей модели")

st.markdown("""
# Тестирование лучшей модели

Для финального обучения была взята полная тренировочная выобрка. Финальное тестирование провоидтся на тестовой выборке
""")


st.code("""
# сделаем копию датасета
data_base = data_eda_1v1_drop.copy()

x_train_all = data_base.drop('Сахарный диабет (ЕСТЬ\НЕТ)', axis = 1)
y_train_all = data_base['Сахарный диабет (ЕСТЬ\НЕТ)']

x_test = transform_test.drop('Сахарный диабет (ЕСТЬ\НЕТ)', axis = 1)
y_test = transform_test['Сахарный диабет (ЕСТЬ\НЕТ)']

# лучшие параметры catboost
best_params = {'iterations': 437,
 'depth': 2,
 'learning_rate': 0.059429953553017466,
 'l2_leaf_reg': 2.0210035460360274}

%%time
# результат базаовой модели catboost - pipeline encoder

model_best = CatBoostClassifier(**best_params, verbose=0)

model_best.fit(x_train_all, y_train_all)

predict_train = model_best.predict(x_train_all)
print(f'f1 score у модели на трейне = {f1_score(y_train_all, predict_train, average = "macro")}')

print('f1 score модели при обучении на кросс-валидации = 0.721.')

predict_test = model_best.predict(x_test)
print(f'f1 score у модели на тесте = {f1_score(y_test, predict_test, average = "macro")}')
       
        
f1 score у модели на трейне = 0.9872912127814089
f1 score модели при обучении на кросс-валидации = 0.721.
f1 score у модели на тесте = 0.7253386804910256
CPU times: user 1.14 s, sys: 630 ms, total: 1.77 s
Wall time: 207 ms""", line_numbers = True)


st.markdown("""
Матрица ошибок нашей модели:
"""
)

img = Image.open('./streamlit_app/data/conf_matrix.png')
st.image(img)

st.markdown("""
    Ошибки распределенеы практически равномерно, но модель тяготеет к ложному предсказыванию диабета у пациена, из 39 ошибочных предсказаний к этому типу ошибки относится 23.
""")

st.markdown(
"""
Полученные веса модели
"""
)

img = Image.open('./streamlit_app/data/ves1.png')
st.image(img)

st.markdown("""
    Самый важный признак для модели - Глюкоза. Что в целом не удивительно, т.к. это один из основных анализов, на котоырй смотрят при определении сахарного диабета
""")

st.markdown(
"""
Веса модели исследованы методом shap
"""
)

img = Image.open('./streamlit_app/data/ves2.png')
st.image(img)

st.markdown("""
    По весам полученным их shap видно что есть ещё много локальных вбросов по значениям, который были оставлены ранее при обработке данных.
    Так же видно что не все веса имеют четкое распределение между таргетами и часто пересекаются друг с другом.
""")

st.markdown(
"""
Веса модели исследованы методом lime
"""
)

img = Image.open('./streamlit_app/data/lime.png')
st.image(img)

st.markdown("""
Для примера взяли 128 строчку из трейна для анализа весов, тут немного картина отличается от весов shap, но суть остается такой же, по прежнему важными признаками остается Глюкоза ИМТ и т.д.
""")

st.markdown(
"""
Исследована разделяющая способность дерева
"""
)

img = Image.open('./streamlit_app/data/pca_tree.png')
st.image(img)

st.markdown("""
Разделяющая поверхность дерева показала нам, что данные трудно разделими и модель скорее заучивает ответы.
""")

# st.markdown(
#     "<span style='font-weight: bold;'> <span style='font-size: 20px;'>Полученные веса модели</span></span>", 
#     unsafe_allow_html=True
# )


st.markdown(
"""<div style="border: 3px solid black; border-radius: 10px; padding: 10px;">

## ИТОГ

**Выполнено:**
- Загружены и подготолены данные
- Обработан и подготовлен датасет для подбора моделей
- Сформирована baseline модель
- Подобрана оптимальаня модель обучения
- Проведено финальное тестирование лучшей модели
- Исследован результат лучшей модели


В ходе исследования была выявлена лучшая модель -  CatBoostClassifier. Метрика на трейне при кросс-валидации составила 0.721, на тесте составила 0.725. Модель немного недообучилась, скорее всего ввиду малого кол-ва данных.

В ходе разработки моделей было сокращено большое кол-во колонок из первоначального датасета, на текущий момент модель принимает на вход датасет из 34 колонок (изначальный датасет состоит из 107). В целом данное кол-во можно ещё сильнее сократить при дальнейшем тюнинге модели.

В ходе подбора модели выбранная нами лишь немного превышает метрику, полученную при baseline, вероятно это из за размерности датасета.

В лучшей модели были определены самые важные признаки (топ 5): Глюкоза, Холестерин, ЛПНП, Гемоглобин, Индекс атерогенности 2.

Для презентации нашей модели, был разработан сервис streamlit.
</div>""", 
    unsafe_allow_html=True
)






