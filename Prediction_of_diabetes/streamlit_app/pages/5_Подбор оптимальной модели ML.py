import streamlit as st
from set_params import load_page_setting

st.set_page_config(
    page_title="ML search",
    page_icon="📟"
)

# загрузка параметров фона стилей и т.д.
load_page_setting()

st.sidebar.header("ML search")
st.sidebar.success("Тюнинг моделей и выбор лучшей по метрике")


st.markdown("""
# Подбор оптимальной модели ML

Начнем с LogisticRegression. Сделаем 2 обучения - без кодирования данных и с кодированием.
""")
with st.expander("LogisticRegression"):
    st.code("""
    model_logreg = LogisticRegression(random_state=RANDOM_STATE)

    params_logreg = {
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'class_weight' : ['balanced', None],
        'solver': ['liblinear', 'sag', 'saga'],
        'max_iter': [x for x in range(20, 201, 20)],
        'l1_ratio': [0, 0.5, 1, None]
    }

    cv = ShuffleSplit(n_splits=3, test_size=0.10, random_state=RANDOM_STATE)

    gs_logreg = GridSearchCV(
        model_logreg,
        params_logreg,
        n_jobs=-1,
        cv=cv,
        scoring="f1_macro",
        verbose = 1,
        return_train_score = True

    )
            
    %%time
    # данные не кодированы
    gs_logreg.fit(x_train, y_train)
            
    predict = gs_logreg.best_estimator_.predict(x_val)
    print(f'f1 score LogisticRegression = {f1_score(y_val, predict, average = "macro")}')
    print(f'лучшие параметры = {gs_logreg.best_params_}')
    print(f'лучшая метрика на кроссвалдиации = {gs_logreg.best_score_}')
            
    f1 score LogisticRegression = 0.5733333333333333
    лучшие параметры = {'class_weight': 'balanced', 'l1_ratio': 0, 'max_iter': 20, 'penalty': 'l2', 'solver': 'liblinear'}
    лучшая метрика на кроссвалдиации = 0.6324795693082138

    %%time
    # данные кодированы
    gs_logreg.fit(pipeline_transform.transform(x_train), y_train)

    predict = gs_logreg.best_estimator_.predict(pipeline_transform.transform(x_val))
    print(f'f1 score LogisticRegression данные кодлированы= {f1_score(y_val, predict, average = "macro")}')
    print(f'лучшие параметры = {gs_logreg.best_params_}')
    print(f'лучшая метрика на кроссвалдиации = {gs_logreg.best_score_}')
            
    f1 score LogisticRegression данные кодлированы= 0.5465587044534412
    лучшие параметры = {'class_weight': None, 'l1_ratio': 0, 'max_iter': 20, 'penalty': 'l1', 'solver': 'saga'}
    лучшая метрика на кроссвалдиации = 0.636005530417295


    """, line_numbers = True)

st.markdown("""
    Теперь возьмем KNeighborsClassifier
""")

with st.expander("KNeighborsClassifier"):
    st.code("""
    model_knn =  KNeighborsClassifier()

    params_knn = {
        'n_neighbors': [x for x in range(5, 46, 5)],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'p': [1, 2]
    }

    cv = ShuffleSplit(n_splits=3, test_size=0.10, random_state=RANDOM_STATE)

    gs_knn = GridSearchCV(
        model_knn,
        params_knn,
        n_jobs=-1,
        cv=cv,
        scoring="f1_macro",
        verbose = 1,
        return_train_score = True
    )
    %%time
    # данные не кодированы
    gs_knn.fit(x_train, y_train)  

    predict = gs_knn.best_estimator_.predict(x_val)
    print(f'f1 score KNeighborsClassifier = {f1_score(y_val, predict, average = "macro")}')
    print(f'лучшие параметры = {gs_knn.best_params_}')
    print(f'лучшая метрика на кроссвалдиации = {gs_knn.best_score_}')

    f1 score KNeighborsClassifier = 0.3552492046659597
    лучшие параметры = {'algorithm': 'auto', 'n_neighbors': 25, 'p': 1}
    лучшая метрика на кроссвалдиации = 0.5369221578898998
            
    %%time
    # данные кодированы

    gs_knn.fit(pipeline_transform.transform(x_train), y_train)

            
    predict = gs_knn.best_estimator_.predict(pipeline_transform.transform(x_val))
    print(f'f1 score KNeighborsClassifier данные кодированы = {f1_score(y_val, predict, average = "macro")}')
    print(f'лучшие параметры = {gs_knn.best_params_}')
    print(f'лучшая метрика на кроссвалдиации = {gs_knn.best_score_}')
            
    f1 score KNeighborsClassifier данные кодированы = 0.7408906882591093
    лучшие параметры = {'algorithm': 'auto', 'n_neighbors': 15, 'p': 1}
    лучшая метрика на кроссвалдиации = 0.6650736758263639
""",
line_numbers = True)
    
st.markdown("""
    Переходим к бустингам. Для начала LGBMClassifie
""")

with st.expander("LGBMClassifier"):
    st.code("""
    model_lgbm = LGBMClassifier(random_state=RANDOM_STATE)

params_lgbm = {
    'n_estimators': [x for x in range(40, 80, 20)],
    'learning_rate': [0.1, 0.01],
}

cv = ShuffleSplit(n_splits=3, test_size=0.10, random_state=RANDOM_STATE)

gs_lgbm = GridSearchCV(
    model_lgbm,
    params_lgbm,
    n_jobs=-1,
    cv=cv,
    scoring="f1_macro",
    verbose = 0,
    return_train_score = True
)
            
    %%time
    # данные не кодированы
    gs_lgbm.fit(x_train.values, y_train.values)
                

    predict = gs_lgbm.best_estimator_.predict(x_val)
    print(f'f1 score LGBMClassifier = {f1_score(y_val, predict, average = "macro")}')
    print(f'лучшие параметры = {gs_lgbm.best_params_}')
    print(f'лучшая метрика на кроссвалдиации = {gs_lgbm.best_score_}')
                

    f1 score LGBMClassifier = 0.716256157635468
    лучшие параметры = {'learning_rate': 0.1, 'n_estimators': 60}
    лучшая метрика на кроссвалдиации = 0.722039072039072
            

    %%time
    # данные кодированы
    gs_lgbm.fit(pipeline_transform.transform(x_train), y_train.values)

    predict = gs_lgbm.best_estimator_.predict(pipeline_transform.transform(x_val))
    print(f'f1 score LGBMClassifier данные кодированы = {f1_score(y_val, predict, average = "macro")}')
    print(f'лучшие параметры = {gs_lgbm.best_params_}')
    print(f'лучшая метрика на кроссвалдиации = {gs_lgbm.best_score_}')
                
    f1 score LGBMClassifier данные кодированы = 0.6761133603238867
    лучшие параметры = {'learning_rate': 0.1, 'n_estimators': 40}
    лучшая метрика на кроссвалдиации = 0.709807055850808

""", line_numbers = True)
    
st.markdown("""
    Сделаем локальный эксперемент для catboost, даст ли эффект кодирование датасета
""")

with st.expander("Локальный эксперемент 1"):
    st.code("""
%%time
# результат базаовой модели catboost

# сделаем копию датасета
data_base = data_eda_1v1_drop.copy()

x_train_all = data_base.drop('Сахарный диабет (ЕСТЬ\НЕТ)', axis = 1)
y_train_all = data_base['Сахарный диабет (ЕСТЬ\НЕТ)']

x_train, x_val, y_train, y_val = train_test_split(x_train_all, y_train_all, test_size = 0.10, shuffle = True, random_state = RANDOM_STATE)

model = CatBoostClassifier(random_state = RANDOM_STATE, verbose = 0)
model.fit(x_train, y_train)
predict = model.predict(x_val)
print(f'f1 score у baseline модели (pipeline encoder)= {f1_score(y_val, predict, average = "macro")}')
f1 score у baseline модели (pipeline encoder)= 0.7117117117117118
""",
line_numbers = True)


with st.expander("Локальный эксперемент 2"):
    st.code("""
%%time
# результат базаовой модели catboost - pipeline encoder

x_all = data_base.drop('Сахарный диабет (ЕСТЬ\НЕТ)', axis = 1)
y_all = data_base['Сахарный диабет (ЕСТЬ\НЕТ)']

x_train, x_val, y_train, y_val = train_test_split(x_all, y_all, test_size = 0.10, shuffle = True, random_state = RANDOM_STATE)

model = CatBoostClassifier(random_state = RANDOM_STATE, verbose = 0)
model.fit(pipeline_transform.transform(x_train), y_train)
predict = model.predict(pipeline_transform.transform(x_val))
print(f'f1 score у baseline модели (pipeline encoder)= {f1_score(y_val, predict, average = "macro")}')
f1 score у baseline модели (pipeline encoder)= 0.6389743589743591
""", line_numbers = True)


st.markdown("""
Эксперемент показал, что catboost лучше всего применять без кодирования данных
            
Для тюнинга CatBoostClassifier и RandomForestClassifier, и посика лучших параметров воспользовались бибиоиотекой optuna
""")

with st.expander("Поиск лучших параметров для CatBoostClassifier"):
    st.code("""# поиск лучших параметров для CatBoostClassifier

    def CatBoostClassifier_p1(trial):
        '''функция для перебора параметров для optuna'''

        # параметры
        iterations = trial.suggest_int("iterations", 5, 500)
        depth = trial.suggest_int("depth", 1, 16)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1)
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', 2, 3.5)

        print('Выбранные параметры: ', iterations, depth, learning_rate, l2_leaf_reg)

        pipeline = Pipeline(steps=[
        ('CatBoost', CatBoostClassifier(iterations = iterations,
                            depth = depth,
                            learning_rate = learning_rate,
                            l2_leaf_reg = l2_leaf_reg,
                            random_state = RANDOM_STATE,
                            verbose = 0))
        ])


        cv = ShuffleSplit(n_splits=3, test_size=0.10, random_state=RANDOM_STATE)

        start = time.time()

        scores = cross_val_score(pipeline, x_train_all, y_train_all, scoring = 'f1_macro',  cv=cv, verbose = 1)

        end = time.time()

        print(f'Время обучения составило = {round(end - start, 2)} секунд')
        print()

        return scores.mean()
     
    %%time
    study_CB = optuna.create_study(direction="maximize")
    study_CB.optimize(CatBoostClassifier_p1, n_trials=20)
    study_CB.best_params
            
    # Результаты лучшей модели
    Лучшие параметры для CatBoost
    {'iterations': 437,
    'depth': 2,
    'learning_rate': 0.059429953553017466,
    'l2_leaf_reg': 2.0210035460360274}   Best is trial 8 with value: 0.7208769444063563.""", line_numbers = True)

with st.expander("Поиск лучших параметров для RandomForestClassifier"):

    st.code("""# поиск лучших параметров для RandomForestClassifier

    def objective_RF(trial):
        '''функция для перебора параметров для optuna'''
        # параметры
        n_estimators = trial.suggest_int("n_estimators", 10, 180)
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        max_depth = trial.suggest_int("max_depth", 8, 25)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        min_samples_split = trial.suggest_categorical('min_samples_split', [2, 3, 4])
        min_samples_leaf = trial.suggest_categorical('min_samples_leaf', [1, 2, 3])
        class_weight = trial.suggest_categorical("class_weight", ["balanced", None])


        pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('RandomForestClassifier', RandomForestClassifier(n_estimators  = n_estimators ,
                                criterion = criterion ,
                                    max_depth= max_depth,
                                    max_features = max_features,
                                    min_samples_split = min_samples_split,
                                    min_samples_leaf = min_samples_leaf,
                                    class_weight = class_weight,
                                    random_state = RANDOM_STATE, n_jobs = -1))
        ])


        # через кросс-валидацию смотрим на результат выбранных параметров
        cv = ShuffleSplit(n_splits=3, test_size=0.1, random_state=RANDOM_STATE)

        start = time.time()

        cores = cross_val_score(pipeline, x_train_all, y_train_all, scoring = 'f1_macro',  cv=cv, verbose = 1)

        end = time.time()

        print(f'Время обучения составило = {round(end - start, 2)} секунд')
        print()

        return scores.mean()

    # %%time
    study_RF_p = optuna.create_study(direction="maximize")
    study_RF_p.optimize(objective_RF, n_trials=100)
    study_RF_p.best_params

    # Результаты лучшей модели
    {'n_estimators': 88,
    'criterion': 'gini',
    'max_depth': 14,
    'max_features': 'log2',
    'min_samples_split': 3,
    'min_samples_leaf': 2,
    'class_weight': None} Best is trial 61 with value: 0.7156821672950705.""", line_numbers = True)

st.markdown(
"""
### Выводы
Путем перебора параметров и сравнении метрики на валидационной выборке, лучшей моделью оказался Catboost c метрикой f1 macro 0.72.
"""
)