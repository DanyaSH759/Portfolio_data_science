# Описание задачи: 
Сетевой супермаркет «Хлеб-Соль» внедряет систему компьютерного зрения для обработки фотографий покупателей. Постройте модель, которая по фотографии определит приблизительный возраст человека. В вашем распоряжении набор фотографий людей с указанием возраста.

# Описание данных: 
В датасете имеется путь к фотографии и целевой признак - возраст.

# Общий итог: 
В ходе исследования была изучена информация о имеющихся данных для обучения. Выбран оптимальный размер картинок (путем вычисления среднего). За основу модели была взята сеть ResNet50. Лучшая метрика MAE на тестовой выборке составляет 6.0641. Данная метрика было достигнута выставлением шага убывание градинта = 0.0001, и 10 эпох обучения. 
