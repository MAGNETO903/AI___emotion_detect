# импорт
import numpy as np  # по идее он не нужен
import pandas as pd  # для работы с таблицами

from sklearn.ensemble import RandomForestClassifier  # Случайный лес
from sklearn.feature_extraction.text import CountVectorizer  # Текст в вектор
from sklearn.metrics import accuracy_score, f1_score  # метрики качества

# разделение выборки на train и test
from sklearn.model_selection import train_test_split, GridSearchCV

# вектор для обработки
vec = CountVectorizer(max_features=100,
                      ngram_range=(2, 2),
                      analyzer='word')

# чтение информации
data = pd.read_csv('new_data.csv')

# принимаем текст, предсказываем эмоциональную окраску
X = data['ttext']
y = data['sentiment']

# преобразование текста в вектор для обработки
X_transformed = vec.fit_transform(X)

# разделение на train и на test
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.1, random_state=42)

# создание дерева
forest = RandomForestClassifier(n_estimators=10, max_depth=5)

# перебираемые параметры
params = {'n_estimators': [5, 10], 'max_depth': [5, 10]}

# создание кросс-валидации
grid_search = GridSearchCV(forest, params, cv=10)

# выполнение кросс-валидации
grid_search.fit(X_train, y_train)

# вывод точности, самой точной модели
print(grid_search.best_score_)

# вывод параметров наилучшей модели
print(grid_search.best_params_)

# сохранение наилучшей модели
best_forest = grid_search.best_estimator_

# предсказывание наилучшей модели
y_predicted = best_forest.predict(X_test)

# итоги
result_1 = accuracy_score(y_test, y_predicted)
result_2 = f1_score(y_test, y_predicted)

print(result_1)
print(result_2)
