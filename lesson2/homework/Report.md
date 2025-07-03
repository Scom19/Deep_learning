## Итоговый результат
1. Обучены модели:
   * линейная регрессия на `spotify.csv` – веса в `homework/models/spotify.csv_lin.pt`;
   * softmax-регрессия на `Titanic.csv` – веса в `homework/models/Titanic.csv_log.pt`.
2. Сохранены графики обучения, подборов гиперпараметров, сравнения признаков
   в `homework/plots/`.
3. CSV-таблицы с результатами grid-search:
   * `hyperparam_results.csv` (регрессия);
   * `hyperparam_results_cls.csv` (классификация).
4. Матрица ошибок для Titanic – `titanic_cm.png`.
5. Весь код покрыт логированием.