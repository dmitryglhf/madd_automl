Таблица из статьи:
| Case                 | Model          | Accuracy | F1 score |
|----------------------|----------------|----------|----------|
|                      | **MADDAutoML** | **0.951** | **0.914** |
| Alzheimer's disease  | CatBoost       | 0.810    | 0.810    |
|                      | Random Forest  | 0.822    | 0.829    |
|                      | XGBoost        | 0.803    | 0.803    |
|                      | **Extra Trees**| **0.823**| **0.829**|
|                      | LightGBM       | 0.810    | 0.820    |
| Mupltiple sclerosis  | CatBoost       | 0.865    | 0.905    |
|                      | MADDAutoML     | 0.877    | 0.913    |
|                      | **Random Forest**| **0.887**| **0.920**|
|                      | XGBoost        | 0.876    | 0.912    |
|                      | Extra Trees    | 0.886    | 0.919    |
|                      | LightGBM       | 0.885    | 0.918    |
| Parkinson's disease  | **CatBoost**   | **0.910**| **0.920**|
|                      | Random Forest  | 0.890    | 0.900    |
|                      | XGBoost        | 0.910    | 0.910    |
|                      | Extra Trees    | 0.890    | 0.900    |
|                      | LightGBM       | 0.900    | 0.910    |
| Dyslipidemia         | **CatBoost**   | **0.778**| **0.818**|
|                      | Random Forest  | 0.775    | 0.813    |
|                      | XGBoost        | 0.725    | 0.772    |
|                      | Extra Trees    | 0.750    | 0.782    |
|                      | LightGBM       | 0.725    | 0.775    |
| Drug resistance      | **CatBoost**   | **0.838**| **0.848**|
|                      | Random Forest  | 0.828    | 0.836    |
|                      | XGBoost        | 0.823    | 0.826    |
|                      | Extra Trees    | 0.837    | 0.844    |
|                      | LightGBM       | 0.832    | 0.835    |
| Lung cancer          | CatBoost       | 0.822    | 0.822    |
|                      | Random Forest  | 0.838    | 0.835    |
|                      | XGBoost        | 0.827    | 0.826    |
|                      | **Extra Trees**| **0.843**| **0.842**|
|                      | LightGBM       | 0.811    | 0.809    |


Наша таблица:

| Датасет                 | Модель                                        | Accuracy (1024) | F1-score (1024) | Accuracy (2048/512) | F1-score (2048/512) |
| ----------------------- | --------------------------------------------- | --------------- | --------------- | ------------------- | ------------------- |
| **Alzheimer's disease** | MADDAutoML(blending\_no\_tuner\_not\_compose) | 0.9544          | 0.9733          | 0.9578              | 0.9753              |
|                         | MADDAutoML(clear\_fedot)                      | 0.9628          | 0.9781          | 0.9578              | 0.9752              |
|                         | MADDAutoML(artillery\_no\_tuning)             | 0.9476          | 0.9696          | 0.9493              | 0.9704              |
|                         | MADDAutoML(tabpfn)                            | 0.9459          | 0.9686          | 0.9510 (512)        | 0.9713 (512)        |
|                         | MADDAutoML(artillery\_with\_tuning)           | 0.9561          | 0.9743          | 0.9459              | 0.9683              |
|                         | **MADDAutoML(gbm_stacking_with_tuning)**     | **0.9628**      | **0.9782**      | **0.9645**          | **0.9792**          |
| **Multiple sclerosis**  | MADDAutoML(blending\_no\_tuner\_not\_compose) | 0.8724          | 0.9092          | **0.8891**          | **0.9211**          |
|                         | MADDAutoML(clear\_fedot)                      | **0.8861**      | **0.9194**      | 0.8825              | 0.9161              |
|                         | MADDAutoML(artillery\_no\_tuning)             | 0.8724          | 0.9099          | 0.8760              | 0.9125              |
|                         | MADDAutoML(tabpfn)                            | 0.8637          | 0.9049          | 0.8695 (512)        | 0.9097 (512)        |
|                         | MADDAutoML(artillery\_with\_tuning)           | 0.8840          | 0.9181          | 0.8738              | 0.9107              |
|                         | MADDAutoML(gbm_stacking_with_tuning)         | 0.8811          | 0.9154          | 0.8825              | 0.9153              |
| **Parkinson's disease** | MADDAutoML(blending\_no\_tuner\_not\_compose) | 0.8582          | 0.8439          | 0.8582              | 0.8452              |
|                         | MADDAutoML(clear\_fedot)                      | 0.8608          | 0.8471          | 0.8570              | 0.8444              |
|                         | MADDAutoML(artillery\_no\_tuning)             | 0.8506          | 0.8331          | 0.8531              | 0.8350              |
|                         | MADDAutoML(tabpfn)                            | 0.8557          | 0.8360          | 0.8493 (512)        | 0.8295 (512)        |
|                         | MADDAutoML(artillery\_with\_tuning)           | 0.8467          | 0.8315          | 0.8544              | 0.8399              |
|                         | **MADDAutoML(gbm_stacking_with_tuning)**     | **0.8633**      | **0.8499**      | **0.8633**          | **0.8508**          |
| **Dyslipedemia**        | MADDAutoML(blending\_no\_tuner\_not\_compose) | 0.7407          | 0.6711          | 0.7672              | 0.6944              |
|                         | MADDAutoML(clear\_fedot)                      | 0.7302          | 0.6623          | 0.7090              | 0.6358              |
|                         | MADDAutoML(artillery\_no\_tuning)             | 0.7566          | 0.6892          | **0.7725**          | **0.7034**          |
|                         | MADDAutoML(tabpfn)                            | **0.7672**      | **0.7027**      | 0.7566 (512)        | 0.6806 (512)        |
|                         | MADDAutoML(artillery\_with\_tuning)           | 0.7513          | 0.6887          | 0.7513              | 0.6846              |
|                         | MADDAutoML(gbm_stacking_with_tuning)         | 0.7196          | 0.6490          | 0.7460              | 0.6712              |
| **Drug resistance**     | MADDAutoML(blending\_no\_tuner\_not\_compose) | 0.8109          | 0.8706          | 0.8109              | 0.8701              |
|                         | MADDAutoML(clear\_fedot)                      | **0.8453**      | **0.8958**      | 0.8281              | **0.8864**          |
|                         | MADDAutoML(artillery\_no\_tuning)             | 0.8109          | 0.8736          | 0.8109              | 0.8726              |
|                         | MADDAutoML(tabpfn)                            | 0.8052          | 0.8702          | 0.7564 (512)        | 0.8317 (512)        |
|                         | MADDAutoML(artillery\_with\_tuning)           | 0.8309          | 0.8876          | **0.8195**          | 0.8795              |
|                         | MADDAutoML(gbm_stacking_with_tuning)         | 0.8166          | 0.8764          | 0.8166              | 0.8755              |
| **Lung cancer**         | MADDAutoML(blending\_no\_tuner\_not\_compose) | 0.7220          | 0.7138          | 0.7437              | 0.7361              |
|                         | MADDAutoML(clear\_fedot)                      | 0.7437          | 0.7361          | 0.7365              | 0.7068              |
|                         | MADDAutoML(artillery\_no\_tuning)             | 0.7473          | 0.7407          | 0.7256              | 0.7054              |
|                         | MADDAutoML(tabpfn)                            | 0.7798          | 0.7698          | 0.7726 (512)        | 0.7726 (512)        |
|                         | MADDAutoML(artillery\_with\_tuning)           | 0.7690          | 0.7647          | **0.7834**          | **0.7778**          |
|                         | MADDAutoML(gbm_stacking_with_tuning)         | **0.7798**      | **0.7698**      | 0.7653              | 0.7529              |



### Выводы:  

---

### 🔹 **1. Alzheimer's disease**

* **Лидер**: `MADDAutoML(clear_fedot)` — показала **наивысшие значения Accuracy (0.9628) и F1-score (0.9781)** при 1024.
* В целом, различия между моделями невелики, но `clear_fedot` стабильно лидирует.

---

### 🔹 **2. Multiple sclerosis**

* **Лучшие результаты**:

  * При 1024 — `clear_fedot` (0.8861 / 0.9194),
  * При 2048/512 — `blending_no_tuner_not_compose` (0.8891 / 0.9211).
* Вывод: **наиболее стабильные результаты между clear\_fedot и blending**.

---

### 🔹 **3. Parkinson's disease**

* **Небольшие отличия**, но:

  * `clear_fedot` снова лидер по Accuracy (0.8608) и F1-score (0.8471) при 1024.
  * `blending` почти наравне при 2048/512.
* Итог: **clear\_fedot чуть лучше, но разница минимальна**.

---

### 🔹 **4. Dyslipidemia**

* На 1024: **лучший F1-score у `tabpfn` (0.7027)**, но не Accuracy.
* На 2048/512: лидирует **`artillery_no_tuning`** по обоим метрикам (0.7725 / 0.7034).
* Вывод: **`artillery_no_tuning` показывает лучшие результаты при увеличении данных**.

---

### 🔹 **5. Drug resistance**

* **Лучший Accuracy и F1-score** при 1024 — `clear_fedot` (0.8453 / 0.8958).
* При 2048/512 лидирует `artillery_with_tuning` по Accuracy (0.8195), но немного уступает в F1-score.
* Итог: **clear\_fedot наиболее сбалансированная модель**, особенно при меньших объёмах данных.

---

### 🔹 **6. Lung cancer**

* **Явный лидер — `tabpfn` на 1024** (0.7798 / 0.7698).
* При 2048/512: **лучше всех — `artillery_with_tuning` (0.7834 / 0.7778)**.
* Вывод: **модель с тюнингом артиллерии показала лучшие результаты на расширенном датасете**.

---

### 📌 **Общие тенденции:**

* **`clear_fedot`** — наиболее стабильная модель по многим задачам, особенно на 1024.
* **`artillery_with_tuning`** — эффективна на больших выборках.
* **`tabpfn`** часто показывает хорошие результаты при 1024, но теряет эффективность на 512.
* **`blending_no_tuner_not_compose`** нередко оказывается в топе при 2048.

---


