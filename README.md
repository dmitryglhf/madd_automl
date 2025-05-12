Вот таблица с **выделенными наилучшими метриками** для каждого датасета:  

| Датасет                     | Модель                                   | Accuracy (1024) | F1-score (1024) | Accuracy (2048/512) | F1-score (2048/512) |
|-----------------------------|------------------------------------------|-----------------|------------------|---------------------|---------------------|
| **Alzheimer's disease**      | MADDAutoML(blending_no_tuner_not_compose) | 0.9544          | 0.9733           | 0.9578              | 0.9753              |
|                             | MADDAutoML(clear_fedot)                  | **0.9628**      | **0.9781**       | 0.9578              | 0.9752              |
|                             | MADDAutoML(artillery_no_tuning)          | 0.9476          | 0.9696           | 0.9493              | 0.9704              |
|                             | MADDAutoML(tabpfn)                       | 0.9459          | 0.9686           | **0.9510 (512)**    | **0.9713 (512)**    |
|                             |                                          |                 |                  |                     |                     |
| **Multiple sclerosis**       | MADDAutoML(blending_no_tuner_not_compose) | 0.8724          | 0.9092           | **0.8891**          | **0.9211**          |
|                             | MADDAutoML(clear_fedot)                  | **0.8861**      | **0.9194**       | 0.8825              | 0.9161              |
|                             | MADDAutoML(artillery_no_tuning)          | 0.8724          | 0.9099           | 0.8760              | 0.9125              |
|                             | MADDAutoML(tabpfn)                       | 0.8637          | 0.9049           | 0.8695 (512)        | 0.9097 (512)        |
|                             |                                          |                 |                  |                     |                     |
| **Parkinson's disease**      | MADDAutoML(blending_no_tuner_not_compose) | 0.8582          | 0.8439           | **0.8582**          | **0.8452**          |
|                             | MADDAutoML(clear_fedot)                  | **0.8608**      | **0.8471**       | 0.8570              | 0.8444              |
|                             | MADDAutoML(artillery_no_tuning)          | 0.8506          | 0.8331           | 0.8531              | 0.8350              |
|                             | MADDAutoML(tabpfn)                       | 0.8557          | 0.8360           | 0.8493 (512)        | 0.8295 (512)        |
|                             |                                          |                 |                  |                     |                     |
| **Dyslipedemia**            | MADDAutoML(blending_no_tuner_not_compose) | 0.7407          | 0.6711           | **0.7672**          | **0.6944**          |
|                             | MADDAutoML(clear_fedot)                  | 0.7302          | 0.6623           | 0.7090              | 0.6358              |
|                             | MADDAutoML(artillery_no_tuning)          | **0.7566**      | **0.6892**       | 0.7725              | 0.7034              |
|                             | MADDAutoML(tabpfn)                       | 0.7672          | 0.7027           | 0.7566 (512)        | 0.6806 (512)        |
|                             |                                          |                 |                  |                     |                     |
| **Drug resistance**         | MADDAutoML(blending_no_tuner_not_compose) | 0.8109          | 0.8706           | 0.8109              | 0.8701              |
|                             | MADDAutoML(clear_fedot)                  | **0.8453**      | **0.8958**       | 0.8281              | **0.8864**          |
|                             | MADDAutoML(artillery_no_tuning)          | 0.8109          | 0.8736           | 0.8109              | 0.8726              |
|                             | MADDAutoML(tabpfn)                       | 0.8052          | 0.8702           | 0.7564 (512)        | 0.8317 (512)        |
|                             |                                          |                 |                  |                     |                     |
| **Lung cancer**             | MADDAutoML(blending_no_tuner_not_compose) | 0.7220          | 0.7138           | 0.7437              | 0.7361              |
|                             | MADDAutoML(clear_fedot)                  | **0.7437**      | **0.7361**       | 0.7365              | 0.7068              |
|                             | MADDAutoML(artillery_no_tuning)          | 0.7473          | 0.7407           | 0.7256              | 0.7054              |
|                             | MADDAutoML(tabpfn)                       | **0.7798**      | **0.7698**       | **0.7726 (512)**    | **0.7726 (512)**    |

### Выводы:  
- **Alzheimer's disease**: Лучшая модель — `MADDAutoML(clear_fedot)` (Accuracy **0.9628**, F1 **0.9781**).  
- **Multiple sclerosis**: Лучшая модель — `MADDAutoML(clear_fedot)` (Accuracy **0.8861**, F1 **0.9194**) для 1024 фичей и `MADDAutoML(blending_no_tuner_not_compose)` (Accuracy **0.8891**, F1 **0.9211**) для 2048.  
- **Parkinson's disease**: Лучшая модель — `MADDAutoML(clear_fedot)` (Accuracy **0.8608**, F1 **0.8471**) для 1024 фичей и `MADDAutoML(blending_no_tuner_not_compose)` (Accuracy **0.8582**, F1 **0.8452**) для 2048.  
- **Dyslipedemia**: Лучшая модель — `MADDAutoML(artillery_no_tuning)` (Accuracy **0.7566**, F1 **0.6892**) для 1024 фичей и `MADDAutoML(blending_no_tuner_not_compose)` (Accuracy **0.7672**, F1 **0.6944**) для 2048.  
- **Drug resistance**: Лучшая модель — `MADDAutoML(clear_fedot)` (Accuracy **0.8453**, F1 **0.8958**) для 1024 фичей и (F1 **0.8864**) для 2048.  
- **Lung cancer**: Лучшая модель — `MADDAutoML(tabpfn)` (Accuracy **0.7798**, F1 **0.7698**) для 1024 фичей и (Accuracy **0.7726**, F1 **0.7726**) для 512.
