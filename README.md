# nner_diploma

В репозитории сохранены разработки по распознаванию вложенных именованных сущностей.
В ноутбуке general_pipeline.ipynb подробно описаны требуемые данные и порядок запуска для общей задачи распознавания на больших данных
В ноутбуке few_shot_exp.ipynb описаны эксперименты, порядок запуска и даны ссылки на требуемые данные.

Основные модули приведены в разделе scripts:
* Основные коды для получения представлений отрезков текстов содержатся в разделе scripts/extractor_modules
* Код для классификации отрезков содержится в scripts/classifier_module.py
* Общий код, объединяющий оба модуля, приведен в scripts/span_classifier.py, в зависимости от задачи, как приведено в ноутбуках, используется либо в режиме извлечения, либо классификации
* Общие пайплайны обучения приведены в файле scripts/task_trainers.py
