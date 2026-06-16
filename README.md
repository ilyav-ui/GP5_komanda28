# Групповой проект №5
Команда 28

Проект решает две связанные бизнес задачи для страховой компании. Обе задачи решают проблему более быстрого и точного выявления мошеннических страховых случаев.

## Задачи

**Задача 1 - табличные данные**

По анкете страхового случая (виновник ДТП, тип полиса, наличие свидетелей, история обращений и т.д.) предсказываем, является ли случай мошенническим.

Датасет: fraud_oracle.csv - 15 420 строк, 33 признака, целевая переменная FraudFound_P.

**Задача 2 - изображения**

По фотографиям повреждений автомобиля определяем тип повреждения. Это помогает проверить, соответствуют ли заявленные повреждения реальным.

Датасет: CarDD (COCO-формат) - crop-изображения повреждений, размеченные по классам.

## Структура репозитория

Репозиторий состоит из трёх веток.

**main** - финальные версии ноутбуков:
```
BT1_GP5_komanda28.ipynb       # Задача 1: EDA + предобработка + нейросеть на табличных данных
CARDD_GP5_komanda28.ipynb     # Задача 2: EDA + обучение CNN на изображениях повреждений
.gitignore
README.md
```

**develop** - рабочая ветка, где велась разработка по шагам:
```
v1_EDA_BT1.ipynb                    # EDA, задача 1
v2_EDA_BT1.ipynb                    # продолжение EDA
v3_EDA_BT1.ipynb                    # продолжение EDA
v4_EDA_BT1.ipynb                    # продолжение EDA
v5_EDA_BT1.ipynb                    # продолжение EDA
v6_EDA_BT1.ipynb                    # продолжение EDA
v1_1_CROP_BT2.ipynb                 # создание датасета, задача 2
v1_CROP_EDA_BT2.ipynb               # EDA, задача 2
v2_LOG_И_DATALOADER_BT2.ipynb       # логирование и DataLoader, задача 2
v3_Обучение_и_артефакты__BT2.ipynb  # функции обучения и оценки
v4_baseline_CNN_BT2.ipynb           # Baseline CNN, задача 2
v5_CNN_BatchNorm_Dporout_BT2.ipynb  # CNN + BatchNorm + Dropout, задача 2
v6_VGG16_Transferlearning_BT2.ipynb # VGG16 Transfer Learning, задача 2
v7_Сравнение_моделей_BT2.ipynb      # сравнение моделей, задача 2
BT1_GP5_komanda28.ipynb             # финальная версия, задача 1
CARDD_GP5_komanda28.ipynb           # финальная версия, задача 2
.gitignore
README.md
```

**testing** - ветка для проверки перед слиянием в main:
```
BT1_GP5_komanda28.ipynb
CARDD_GP5_komanda28.ipynb
.gitignore
README.md
```

Датасеты не хранятся в репозитории. Скачать:
- fraud_oracle.csv - [Kaggle: Vehicle Insurance Claim Fraud Detection](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection)
- CarDD - [arXiv.org](https://arxiv.org/abs/2211.00945)


## Как запустить

**Задача 1** (BT1_GP5_komanda28.ipynb):
1. Загрузить датасет fraud_oracle.csv в ноутбук
2. Запустить ячейки по порядку
3. Зависимости: pandas, numpy, matplotlib, scipy

**Задача 2** (CARDD_GP5_komanda28.ipynb):
1. Ноутбук рассчитан на Google Colab с подключённым Google Drive
2. Датасет CarDD распаковать в /content/drive/MyDrive/car_damage_project/raw/
3. Зависимости: torch, torchvision, Pillow, tqdm, wandb

## Модели

**Задача 1 - полносвязные нейронные сети:**
- Baseline MLP
- Deep MLP + BatchNorm + Dropout
- Wide & Deep + Embeddings

**Задача 2 - свёрточные нейронные сети:**
- Baseline CNN
- CNN + BatchNorm + Dropout
- VGG16 Transfer Learning

## Логирование экспериментов

Все эксперименты залогированы в Weights & Biases. Каждая модель - отдельный run с параметрами, метриками по эпохам и сохранённым чекпоинтом модели.
