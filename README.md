# Задание 1: Стандартные аугментации torchvision

Создал пайплайн стандартных аугментаций из RandomHorizontalFlip, RandomCrop, ColorJitter, RandomRotation, RandomGrayscale. Визуализировал разницу между ними. Вот пример:

(На крайней правой картинке применение сразу же всех аугментаций)

![Татсумаки_All](https://github.com/user-attachments/assets/6071e522-145c-4b8c-b415-0c192a707948)

# Задание 2: Кастомные аугментации

Реализовал 3 кастомные аугментации: случайное размытие, случайная перспектива, случайная контрастность

Сравнил кастомные аугментации с аугментациями из практического занятия. RandomBlur_vs_AddGaussianNoise: 

AddGaussianNoise делает так, чтобы изображение "рябило", а RandomBlur добавляет мутность

![Татсумаки_RandomBlur_vs_AddGaussianNoise](https://github.com/user-attachments/assets/5dbe22a3-a866-4ed4-8361-b48fa0993b84)

еще один пример. ElfsticTransform совсем чуть чуть изгибает контуры лица, а кастомный слой сильно изгибает целое изображение

![Татсумаки_RandomPerspectiveCustom_vs_ElasticTransform](https://github.com/user-attachments/assets/f6799364-50ea-4f9e-88d4-c46591936678)

RandomBrightnessContrast_vs_AutoContrast. RandomBrightnessContrast Сильно подчеркивыет любые контуры изображения, AutoContrast практически не изменил картинку

![Татсумаки_RandomBrightnessContrast_vs_AutoContrast](https://github.com/user-attachments/assets/6b2300b8-2c45-4cf4-b961-a1f5944cbae3)

# Задание 3: Анализ датасета

Составил диаграмму классов и распределения размеров изображений

![classes_count_dataset](https://github.com/user-attachments/assets/9d145ac3-9d4a-4e7c-9e4e-5b44aefad288)

Все классы изображений имеют одинаковое число изображений (30 на train и 100 на test)

Распределение размеров изобрежений показывает, что в датасете есть изображения, размер которых сильно больше, чем у большинства

Составил таблицу размеров

| Metric   | Width       | Height      |
|----------|-------------|-------------|
| Min      | 210.0       | 220.0       |
| Max      | 736.0       | 1308.0      |
| Mean     | 545.64      | 629.27      |

У некоторых изображений высота в 2 раза превышает среднее значение

# Задание 4: Pipeline аугментаций

Реализовал класс AugmentationPipeline с методами:
- add_augmentation(name, aug)
- remove_augmentation(name)
- apply(image)
- get_augmentations()

Создал три конфигурации:

- light (AutoContrast)
- medium (RandomBlur, RandomBrightnessContrast)
- heavy (RandomPerspective, ElasticTransform, GaussianNoise)

Применил конфигурации к изображениям из train. Приведу пример:

light (небольшие изменения в контрасте, трудно отличить от оригинала)

![08eca4540751e81088cf48db01c0d391_light](https://github.com/user-attachments/assets/ccffaa0c-78c2-42e1-9dfa-acbc945e2eb3)

heavy (изображение сильно видоизменено, трудно воспринимать)

![08eca4540751e81088cf48db01c0d391_heavy](https://github.com/user-attachments/assets/b49e184e-9aa0-430f-b3de-07bf5f830e78)

# Задание 5: Эксперимент с размерами

Провел эксперимент с разными размерами изображений (64x64, 128x128, 224x224, 512x512)

Привожу таблицу с результатами (замерял для 100 изображений):

| Size      | Time (seconds) | Memory (KB)     |
|-----------|---------------:|----------------:|
| 64×64     |           0.88 |      737.02     |
| 128×128   |           2.54 |      404.64     |
| 224×224   |           4.04 |     1228.55     |
| 512×512   |           7.46 |     6393.44     |

Также построил графики зависимости времени и памяти от размера изображений

![memory_plot](https://github.com/user-attachments/assets/8f8b0aa7-e824-4fc5-90df-62dba2c68b58)

![time_plot](https://github.com/user-attachments/assets/5cdd6b85-cdaa-480a-99ae-85ad0b0183fa)

Видно, что затраты на время и память геометрически возрастают с увеличением размера изображений

# Задание 6: Дообучение предобученных моделей

Использовал предобученную модель resnet18

Модель нестабильно обучалась и переобучилась. привожу графики истории обучения:

![train_history](https://github.com/user-attachments/assets/826cd1b7-f7e6-4f63-a0de-23e6814d34ac)

