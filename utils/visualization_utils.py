import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
import pandas as pd


def show_images(images, path, labels=None, nrow=8, title=None, size=128):
    """Визуализирует батч изображений."""
    images = images[:nrow]

    # Увеличиваем изображения до 128x128 для лучшей видимости
    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]

    # Создаем сетку изображений
    fig, axes = plt.subplots(1, nrow, figsize=(nrow * 2, 2))
    if nrow == 1:
        axes = [axes]

    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализуем для отображения
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def show_single_augmentation(original_img, augmented_img, path, title="Аугментация"):
    """Визуализирует оригинальное и аугментированное изображение рядом."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)

    to_tensor = transforms.ToTensor()
    if not isinstance(original_img, torch.Tensor):
        original_img = to_tensor(original_img)
    if not isinstance(augmented_img, torch.Tensor):
        augmented_img = to_tensor(augmented_img)

    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)

    # Оригинальное изображение
    orig_np = orig_resized.permute(1, 2, 0).numpy()
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')

    # Аугментированное изображение
    aug_np = aug_resized.permute(1, 2, 0).numpy()
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def show_multiple_augmentations(original_img, augmented_imgs, path, titles):
    """Визуализирует оригинальное изображение и несколько аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))

    # Увеличиваем изображения
    resize_transform = transforms.Resize((128, 128), antialias=True)

    to_tensor = transforms.ToTensor()

    if not isinstance(original_img, torch.Tensor):
        original_img = to_tensor(original_img)

    augmented_imgs = [img if isinstance(img, torch.Tensor) else to_tensor(img) for img in augmented_imgs]
    orig_resized = resize_transform(original_img)

    # Оригинальное изображение
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    # Аугментированные изображения
    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_resized = resize_transform(aug_img)
        aug_np = aug_resized.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def dataset_visualization(df):
    class_counts = df['class'].value_counts().sort_index()

    size_df = df[['width', 'height']].agg(['min', 'max', 'mean'])
    size_df.to_csv('results/dataset_analysis/size_dataset_stats.csv')

    plt.figure(figsize=(12, 4))

    # Гистограмма размеров
    plt.subplot(1, 2, 1)
    plt.hist(df['area'], bins=30, color='skyblue', edgecolor='black')
    plt.title("распределение размеров изображений")
    plt.xlabel("размер (width × height)")
    plt.ylabel("Количество")

    # Кол-во изображений по классам
    plt.subplot(1, 2, 2)
    plt.bar(class_counts.index, class_counts.values, color='orange', edgecolor='black')
    plt.title("Количество изображений по классам")
    plt.xticks(rotation=45)
    plt.ylabel("Количество")

    plt.tight_layout()
    plt.savefig('results/dataset_analysis/classes_count_dataset.png')
    plt.close()


def size_visualization(results):
    df = pd.DataFrame(results)
    df.to_csv('results/size_experiments/results.csv', index=False)

    # График времени
    plt.figure()
    plt.plot(df['size'], df['memory_KB'], marker='o')
    plt.xlabel('Image Size')
    plt.ylabel('Time seconds')
    plt.title('Time vs Size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/size_experiments/time_plot.png')
    plt.close()

    # График памяти
    plt.figure()
    plt.plot(df['size'], df['memory_KB'], marker='o', color='red')
    plt.xlabel('Image Size')
    plt.ylabel('Memory (Kb)')
    plt.title('Peak Memory Usage vs Image Size (tracemalloc)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/size_experiments/memory_plot.png')
    plt.close()
