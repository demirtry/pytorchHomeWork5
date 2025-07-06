import time
import tracemalloc
from utils.extra_augs import AddGaussianNoise, Solarize, AutoContrast
from utils.datasets import CustomImageDataset
from torchvision import transforms
from utils.visualization_utils import size_visualization


sizes = [(64, 64), (128, 128), (224, 224), (512, 512)]
results = []

augmentations = [
    AddGaussianNoise(std=0.3),
    Solarize(threshold=128),
    AutoContrast(p=1.0),
]

to_tensor = transforms.ToTensor()
t0 = time.perf_counter()

for size in sizes:
    dataset = CustomImageDataset(root_dir='data/train', target_size=size)
    tracemalloc.start()
    for idx in range(100):
        img, _ = dataset[idx]
        img = to_tensor(img)
        for aug in augmentations:
            img = aug(img)

    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    results.append({
        'size': f'{size[0]}x{size[1]}',
        'time_seconds': round(elapsed, 2),
        'memory_KB': peak / 1024
    })
    size_visualization(results)
