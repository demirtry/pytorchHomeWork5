from pathlib import Path
from PIL import Image
from torchvision import transforms
from utils.visualization_utils import show_multiple_augmentations
from utils.extra_augs import AddGaussianNoise, ElasticTransform, AutoContrast
from utils.custom_augs import RandomBlur, RandomPerspectiveCustom, RandomBrightnessContrast


train_dir = Path('data/train')
output_dir = Path('results/custom_augs')
output_dir.mkdir(parents=True, exist_ok=True)

# Пары для сравнения
pairs = [
    ('RandomBlur', RandomBlur(p=1.0), 'AddGaussianNoise', AddGaussianNoise(std=0.3)),
    ('RandomPerspectiveCustom', RandomPerspectiveCustom(p=1.0), 'ElasticTransform', ElasticTransform(p=1.0, alpha=15, sigma=5)),
    ('RandomBrightnessContrast', RandomBrightnessContrast(p=1.0), 'AutoContrast', AutoContrast(p=1.0))
]

class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])[4:5]
selected = [(d.name, sorted(d.glob('*.*'))[0]) for d in class_dirs if list(d.glob('*.*'))]

# применение и визуализация
for label, img_path in selected:
    img = Image.open(img_path).convert('RGB')
    for cust_name, cust_aug, old_name, old_aug in pairs:
        cust_img = cust_aug(img)
        old_out = old_aug(transforms.ToTensor()(img))

        path = f"{output_dir}/{label}_{cust_name}_vs_{old_name}.png"
        show_multiple_augmentations(img, [cust_img, old_out], path, [cust_name, old_name])
