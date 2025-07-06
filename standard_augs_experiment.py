from PIL import Image
from pathlib import Path
from torchvision import transforms
from utils.visualization_utils import show_multiple_augmentations


train_dir = Path('data/train')
output_dir = Path('results/standard_augs')
output_dir.mkdir(parents=True, exist_ok=True)

# создаю стандартные аугментации
standard_augmentations = {
    'RandomHorizontalFlip': transforms.RandomHorizontalFlip(p=1.0),
    'RandomCrop': transforms.RandomCrop(200, padding=20),
    'ColorJitter': transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    'RandomRotation': transforms.RandomRotation(degrees=30),
    'RandomGrayscale': transforms.RandomGrayscale(p=1.0),
}

all_augs = transforms.Compose(list(standard_augmentations.values()))

# выбираю изображения
class_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])[:5]
selected = []
for class_dir in class_dirs:
    imgs = sorted(class_dir.glob('*.*'))
    if imgs:
        selected.append((class_dir.name, imgs[0]))

# отрисовываю результат применения аугментаций
for label, img_path in selected:
    img = Image.open(img_path).convert('RGB')

    individual_imgs = [aug(img) for aug in standard_augmentations.values()]
    individual_titles = list(standard_augmentations.keys())

    combined_img = all_augs(img)

    augmented_imgs = individual_imgs + [combined_img]
    titles = individual_titles + ['All_Augmentations']

    out_all = output_dir / f"{label}_All.png"
    show_multiple_augmentations(img, augmented_imgs, out_all, titles)
