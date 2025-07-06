from pathlib import Path
from PIL import Image
import pandas as pd
from utils.visualization_utils import dataset_visualization


def analyze_dataset(data_dir='data'):
    data_dir = Path(data_dir)
    splits = ['train', 'test']
    image_info = []

    for split in splits:
        split_dir = data_dir / split

        classes = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for cls in classes:
            for img_path in cls.glob('*.*'):
                try:
                    with Image.open(img_path) as img:
                        width, height = img.size
                        image_info.append({
                            'split': split,
                            'class': cls.name,
                            'width': width,
                            'height': height,
                            'area': width * height
                        })
                except Exception as e:
                    print(f"Ошибка при обработке {img_path}: {e}")

    df = pd.DataFrame(image_info)
    dataset_visualization(df)


if __name__ == '__main__':
    analyze_dataset()
