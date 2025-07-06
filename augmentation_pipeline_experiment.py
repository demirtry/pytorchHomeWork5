from utils.augmentation_pipeline import AugmentationPipeline
from utils.extra_augs import AddGaussianNoise, ElasticTransform, AutoContrast
from custom_augs_experiment import RandomBlur, RandomPerspectiveCustom, RandomBrightnessContrast
from pathlib import Path
from PIL import Image
from torchvision import transforms
from utils.visualization_utils import show_multiple_augmentations


def create_configurations():
    """
    Возвращает словарь конфигураций
    """
    light = AugmentationPipeline()
    light.add_augmentation('autocontrast', AutoContrast(p=1.0))

    medium = AugmentationPipeline()
    medium.add_augmentation('blur', RandomBlur(p=1.0))
    medium.add_augmentation('brightness_contrast', RandomBrightnessContrast(p=1.0))

    heavy = AugmentationPipeline()
    heavy.add_augmentation('perspective', RandomPerspectiveCustom(p=1.0))
    heavy.add_augmentation('elastic', ElasticTransform(p=1.0))
    heavy.add_augmentation('gaussian_noise', AddGaussianNoise(std=0.2))

    return {
        'light': light,
        'medium': medium,
        'heavy': heavy
    }

def apply_pipelines_to_dataset(input_dir='data/train', output_base='results/pipelines'):
    """
    применяет пайплайн аугментаций к датасету (3 картинки на класс)
    """
    input_dir = Path(input_dir)
    output_base = Path(output_base)
    output_base.mkdir(parents=True, exist_ok=True)

    configs = create_configurations()
    tensor_to_pil = transforms.ToPILImage()

    for config_name, pipeline in configs.items():
        output_dir = output_base / config_name
        output_dir.mkdir(exist_ok=True, parents=True)

        for class_dir in input_dir.iterdir():
            if not class_dir.is_dir():
                continue
            images = list(class_dir.glob('*.*'))[:3]  # 3 картинками на класс
            class_output_dir = output_dir / class_dir.name
            class_output_dir.mkdir(exist_ok=True)

            for img_path in images:
                img = Image.open(img_path).convert("RGB")
                # Превращаю в тензор для примеров с занятия
                if any(isinstance(aug, (AddGaussianNoise, ElasticTransform, AutoContrast))
                       for aug in pipeline.augmentations.values()):
                    img_tensor = transforms.ToTensor()(img)
                    for name, aug in pipeline.augmentations.items():
                        if isinstance(aug, (AddGaussianNoise, ElasticTransform, AutoContrast)):
                            img_tensor = aug(img_tensor)
                        else:
                            img = aug(tensor_to_pil(img_tensor))
                            img_tensor = transforms.ToTensor()(img)
                    result = tensor_to_pil(img_tensor)
                else:
                    result = pipeline.apply(img)

                save_path = class_output_dir / f"{img_path.stem}_{config_name}.png"
                result.save(save_path)


if __name__ == '__main__':
    apply_pipelines_to_dataset()
