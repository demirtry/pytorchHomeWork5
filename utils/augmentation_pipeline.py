class AugmentationPipeline:
    def __init__(self):
        self.augmentations = {}

    def add_augmentation(self, name, aug):
        """добавляет аугментацию по имени"""
        self.augmentations[name] = aug

    def remove_augmentation(self, name):
        """Удаляет аугментацию по имени"""
        if name in self.augmentations:
            del self.augmentations[name]

    def apply(self, image):
        """
        Применяет все аугментации к указанному изображению
        """
        for aug in self.augmentations.values():
            image = aug(image)
        return image

    def get_augmentations(self):
        """Возвращает список аугментаций"""
        return list(self.augmentations.keys())
