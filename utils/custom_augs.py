from PIL import Image, ImageFilter, ImageEnhance
import random
import numpy as np


class RandomBlur:
    def __init__(self, p=0.7, radius_range=(1, 3)):
        self.p = p
        self.radius_range = radius_range
    def __call__(self, img):
        if random.random() > self.p:
            return img
        radius = random.uniform(*self.radius_range)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

class RandomPerspectiveCustom:
    def __init__(self, p=0.7, distortion_scale=0.3):
        self.p = p
        self.distortion_scale = distortion_scale
    def __call__(self, img):
        if random.random() > self.p:
            return img
        width, height = img.size
        d = int(min(width, height) * self.distortion_scale)
        startpoints = [(0,0), (width,0), (width,height), (0,height)]
        endpoints = [(x + random.randint(-d, d), y + random.randint(-d, d)) for x, y in startpoints]
        # вычисляем коэффициенты
        mat = []
        for p1, p2 in zip(startpoints, endpoints):
            mat.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            mat.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
        A = np.array(mat, dtype=np.float32)
        B = np.array(endpoints).reshape(8)
        coeffs = np.linalg.solve(A, B).tolist()
        return img.transform((width, height), Image.PERSPECTIVE, coeffs, Image.BICUBIC)

class RandomBrightnessContrast:
    def __init__(self, p=0.7, brightness_range=(0.6,1.4), contrast_range=(0.6,1.4)):
        self.p = p
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    def __call__(self, img):
        if random.random() > self.p:
            return img
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*self.brightness_range))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*self.contrast_range))
        return img