import numpy as np
import io
import random
import os
from PIL import Image, ImageFilter
from torchvision import transforms


class GaussianNoise:
    """随机高斯噪声，模拟低质量扫描"""
    def __init__(self, prob=0.3, mean=0, std=25):
        self.prob = prob
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if random.random() < self.prob:
            arr = np.array(img).astype(np.float32)
            arr = np.clip(arr + np.random.normal(self.mean, self.std, arr.shape), 0, 255)
            return Image.fromarray(arr.astype(np.uint8))
        return img


class GaussianBlur:
    """随机高斯模糊，模拟失焦"""
    def __init__(self, prob=0.3, radius_range=(1, 3)):
        self.prob = prob
        self.radius_range = radius_range

    def __call__(self, img):
        if random.random() < self.prob:
            radius = random.uniform(*self.radius_range)
            return img.filter(ImageFilter.GaussianBlur(radius=radius))
        return img


class JpegCompression:
    """随机 JPEG 压缩，模拟压缩伪影"""
    def __init__(self, prob=0.3, quality_range=(40, 85)):
        self.prob = prob
        self.quality_range = quality_range

    def __call__(self, img):
        if random.random() < self.prob:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(*self.quality_range))
            buf.seek(0)
            return Image.open(buf).copy()
        return img


# 训练用增强 pipeline
train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    GaussianNoise(prob=0.3),
    GaussianBlur(prob=0.3),
    JpegCompression(prob=0.3),
])


def _load_image(image_path):
    """加载图片"""
    image = Image.open(image_path).convert("RGB")
    return image


def process_fn(examples, image_root):
    """
    PaddleFormers 数据处理函数
    - 加载图片
    - 应用数据增强
    """
    images = []
    for img_path in examples["images"]:
        full_path = os.path.join(image_root, img_path) if not os.path.isabs(img_path) else img_path
        image = _load_image(full_path)
        image = train_transform(image)
        images.append(image)
    return {"pixel_values": images}
