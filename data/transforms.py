from torchvision import transforms

def build_transforms(cfg, train: bool):
    size = cfg["data"]["image_size"]
    mean = cfg["data"]["imagenet_mean"]
    std  = cfg["data"]["imagenet_std"]
    if train:
        return transforms.Compose([
            transforms.Resize((size, size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
