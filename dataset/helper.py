from torchvision import transforms as T

train_transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
])
test_transform = None