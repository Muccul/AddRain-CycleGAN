from torchvision.transforms import transforms
from RandAugment import RandAugment
import visdom
from PIL import Image


viz = visdom.Visdom()

transform_train = transforms.Compose([
    transforms.RandomCrop(256),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# Add RandAugment with N, M(hyperparameter)
transform_train.transforms.insert(0, RandAugment(2, 10))


AB = Image.open("datasets/img2rain/trainA/norain-1.png").convert('RGB')





