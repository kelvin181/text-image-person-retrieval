import torchvision.transforms as T


def build_transforms(img_size=(384, 128)):
    height, width = img_size
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    return T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
