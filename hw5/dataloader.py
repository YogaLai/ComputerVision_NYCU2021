from torch.utils.data import Dataset
from torchvision import transforms
from util import *

class ImgDataset(Dataset):
    def __init__(self, train_mode):
        self.imgs, self.labels = getDataset(train_mode, (256,256))
        for i in range(len(self.imgs)):
            self.imgs[i] = cv2.cvtColor(self.imgs[i], cv2.COLOR_GRAY2RGB)        
        self.transformations=transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        print(f'>> Found {len(self.labels)} images...')
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img=self.transformations(self.imgs[index])
        label=self.labels[index]
        
        return img, label