import torchvision.transforms as transforms
from torch.utils.data import Dataset
import h5py

class CelebaDataset_h5py(Dataset):
    def __init__(self,opt):
        self.opt = opt
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                (0.5, 0.5, 0.5))]
        )
        self.file = h5py.File(self.opt.datapath,'r',swmr=True)
        self.file_data = self.file['celeba']
    def __getitem__(self,index):
        img = self.file_data[index % self.opt.max_dataset,:,:,:]
        img = self.transform(img)
        return img
    def __len__(self):
        return self.opt.max_dataset