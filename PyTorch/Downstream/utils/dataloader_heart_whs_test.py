from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, labeled_file_dir):
        super(DatasetFromFolder3D, self).__init__()
        self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'image')) if is_image_file(x)]
        self.labeled_file_dir = labeled_file_dir

    def __getitem__(self, index):
        img = sitk.ReadImage(join(self.labeled_file_dir, 'image', self.labeled_filenames[index]))
        img = sitk.GetArrayFromImage(img)
        img = img + 1024
        img = np.where(img < 0., 0., img)
        img = np.where(img > 2048., 2048., img)
        img = img / 2048.
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :, :]

        return img, self.labeled_filenames[index]

    def __len__(self):
        return len(self.labeled_filenames)

