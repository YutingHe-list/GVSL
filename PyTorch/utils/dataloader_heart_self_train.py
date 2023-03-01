from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, unlabeled_file_dir):
        super(DatasetFromFolder3D, self).__init__()
        self.unlabeled_filenames = [x for x in listdir(join(unlabeled_file_dir, 'image')) if is_image_file(x)]
        self.unlabeled_file_dir = unlabeled_file_dir

    def __getitem__(self, index):
        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabed_img1 = sitk.ReadImage(join(self.unlabeled_file_dir, 'image', self.unlabeled_filenames[random_index]))
        unlabed_img1 = sitk.GetArrayFromImage(unlabed_img1)
        unlabed_img1 = np.where(unlabed_img1 < 0., 0., unlabed_img1)
        unlabed_img1 = np.where(unlabed_img1 > 2048., 2048., unlabed_img1)
        unlabed_img1 = unlabed_img1 / 2048.
        unlabed_img1 = unlabed_img1.astype(np.float32)
        unlabed_img1 = unlabed_img1[np.newaxis, :, :, :]

        random_index = np.random.randint(low=0, high=len(self.unlabeled_filenames))
        unlabed_img2 = sitk.ReadImage(join(self.unlabeled_file_dir, 'image', self.unlabeled_filenames[random_index]))
        unlabed_img2 = sitk.GetArrayFromImage(unlabed_img2)
        unlabed_img2 = np.where(unlabed_img2 < 0., 0., unlabed_img2)
        unlabed_img2 = np.where(unlabed_img2 > 2048., 2048., unlabed_img2)
        unlabed_img2 = unlabed_img2 / 2048.
        unlabed_img2 = unlabed_img2.astype(np.float32)
        unlabed_img2 = unlabed_img2[np.newaxis, :, :, :]

        return unlabed_img1, unlabed_img2

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.unlabeled_filenames)

