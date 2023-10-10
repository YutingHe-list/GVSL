from os.path import join
from os import listdir
import SimpleITK as sitk
from torch.utils import data
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii", ".nii.gz"])

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, labeled_file_dir, num_classes, shape=None):
        super(DatasetFromFolder3D, self).__init__()
        self.labeled_filenames = [x for x in listdir(join(labeled_file_dir, 'image')) if is_image_file(x)]
        self.labeled_file_dir = labeled_file_dir
        self.num_classes = num_classes
        self.shape = shape

    def __getitem__(self, index):
        img = sitk.ReadImage(join(self.labeled_file_dir, 'image', self.labeled_filenames[index]))
        img = sitk.GetArrayFromImage(img)
        img = img + 1024
        img = np.where(img < 0., 0., img)
        img = np.where(img > 2048., 2048., img)
        img = img / 2048.
        img = img[np.newaxis, :, :, :]

        lab = sitk.ReadImage(join(self.labeled_file_dir, 'label', self.labeled_filenames[index]))
        lab = sitk.GetArrayFromImage(lab)
        lab = np.where(lab == 205, 1, lab)
        lab = np.where(lab == 420, 2, lab)
        lab = np.where(lab == 500, 3, lab)
        lab = np.where(lab == 550, 4, lab)
        lab = np.where(lab == 600, 5, lab)
        lab = np.where(lab == 820, 6, lab)
        lab = np.where(lab == 850, 7, lab)
        lab = self.to_categorical(lab, self.num_classes)

        if self.shape is not None:
            img, lab = self.reshape_img(img, lab, self.shape)
        img = img.astype(np.float32)
        lab = lab.astype(np.float32)

        return img, lab, self.labeled_filenames[index]

    def reshape_img(self, image, label, shape):
        if image.shape[1] <= shape[0]:
            image = np.concatenate([image, np.zeros((image.shape[0], shape[0]-image.shape[1], image.shape[2], image.shape[3]))], axis=1)
            label = np.concatenate([label, np.zeros((label.shape[0], shape[0]-label.shape[1], label.shape[2], label.shape[3]))], axis=1)
            x_idx = 0
        else:
            x_idx = np.random.randint(image.shape[1] - shape[0])

        if image.shape[2] <= shape[1]:
            image = np.concatenate([image, np.zeros((image.shape[0], image.shape[1], shape[1]-image.shape[2], image.shape[3]))], axis=2)
            label = np.concatenate([label, np.zeros((label.shape[0], label.shape[1], shape[1] - label.shape[2], label.shape[3]))], axis=2)
            y_idx = 0
        else:
            y_idx = np.random.randint(image.shape[2] - shape[1])

        if image.shape[3] <= shape[2]:
            image = np.concatenate([image, np.zeros((image.shape[0], image.shape[1], image.shape[2], shape[2]-image.shape[3]))], axis=3)
            label = np.concatenate([label, np.zeros((label.shape[0], label.shape[1], label.shape[2], shape[2] - label.shape[3]))], axis=3)
            z_idx = 0
        else:
            z_idx = np.random.randint(image.shape[3] - shape[2])

        image = image[:, x_idx:x_idx+shape[0], y_idx:y_idx+shape[1], z_idx:z_idx+shape[2]]
        label = label[:, x_idx:x_idx+shape[0], y_idx:y_idx+shape[1], z_idx:z_idx+shape[2]]
        return image, label

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
        return len(self.labeled_filenames)

