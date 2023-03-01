import numpy as np
import torch
from scipy.special import comb
from torch import nn
import torch.nn.functional as nnf

class MirrorTransform(object):
    def augment_mirroring(self, data, code=(1, 1, 1)):
        if code[0] == 1:
            data = self.flip(data, 2)
        if code[1] == 1:
            data = self.flip(data, 3)
        if code[2] == 1:
            data = self.flip(data, 4)
        return data

    def flip(self, x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                    dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    def rand_code(self):
        code = []
        for i in range(3):
            if np.random.uniform() < 0.5:
                code.append(1)
            else:
                code.append(0)
        return code

class AffineTransformer(nn.Module):
    """
    3-D Affine Transformer
    """
    def __init__(self):
        super().__init__()

    def forward(self, src, mat, mode='bilinear'):
        norm = torch.tensor([[1, 1, 1, src.shape[2]], [1, 1, 1, src.shape[3]], [1, 1, 1, src.shape[4]]], dtype=torch.float).cuda()
        norm = norm[np.newaxis, :, :]
        mat_new = mat/norm
        grid = nnf.affine_grid(mat_new, [src.shape[0], 3, src.shape[2], src.shape[3], src.shape[4]])
        return nnf.grid_sample(src, grid, mode=mode)

class SpatialTransformer(nn.Module):
    def __init__(self):
        super(SpatialTransformer, self).__init__()

    def forward(self, src, flow, mode='bilinear', padding_mode='zeros'):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)

        if torch.cuda.is_available():
            grid = grid.cuda()

        new_locs = grid + flow

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, mode=mode, padding_mode=padding_mode)


class SpatialTransform(object):
    def __init__(self, do_rotation=True, angle_x=(-np.pi / 12, np.pi / 12), angle_y=(-np.pi / 12, np.pi / 12),
                 angle_z=(-np.pi / 12, np.pi / 12), do_scale=True, scale_x=(0.75, 1.25), scale_y=(0.75, 1.25),
                 scale_z=(0.75, 1.25), do_translate=True, trans_x=(-0.1, 0.1), trans_y=(-0.1, 0.1), trans_z=(-0.1, 0.1),
                 do_shear=True, shear_xy=(-np.pi / 18, np.pi / 18), shear_xz=(-np.pi / 18, np.pi / 18),
                 shear_yx=(-np.pi / 18, np.pi / 18), shear_yz=(-np.pi / 18, np.pi / 18),
                 shear_zx=(-np.pi / 18, np.pi / 18), shear_zy=(-np.pi / 18, np.pi / 18),
                 do_elastic_deform=True, alpha=(0., 512.), sigma=(4., 10.)):
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z

        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_translate = do_translate
        self.trans_x = trans_x
        self.trans_y = trans_y
        self.trans_z = trans_z
        self.do_shear = do_shear
        self.shear_xy = shear_xy
        self.shear_xz = shear_xz
        self.shear_yx = shear_yx
        self.shear_yz = shear_yz
        self.shear_zx = shear_zx
        self.shear_zy = shear_zy

        self.stn = SpatialTransformer()
        self.atn = AffineTransformer()

    def augment_spatial(self, data, code_aff=None, code_spa=None, mode='bilinear'):
        if code_aff is not None:
            data = self.atn(data, code_aff, mode)
        if code_spa is not None:
            data = self.stn(data, code_spa, mode=mode, padding_mode='zeros')
        return data

    def rand_coords(self, patch_size):
        coords = torch.from_numpy(self.create_zero_centered_coordinate_mesh(patch_size)).cuda()
        mat = torch.from_numpy(np.identity(3, dtype=np.float32)).cuda()
        if self.do_rotation:
            a_x = np.random.uniform(self.angle_x[0], self.angle_x[1])
            a_y = np.random.uniform(self.angle_y[0], self.angle_y[1])
            a_z = np.random.uniform(self.angle_z[0], self.angle_z[1])
            mat = self.rotate_mat(mat, a_x, a_y, a_z)
        if self.do_scale:
            sc_x = np.random.uniform(self.scale_x[0], self.scale_x[1])
            sc_y = np.random.uniform(self.scale_y[0], self.scale_y[1])
            sc_z = np.random.uniform(self.scale_z[0], self.scale_z[1])
            mat = self.scale_mat(mat, sc_x, sc_y, sc_z)
        if self.do_shear:
            s_xy = np.random.uniform(self.shear_xy[0], self.shear_xy[1])
            s_xz = np.random.uniform(self.shear_xz[0], self.shear_xz[1])
            s_yx = np.random.uniform(self.shear_yx[0], self.shear_yx[1])
            s_yz = np.random.uniform(self.shear_yz[0], self.shear_yz[1])
            s_zx = np.random.uniform(self.shear_zx[0], self.shear_zx[1])
            s_zy = np.random.uniform(self.shear_zy[0], self.shear_zy[1])
            mat = self.shear_mat(mat, s_xy, s_xz, s_yx, s_yz, s_zx, s_zy)
        if self.do_translate:
            t_x = np.random.uniform(self.trans_x[0], self.trans_x[1]) * patch_size[0]
            t_y = np.random.uniform(self.trans_y[0], self.trans_y[1]) * patch_size[1]
            t_z = np.random.uniform(self.trans_z[0], self.trans_z[1]) * patch_size[2]
            mat = self.translate_mat(mat, t_x, t_y, t_z)
        else:
            mat = self.translate_mat(mat, 0, 0, 0)
        if self.do_elastic_deform:
            a = np.random.uniform(self.alpha[0], self.alpha[1])
            s = np.random.uniform(self.sigma[0], self.sigma[1])
            coords = self.deform_coords(coords, a, s)

        ctr = torch.FloatTensor([patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2]).cuda()

        vectors = [torch.arange(0, s) for s in patch_size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = grid.type(torch.FloatTensor).cuda()


        coords += ctr[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis] - grid[np.newaxis, :, :, :, :]

        mat = mat[np.newaxis, :, :]
        return mat, coords

    def create_zero_centered_coordinate_mesh(self, shape):
        tmp = tuple([np.arange(i) for i in shape])
        coords = np.array(np.meshgrid(*tmp, indexing='ij'), dtype=np.float32)
        for d in range(len(shape)):
            coords[d] -= ((np.array(shape, dtype=np.float32) - 1) / 2.)[d]
        return coords[np.newaxis, :, :, :, :]

    def rotate_mat(self, mat, angle_x, angle_y, angle_z):
        rot_mat_x = torch.FloatTensor([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]]).cuda()
        rot_mat_y = torch.FloatTensor([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]]).cuda()
        rot_mat_z = torch.FloatTensor([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]]).cuda()
        mat = torch.matmul(rot_mat_z, torch.matmul(rot_mat_y, torch.matmul(rot_mat_x, mat)))
        return mat

    def deform_coords(self, coords, alpha, sigma):
        offsets = torch.rand(coords.shape).cuda()*2 -1
        ker1d = self._gaussian_kernel1d(sigma).astype(np.float32)[np.newaxis, np.newaxis, :]
        ker1d = torch.from_numpy(ker1d).cuda()
        ker1d1 = ker1d[:, :, :, np.newaxis, np.newaxis]
        ker1d2 = ker1d[:, :, np.newaxis, :, np.newaxis]
        ker1d3 = ker1d[:, :, np.newaxis, np.newaxis, :]

        for i in range(3):
            offsets[:, i:i+1] = torch.conv3d(input=offsets[:, i:i+1], weight=ker1d1, padding=[ker1d.shape[-1]//2, 0, 0])
            offsets[:, i:i+1] = torch.conv3d(input=offsets[:, i:i+1], weight=ker1d2, padding=[0, ker1d.shape[-1]//2, 0])
            offsets[:, i:i+1] = torch.conv3d(input=offsets[:, i:i+1], weight=ker1d3, padding=[0, 0, ker1d.shape[-1]//2])
        offsets = offsets * alpha
        indices = offsets + coords
        return indices

    def _gaussian_kernel1d(self, sigma):
        sd = float(sigma)
        radius = int(4 * sd + 0.5)
        sigma2 = sigma * sigma
        x = np.arange(-radius, radius + 1)
        phi_x = np.exp(-0.5 / sigma2 * x ** 2)
        phi_x = phi_x / phi_x.sum()

        return phi_x


    def scale_mat(self, mat, scale_x, scale_y, scale_z):
        scale_mat = torch.FloatTensor([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, scale_z]]).cuda()
        mat = torch.matmul(scale_mat, mat)
        return mat

    def shear_mat(self, mat, shear_xy, shear_xz, shear_yx, shear_yz, shear_zx, shear_zy):
        shear_mat = torch.FloatTensor([[1, np.tan(shear_xy), np.tan(shear_xz)], [np.tan(shear_yx), 1, np.tan(shear_yz)],
                     [np.tan(shear_zx), np.tan(shear_zy), 1]]).cuda()
        mat = torch.matmul(shear_mat, mat)
        return mat

    def translate_mat(self, mat, trans_x, trans_y, trans_z):
        trans = torch.FloatTensor([trans_x, trans_y, trans_z]).cuda()
        trans = trans[:, np.newaxis]
        mat = torch.cat([mat, trans], dim=-1)
        return mat


class AppearanceTransform(object):
    def __init__(self, local_rate=0.8, nonlinear_rate=0.9, paint_rate=0.9, inpaint_rate=0.2, is_local=True, is_nonlinear=True, is_in_painting=True):
        self.is_local = is_local
        self.is_nonlinear = is_nonlinear
        self.is_in_painting = is_in_painting
        self.local_rate = local_rate
        self.nonlinear_rate = nonlinear_rate

        self.paint_rate = paint_rate
        self.inpaint_rate = inpaint_rate


    def rand_aug(self, data):
        if self.is_local:
            data = self.local_pixel_shuffling(data, prob=self.local_rate)
        if self.is_nonlinear:
            data = self.nonlinear_transformation(data, self.nonlinear_rate)
        if self.is_in_painting:
            data = self.image_in_painting(data)
        data = data.astype(np.float32)
        return data


    def bernstein_poly(self, i, n, t):
        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, nTimes=1000):
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        return xvals, yvals

    def nonlinear_transformation(self, x, prob=0.5):
        if np.random.random() >= prob:
            return x
        points = [[0, 0], [np.random.random(), np.random.random()], [np.random.random(), np.random.random()], [1, 1]]

        xvals, yvals = self.bezier_curve(points, nTimes=100000)

        xvals, yvals = np.sort(xvals), np.sort(yvals)
        nonlinear_x = np.interp(x, xvals, yvals)
        return nonlinear_x

    def local_pixel_shuffling(self, x, prob=0.5):
        if np.random.random() >= prob:
            return x
        image_temp = x.copy()
        orig_image = x.copy()
        _, img_rows, img_cols, img_deps = x.shape
        num_block = 5000
        block_noise_size_x = int(img_rows // 20)
        block_noise_size_y = int(img_cols // 20)
        block_noise_size_z = int(img_deps // 20)
        noise_x = np.random.randint(low=img_rows - block_noise_size_x, size=num_block)
        noise_y = np.random.randint(low=img_cols - block_noise_size_y, size=num_block)
        noise_z = np.random.randint(low=img_deps - block_noise_size_z, size=num_block)
        window=[orig_image[:, noise_x[i]:noise_x[i] + block_noise_size_x, noise_y[i]:noise_y[i] + block_noise_size_y,
                     noise_z[i]:noise_z[i] + block_noise_size_z,] for i in range(num_block)]
        window = np.concatenate(window, axis=0)
        window = window.reshape(num_block, -1)
        np.random.shuffle(window.T)
        window = window.reshape((num_block, block_noise_size_x,
                                 block_noise_size_y,
                                 block_noise_size_z))
        for i in range(num_block):
            image_temp[0, noise_x[i]:noise_x[i] + block_noise_size_x,
            noise_y[i]:noise_y[i] + block_noise_size_y,
            noise_z[i]:noise_z[i] + block_noise_size_z] = window[i]
        local_shuffling_x = image_temp

        return local_shuffling_x

    def image_in_painting(self, x):
        _, img_rows, img_cols, img_deps = x.shape
        cnt = 30
        while cnt > 0 and np.random.random() < 0.95:
            block_noise_size_x = np.random.randint(img_rows // 10, img_rows // 5)
            block_noise_size_y = np.random.randint(img_cols // 10, img_cols // 5)
            block_noise_size_z = np.random.randint(img_deps // 10, img_deps // 5)
            noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = np.random.rand(block_noise_size_x,
                                                                   block_noise_size_y,
                                                                   block_noise_size_z, ) * 1.0
            cnt -= 1
        return x

    def image_out_painting(self, x):
        _, img_rows, img_cols, img_deps = x.shape
        image_temp = x.copy()
        x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
        block_noise_size_x = img_rows - np.random.randint(3 * img_rows // 7, 4 * img_rows // 7)
        block_noise_size_y = img_cols - np.random.randint(3 * img_cols // 7, 4 * img_cols // 7)
        block_noise_size_z = img_deps - np.random.randint(3 * img_deps // 7, 4 * img_deps // 7)
        noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
        noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
        noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
        x[:,
        noise_x:noise_x + block_noise_size_x,
        noise_y:noise_y + block_noise_size_y,
        noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                noise_y:noise_y + block_noise_size_y,
                                                noise_z:noise_z + block_noise_size_z]
        cnt = 4
        while cnt > 0 and np.random.random() < 0.95:
            block_noise_size_x = img_rows - np.random.randint(3 * img_rows // 7, 4 * img_rows // 7)
            block_noise_size_y = img_cols - np.random.randint(3 * img_cols // 7, 4 * img_cols // 7)
            block_noise_size_z = img_deps - np.random.randint(3 * img_deps // 7, 4 * img_deps // 7)
            noise_x = np.random.randint(3, img_rows - block_noise_size_x - 3)
            noise_y = np.random.randint(3, img_cols - block_noise_size_y - 3)
            noise_z = np.random.randint(3, img_deps - block_noise_size_z - 3)
            x[:,
            noise_x:noise_x + block_noise_size_x,
            noise_y:noise_y + block_noise_size_y,
            noise_z:noise_z + block_noise_size_z] = image_temp[:, noise_x:noise_x + block_noise_size_x,
                                                    noise_y:noise_y + block_noise_size_y,
                                                    noise_z:noise_z + block_noise_size_z]
            cnt -= 1
        return x
