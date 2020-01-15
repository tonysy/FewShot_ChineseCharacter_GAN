import os.path
from pccgan.data.base_dataset import BaseDataset, get_params, get_transform
from pccgan.data.image_folder import make_dataset
from PIL import Image
import torch

class FontAlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        self.cat_ids = list(set([int(os.path.basename(item).strip().split('_')[0]) for item in self.AB_paths]))
        self.cat_contiugous_ids = {k: i for i, k in enumerate(self.cat_ids)}

        # 57 * 512 tensor
        # tensor_path = os.path.join(opt.checkpoints_dir, "mean_fc_feat.pt")
        tensor_path = "./cat_embedding_train/mean_fc_feat.pt"
        assert os.path.exists(tensor_path), "{} not exists!".format(tensor_path)
        self.cat_embeding_list = torch.load(tensor_path,map_location='cpu' )
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index - - a random integer for data indexing
        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)
        
        cls_label = self.cat_contiugous_ids[int(os.path.basename(AB_path).strip().split('_')[0])]
        cat_embedding_tensor = self.cat_embeding_list[cls_label]

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path,'cls_label':cls_label, "cat_emb":cat_embedding_tensor}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)