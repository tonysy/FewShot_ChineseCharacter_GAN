import os
import torch
from torch import nn
import PIL
import numpy as np
from PIL import Image, ImageFont
from PIL import ImageDraw
from pccgan.models.classification_model import resnet18
from pccgan.models import create_model
from pccgan.options.inference_option import InferenceOptions
from pccgan.util.processing_helper import draw_single_char_by_font
from pccgan.data.base_dataset import get_params, get_transform
from pccgan.util import util
import numpy as np
import os
import sys
import ntpath
import time
from tqdm import tqdm

opt = InferenceOptions().parse()  # get test options
# hard-code some parameters for test
opt.num_threads = 0   # test code only supports num_threads = 1
opt.batch_size = 1    # test code only supports batch_size = 1
opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.


class Inferener(object):
    """
    Inference for Demo.

    """
    def __init__(self, option):
        super(Inferener, self).__init__()
        self.option = option
        self.init_category_embedding(option.cat_emb_path)
        self.build_and_load_model(option)
        self.load_ckpt()
        # import pdb; pdb.set_trace()

    def init_category_embedding(self,cat_emb_path):
        self.cat_embedding_dict = {}
        cat_emb_trained = torch.load(cat_emb_path, map_location="cpu")

        for i in range(cat_emb_trained.size(0)):
            self.cat_embedding_dict[i] = cat_emb_trained[i]
    
    def build_and_load_model(self,option):
        self.model = create_model(option)
        self.model.setup(option)
        self.model.eval()
    
    def load_ckpt(self):
        """Load all the networks from the disk.
        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in ['G']:
            if isinstance(name, str):
                # load_filename = '%s_net_%s.pth' % (epoch, name)
                # load_path = os.path.join(self.save_dir, load_filename)
                # ckpt_path = 
                net = getattr(self.model, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % self.option.ckpt_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(self.option.ckpt_path, map_location=str(self.model.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                # for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                #     self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def transfer_imgs(self, style_idx, char_id, save=False, save_name=""):
        """
        Generate the style transfered image.
        
        Args:
            style_idx:
            char_id:
        """

        # Generate source image
        src_img = self.generate_source_img(char_id, font='SimSun')
        # Get font style embedding
        font_cat_embedding = self.cat_embedding_dict[style_idx]

        # Transform the source image and embedding to construct input for model
        inputs_data = self.data_process(src_img, font_cat_embedding, style_idx)

        self.model.set_input_inference(inputs_data)
        self.model.test()
        # output_img = self.model.fake_B
        if save:
            self.save_img(self.model.fake_B,'infered_{}_fake_B.png'.format(save_name))
            self.save_img(self.src_img,'infered_{}_real_A.png'.format(save_name))
        # return output_img
        
    def data_process(self,src_img, cat_embedding, style_idx):

        # src img transform
        src_img = src_img.convert('RGB')
        transform_params = get_params(self.option, src_img.size)
        img_transform = get_transform(self.option, transform_params, grayscale=False)
        src_img = img_transform(src_img)

        src_img = src_img.unsqueeze(0).cuda()
        cat_lbl = style_idx
        # cat_lbl = torch.tensor(style_idx).view(1,-1).cuda()
        cat_embedding = cat_embedding.unsqueeze(0).cuda()
        self.src_img = src_img

        return {"B": src_img, 'cls_label':cat_lbl, "cat_emb":cat_embedding}
        
    def generate_source_img(self, char_id, font='SimSun', font_root='../datasets/font_process/font_ttf', canvas_size=256,char_size=220):
        src_font_file = os.path.join(font_root, font+'.ttf')
        assert os.path.exists(src_font_file)
        src_font = ImageFont.truetype(src_font_file, size=char_size)

        char_img = draw_single_char_by_font(char_id, src_font, canvas_size, char_size)
        # char_img.save("input_src_temp.jpg",mode='F')
        return char_img

    def save_img(self, img_tensor, img_save_path,aspect_ratio=1.0, width=256):
        img = util.tensor2im(img_tensor)
        util.save_image(img, img_save_path, aspect_ratio=aspect_ratio)


    def add_new_cats(self, input_imgs, style_key='new_user1'):
        classification_net = resnet18(num_classes=57)
        cls_net = nn.DataParallel( classification_net).cuda()
        state_dict = torch.load(self.option.cls_ckpt, map_location='cpu')
        cls_net.load_state_dict(state_dict['state_dict'])

        fc_feats = torch.zeros(512).cuda()
        # fc_feats_num = dict(zip(list(range(57)), [0]*57))
        fc_feats_num = torch.zeros(1).cuda()

        input_imgs = self.cls_process_imgs(input_imgs)

        for img in input_imgs:
            tmp, last_fc_feats = cls_net(img)
            fc_feats += last_fc_feats.squeeze()#[i]
            fc_feats_num += 1

        import pdb; pdb.set_trace()
        mean_feat = fc_feats / fc_feats_num #.view(-1, 1)
        
        self.cat_embedding_dict[style_key] = mean_feat
        print("Add new font style into cat embedding dict")


    def cls_process_imgs(self, imgs):
        output_list = []
        for im in imgs:
            im = im.convert('RGB')

            transform_params = get_params(self.option, im.size)
            img_transform = get_transform(self.option, transform_params, grayscale=False)
            im = img_transform(im)

            im = im.unsqueeze(0).cuda()
            output_list.append(im)
        
        assert len(output_list) == len(imgs)

        return output_list

# char_list = ['敬']
# char_list = ['谷歌机器学习冬令营']
def test_with_specified_chars(style_idx=0,char_list = '谷歌机器学习冬令营',direction='horizontal',prefix=''):
    
    # style_idx = 0

    out_fake_imgs = []
    real_imgs = []
    for char in char_list:
        # out_img = 
        true_inferencer.transfer_imgs(style_idx, char)#, save=True)
        # import pdb; pdb.set_trace()

        out_fake_imgs.append(true_inferencer.model.fake_B)
        real_imgs.append(true_inferencer.src_img)

    # direction = 'horizontal' # horizontal

    if direction == 'vertical':
        out_img = torch.cat(out_fake_imgs, dim=2)
    elif direction == 'horizontal':
        out_img = torch.cat(out_fake_imgs, dim=3)

    true_inferencer.save_img(out_img, '{}_style_{}_infered.png'.format(prefix, style_idx))

if __name__ == "__main__":
    true_inferencer = Inferener(opt)



    """"
    Add new font style
    """"
    # folder = '../datasets/Calligraphy_Processed/'
    
    # wangxizi_imgs = os.listdir(folder)
    # wangxizi_imgs = [item for item in wangxizi_imgs if '.jpg' in item]
    # # import pdb; pdb.set_trace()
    # img_paths = [os.path.join(folder,item) for item in wangxizi_imgs]

    # img_readed = [Image.open(item).convert('RGB') for item in img_paths]
    
    
    # true_inferencer.add_new_cats(img_readed,style_key='user1')
    
    
    """Generate specific Characters
    """
    char_list = '人丑无怨屏幕轻闪码上有对象'
    import pdb; pdb.set_trace()
    for i in tqdm(range(57)):
        test_with_specified_chars(i, char_list,prefix='duilian_shang')


        test_with_specified_chars('user1', char_list = '上拜图灵只佑服务可用',prefix='duilian_shang_wangxizi')

        test_with_specified_chars('user1', char_list = '下跪关公但求永不宕机',prefix='duilian_xia_wangxizi')


        [test_with_specified_chars(idx, char_list = '上拜图灵只佑服务可用',prefix='duilian_shang_{}'.format(idx)) for idx in range(57)]
        [test_with_specified_chars(idx, char_list = '下跪关公但求永不宕机',prefix='duilian_xia_{}'.format(idx)) for idx in range(57)]

        [test_with_specified_chars(idx, char_list = '上拜图灵只佑服务可用',prefix='hor_duilian_shang_{}'.format(idx),direction='horizontal') for idx in range(57)]
        [test_with_specified_chars(idx, char_list = '下跪关公但求永不宕机',prefix='hor_duilian_xia_{}'.format(idx),direction='horizontal') for idx in range(57)]


        test_with_specified_chars(21, char_list = '风调码顺',prefix='duilian_hangpi_21')

        test_with_specified_chars(21, char_list = '福',prefix='duilian_fu_21')

        test_with_specified_chars(21, char_list = '上拜图灵只佑服务可用',prefix='hor_duilian_shang_21',direction='vertical')


        test_with_specified_chars(21, char_list = '下跪关公但求永不宕机',prefix='./duilian/hor_duilian_xia_21',direction='vertical')


        [test_with_specified_chars(21, char_list = zi,prefix='./duilian/hor_duilian_shang_21_{}'.format(zi),direction='vertical') for zi in ['上','拜','图','灵','只','佑','服','务','可','用']]

        [test_with_specified_chars(21, char_list = zi,prefix='./duilian/hor_duilian_xia_21_{}'.format(zi),direction='vertical') for zi in ['下','跪','关','公','但','求','永','不','宕','机']]

        [test_with_specified_chars(21, char_list = zi,prefix='./duilian/hor_heng_xia_21_{}'.format(zi),direction='vertical') for zi in ['风','调','码','顺']]