import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler
from torch.nn import init
from torch.nn import functional as F

class EncoderNetwork(nn.Module):
    def __init__(self, in_channels, num_downs, ngf=64, norm_layer=nn.BatchNorm2d):
        super(EncoderNetwork, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        

        # self.top_layer = TopDownBlock(in_channels, ngf, use_bias)

        self.top_layer = nn.Conv2d(
            in_channels, ngf, 
            kernel_size=4,stride=2,
            padding=1,bias=use_bias)

        self.encode_layer_1 = DownBlock(ngf, ngf*2, use_bias, norm_layer)
        self.encode_layer_2 = DownBlock(ngf*2, ngf*4, use_bias, norm_layer)
        self.encode_layer_3 = DownBlock(ngf*4, ngf*8, use_bias, norm_layer)
        
        # self.inter_layers = []
        self.num_downs = num_downs
        for i in range(num_downs - 5):
            self.add_module('encoder_inter_{}'.format(i), 
                                    DownBlock(ngf*8, ngf*8,use_bias, norm_layer))

        self.last_encode_layer = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                ngf*8, ngf*8,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias)
        )

    def forward(self, x):
        e1 = self.top_layer(x)
        e2 = self.encode_layer_1(e1)
        e3 = self.encode_layer_2(e2)
        e4 = self.encode_layer_3(e3)
        
        inter_encoder_outs = []
        for idx in range(self.num_downs - 5):
            if len(inter_encoder_outs) == 0:
                inter_outs = eval("self.encoder_inter_{}".format(idx))(e4)
            else:
                inter_outs = eval("self.encoder_inter_{}".format(idx))(inter_outs)

            inter_encoder_outs.append(inter_outs)
        
        final_out = self.last_encode_layer(inter_outs)

        return [e1, e2, e3, e4] + inter_encoder_outs + [final_out]

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias, norm_layer):
        super(DownBlock, self).__init__()
        
        downconv = nn.Conv2d(in_channels, out_channels, kernel_size=4,stride=2,padding=1,bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(out_channels)

        layer_list = [downrelu, downconv, downnorm]
        
        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)

class DecoderNetwork(nn.Module):
    def __init__(self, out_channels, num_ups, cat_embedding_nc=256, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(DecoderNetwork, self).__init__()
        # num_ups should == num_downs
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # import pdb; pdb.set_trace()
        self.cat_aware_layer = nn.Sequential(
            nn.Conv2d(ngf*8+cat_embedding_nc, 
                    ngf*8, kernel_size=1,
                    stride=1,padding=0,bias=use_bias),
            # nn.ReLU(True),
            norm_layer(ngf*8),
        )

        self.first_decoder_layer = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4,
            stride=2,padding=1,bias=use_bias),
            norm_layer(ngf*8),
        )

        self.inter_layers = []

        self.num_ups = num_ups
        for i in range(num_ups - 5):
            # self.inter_layers.append(
                # UpBlock(ngf*16, ngf*8, use_bias, norm_layer, use_dropout))
            self.add_module('decoder_inter_{}'.format(i), UpBlock(ngf*16, ngf*8, use_bias, norm_layer, use_dropout))

        self.decoder_up_1 = UpBlock(ngf*16, ngf*4, use_bias, norm_layer)
        self.decoder_up_2 = UpBlock(ngf*8, ngf*2, use_bias, norm_layer)
        self.decoder_up_3 = UpBlock(ngf*4, ngf, use_bias, norm_layer)

        self.last_decoder_layer = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(
                ngf*2, out_channels,
                kernel_size=4,
                stride=2,
                padding=1),
            nn.Tanh(),
            )

    def forward(self, encoder_inputs, cat_embedding):
        b,c,w,h = encoder_inputs[-1].size()
        cat_embedding = cat_embedding.unsqueeze(-1).unsqueeze(-1)
        new_cat_embedding = cat_embedding.repeat(1,1,w,h)
        # import pdb; pdb.set_trace()

        cat_aware_encodings = torch.cat(
                [encoder_inputs[-1], new_cat_embedding], dim=1) 
        
        cat_aware_encodings = self.cat_aware_layer(cat_aware_encodings)
        
        d1 = self.first_decoder_layer(cat_aware_encodings)

        for i in range(self.num_ups-5):
            if i == 0:
                inputs = torch.cat([d1,encoder_inputs[-(2+i)]],dim=1) 
            else:
                inputs = torch.cat([outputs,encoder_inputs[-(2+i)]],dim=1)

            outputs = eval("self.decoder_inter_{}".format(i))(inputs)

        inputs = torch.cat([outputs, encoder_inputs[3]],dim=1) # 512*2
        outputs = self.decoder_up_1(inputs)

        inputs = torch.cat([outputs, encoder_inputs[2]],dim=1) # 256*2
        outputs = self.decoder_up_2(inputs)

        inputs = torch.cat([outputs, encoder_inputs[1]],dim=1) # 128*2
        outputs = self.decoder_up_3(inputs)

        inputs = torch.cat([outputs, encoder_inputs[0]],dim=1) # 64*2
        outputs = self.last_decoder_layer(inputs)

        return outputs

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_bias, norm_layer, use_dropout=False):
        super(UpBlock, self).__init__()
        
        
        upconv = nn.ConvTranspose2d(
            in_channels, out_channels, 
            kernel_size=4,stride=2,
            padding=1,bias=use_bias)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(out_channels)
        
        layer_list = [uprelu, upconv, upnorm]

        if use_dropout:
            layer_list.append(nn.Dropout(0.5))
        
        self.model = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.model(x)

class FontUNetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, cat_embedding_nc=256, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(FontUNetGenerator, self).__init__()
        self.encoder = EncoderNetwork(input_nc,num_downs,ngf,norm_layer)
        self.decoder = DecoderNetwork(
            out_channels=output_nc,
            num_ups=num_downs,
            cat_embedding_nc=cat_embedding_nc,
            ngf=ngf,norm_layer=norm_layer,
            use_dropout=use_dropout
        )
            # output_nc,num_downs,cat_embedding_nc,ngf,norm_layer,use_dropout)

    def forward(self, x, cat_embedding):
        encoder_list = self.encoder(x)
        output = self.decoder(encoder_list, cat_embedding)

        return output, encoder_list


# if __name__ == "__main__":
#     network = FontUNetGenerator(
#         input_nc=3,
#         output_nc=3,num_downs=7,cat_embedding_nc=256
#     ).cuda()

#     x = torch.randn(2,3,256,256).cuda()
#     cat_embedding = torch.randn(2,256,1,1).cuda()

#     output = network(x, cat_embedding)
#     import pdb; pdb.set_trace()


    # network = FontNLayerDiscriminator(
    #     input_nc=3,
    #     ndf=64,n_layers=3,
    #     # output_nc=3,num_downs=7,cat_embedding_nc=256
    # )

    # x = torch.randn(2,3,256,256)
    # # cat_embedding = torch.randn(2,256,1,1)

    # output = network(x) #@, cat_embedding)
    # import pdb; pdb.set_trace()