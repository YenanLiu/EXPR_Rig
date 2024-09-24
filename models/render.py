
import torch.nn as nn
import math

import torch 

class GlobalGeneratorForVal(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, output_size=256, n_blocks=6, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGeneratorForVal, self).__init__()        
        activation = nn.ReLU(True)        

        lowest_res = 4
        n_downsampling = int(math.log2(output_size//lowest_res))

        scale = 16
        mult = min(2**n_downsampling, scale)

        conv_in = nn.Sequential(
            nn.ConvTranspose2d(input_nc, ngf * mult, kernel_size=4, stride=1, padding=0, bias=False),
            norm_layer(ngf * mult),
            activation
        )
        model = [conv_in]
        ### resnet blocks
        # mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample
        last_ch =  ngf * mult   
        for i in range(n_downsampling):
            cur_ch = ngf * min(2**(n_downsampling - i - 1), scale)
            model += [nn.ConvTranspose2d(last_ch, cur_ch, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(cur_ch), activation]
            last_ch = cur_ch

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
    
    def train(self, mode: bool):
        """ avoid pytorch lighting auto set trian mode """
        # super().eval()
        return super().train(False)

    def forward(self, input):
        out = self.model(input) 

        return out

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

def init_decoder(state_dict_path):
    decoder = GlobalGeneratorForVal(139, 3, output_size=256, n_blocks=6, norm_layer=nn.BatchNorm2d)
    state_dict = torch.load(state_dict_path)
    decoder.load_state_dict(state_dict, strict=True)
    decoder.eval()
    for params in decoder.parameters():
        params.requires_grad = False
    return decoder
    
# # inference
# recons_img = decoder(rigs.unsqueeze(-1).unsqueeze(-1))
# recons_img = (recons_img+1)/2


if __name__ == "__main__":
    state_dict_path = "/project/_expr_train/rig/pre_pth/rig_pretrain.pth"
    rigs = torch.randn((4, 139,1,1))
    decoder = init_decoder(state_dict_path)
    out = decoder(rigs)
    print(out.shape) # s[4, 3, 256, 256]
    print(torch.max(out))
    print(torch.min(out))