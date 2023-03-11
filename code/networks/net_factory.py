from networks.unet import UNet, UNet_CCT
from networks.ynet import YNet_general


def net_factory(net_type="unet", in_chns=3, class_num=2):
    if net_type == "unet":
        net = UNet(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "unet_cct":
        net = UNet_CCT(in_chns=in_chns, class_num=class_num).cuda()
    elif net_type == "ynet_ffc":
        net = YNet_general(in_chns, class_num).cuda()
    else:
        net = None
    return net
