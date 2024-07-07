from timm_vit.vision_transformer import *
from timm_vit.vision_transformer import _cfg, VisionTransformer
from timm_vit.fair_vit import Fair_VisionTransformer
from functools import partial
import torch
import torch.nn as nn
import os
import time
from transform import *

def model_type(args = None, training = True, retrained=True, **kwargs):
    if(args.model_type == 'deit_base_patch16_224'):
        if training :
            model = Fair_VisionTransformer(
            num_classes = args.nb_classes, mask_numbers = len(args.varsigma), patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,head_masking=True, 
            fc_masking=True,classifiers=args.classifiers)
            model.default_cfg = _cfg()
            if not retrained:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                    map_location="cpu", check_hash=True
                )
                checkpoint_model = checkpoint['model']
                state_dict = model.state_dict()
                for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']: # no removing
                    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                        print(f"Removing key {k} from pretrained checkpoint")
                        del checkpoint_model[k]
                model.load_state_dict(checkpoint_model, strict=False)
            else:  # keep training
                model.load_state_dict(torch.load("models/"+args.feature+".pt")['state_dict'])

        else: # testing
            model = Fair_VisionTransformer(
            num_classes = args.nb_classes,mask_numbers = len(args.varsigma), patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,head_masking=True, 
            fc_masking=True,classifiers=args.classifiers, classifier_choose=12) # different from training
            model.LoadParams(dir="models/"+args.feature+".pt")

    # add model types here
    elif(args.model_type == 'vit_small_patch16_224'):
        if not retrained:
            model = vit_small_patch16_224(pretrained=True)
        else:
            model = VisionTransformer(patch_size=16, embed_dim=768, depth=8, num_heads=8, mlp_ratio=3., **kwargs)
            model.load_state_dict(torch.load("models/"+args.feature+".pt")['state_dict'])
    elif(args.model_type == 'vit_base_patch16_224'):
        if not retrained:
            model = vit_base_patch16_224(pretrained=True)
        else:
            model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
            model.load_state_dict(torch.load("models/"+args.feature+".pt")['state_dict'])
    else:
        raise('model type error!')
    return model

def save_checkpoint(state, filename='checkpoint.pt'):
        if not os.path.exists(os.path.dirname(filename)):
                os.makedirs(os.path.dirname(filename))
        torch.save(state, filename)

def save_model(args, model):
    meta_dict = {'time':time.localtime()}
    save_checkpoint({
    'state_dict': model.state_dict(),
    'meta_dict': meta_dict
    }, filename=os.path.join('models', args.feature+".pt"))