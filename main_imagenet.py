import os
import random
import argparse
import yaml
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

from datasets.imagenet import ImageNet
import clip
from utils import *

from blip.models.blip import blip_feature_extractor

def get_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', help='settings of Tip-Adapter in yaml format')
    parser.add_argument('--use_blip', default=False, action="store_true", help='whether to use blip model')
    args = parser.parse_args()

    return args


def run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights):
    
    # Zero-shot CLIP
    clip_logits = 100. * test_features[:, :1024] @ clip_weights
    acc = cls_acc(clip_logits, test_labels)
    print("\n**** Zero-shot CLIP's test accuracy: {:.2f}. ****\n".format(acc))

    # Tip-Adapter
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    
    affinity = test_features @ cache_keys
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    
    tip_logits = clip_logits + cache_logits * alpha
    acc = cls_acc(tip_logits, test_labels)
    print("**** Tip-Adapter's test accuracy: {:.2f}. ****\n".format(acc))

    # Search Hyperparameters
    _ = search_hp(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)


def run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F, blip_model=None):
    
    # Enable the cached keys to be learnable
    adapter = nn.Linear(cache_keys.shape[0], cache_keys.shape[1], bias=False).to(clip_model.dtype).cuda()
    adapter.weight = nn.Parameter(cache_keys.t())
    
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=cfg['lr'], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg['train_epoch'] * len(train_loader_F))
    
    beta, alpha = cfg['init_beta'], cfg['init_alpha']
    best_acc, best_epoch = 0.0, 0

    for train_idx in range(cfg['train_epoch']):
        # Train
        adapter.train()
        correct_samples, all_samples = 0, 0
        loss_list = []
        print('Train Epoch: {:} / {:}'.format(train_idx, cfg['train_epoch']))

        for i, (images, target) in enumerate(tqdm(train_loader_F)):
            images, target = images.cuda(), target.cuda()
            with torch.no_grad():
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if blip_model:
                    blip_img_features = blip_model(images, caption=None, mode='image')[:,0,:].squeeze().half()
                    blip_img_features /= blip_img_features.norm(dim=-1, keepdim=True)
                    image_features = torch.cat([image_features, blip_img_features], dim=1)

            affinity = adapter(image_features)
            cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
            clip_logits = 100. * image_features[:, :1024] @ clip_weights
            tip_logits = clip_logits + cache_logits * alpha

            loss = F.cross_entropy(tip_logits, target)

            acc = cls_acc(tip_logits, target)
            correct_samples += acc / 100 * len(tip_logits)
            all_samples += len(tip_logits)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
        print('LR: {:.6f}, Acc: {:.4f} ({:}/{:}), Loss: {:.4f}'.format(current_lr, correct_samples / all_samples, correct_samples, all_samples, sum(loss_list)/len(loss_list)))

        # Eval
        adapter.eval()

        affinity = adapter(test_features)
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        clip_logits = 100. * test_features[:, :1024] @ clip_weights
        tip_logits = clip_logits + cache_logits * alpha
        acc = cls_acc(tip_logits, test_labels)

        print("**** Tip-Adapter-F's test accuracy: {:.2f}. ****\n".format(acc))
        if acc > best_acc:
            best_acc = acc
            best_epoch = train_idx
            torch.save(adapter.weight, cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    
    adapter.weight = torch.load(cfg['cache_dir'] + "/best_F_" + str(cfg['shots']) + "shots.pt")
    print(f"**** After fine-tuning, Tip-Adapter-F's best test accuracy: {best_acc:.2f}, at epoch: {best_epoch}. ****\n")

    # Search Hyperparameters
    _ = search_hp(cfg, affinity, cache_values, test_features, test_labels, clip_weights, adapter=adapter)


def main():

    for shots in [8, 4, 2, 1]:
        print(f'---------------------- {shots} shots -----------------------------')

        # Load config file
        args = get_arguments()
        assert (os.path.exists(args.config))
        
        cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

        cache_dir = os.path.join('./caches', cfg['dataset'])
        os.makedirs(cache_dir, exist_ok=True)
        cfg['cache_dir'] = cache_dir

        print("\nRunning configs.")
        print(cfg, "\n")

        # CLIP
        clip_model, preprocess = clip.load(cfg['backbone'])
        clip_model.eval()

        # ImageNet dataset
        random.seed(1)
        torch.manual_seed(1)
        
        print("Preparing ImageNet dataset.")
        imagenet = ImageNet(cfg['root_path'], shots, preprocess)

        test_loader = torch.utils.data.DataLoader(imagenet.test, batch_size=64, num_workers=8, shuffle=False)

        train_loader_cache = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=False)
        train_loader_F = torch.utils.data.DataLoader(imagenet.train, batch_size=256, num_workers=8, shuffle=True)

        # blip model initialization
        blip_model = None
        blip_weights = None
        if args.use_blip:
            model_url = './blip/model_large.pth'
            blip_model = blip_feature_extractor(pretrained=model_url, image_size=224, vit='large')
            blip_model.eval()
            blip_model = blip_model.cuda()
            # blip_weights = None
            # blip_weights = blip_classifier(dataset.classnames, dataset.template, blip_model)

        # Textual features
        print("Getting textual features as CLIP's classifier.")
        clip_weights = clip_classifier(imagenet.classnames, imagenet.template, clip_model, blip_model=None)

        # Construct the cache model by few-shot training set
        print("\nConstructing cache model by few-shot visual features and labels.")
        cache_keys, cache_values = build_cache_model(cfg, clip_model, train_loader_cache, blip_model=blip_model)

        # Pre-load test features
        print("\nLoading visual features and labels from test set.")
        test_features, test_labels = pre_load_features(cfg, "test", clip_model, test_loader, blip_model=blip_model)

        # ------------------------------------------ Tip-Adapter ------------------------------------------
        run_tip_adapter(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights)

        # ------------------------------------------ Tip-Adapter-F ------------------------------------------
        run_tip_adapter_F(cfg, cache_keys, cache_values, test_features, test_labels, clip_weights, clip_model, train_loader_F, blip_model=blip_model)
            
        print(f'---------------------- {shots} shots -----------------------------')

if __name__ == '__main__':
    main()