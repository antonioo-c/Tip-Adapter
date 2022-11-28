from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn

import clip


def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = float(correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
    acc = 100 * acc / target.shape[0]
    return acc

def clip_classifier(classnames, template, clip_model, blip_model=None):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts0 = [t.format(classname) for t in template]
            texts = clip.tokenize(texts0).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()

            if blip_model is not None:
                blip_embeddings = blip_model(image=None, caption=texts, mode='text')[:,0,:].squeeze(1).half()
                blip_embeddings /= blip_embeddings.norm(dim=-1, keepdim=True)
                blip_embedding = blip_embeddings.mean(dim=0)
                blip_embedding /= blip_embedding.norm()
                class_embedding = torch.cat((class_embedding, blip_embedding), dim=-1)
                # print(class_embedding.shape)

            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def clip_classifier2(classnames, template, clip_model):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).cuda()
            # prompt ensemble for ImageNet
            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).cuda()
    return clip_weights

def blip_classifier(classnames, template, blip_model):
    with torch.no_grad():
        blip_weights = []
        print('blip--------------------------')

        for classname in classnames:
            # Tokenize the prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            # prompt ensemble for ImageNet
            class_embeddings = blip_model(image=None, caption=texts, mode='text')[:,0,:].squeeze(1).half()              
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            blip_weights.append(class_embedding)

        blip_weights = torch.stack(blip_weights, dim=1).cuda()
    return blip_weights


def build_cache_model(cfg, clip_model, train_loader_cache, blip_model=None):

    if cfg['load_cache'] == False:    
        cache_keys = []
        cache_values = []

        with torch.no_grad():
            # Data augmentation for the cache model
            for augment_idx in range(cfg['augment_epoch']):
                train_features = []

                print('Augment Epoch: {:} / {:}'.format(augment_idx, cfg['augment_epoch']))
                for i, (images, target) in enumerate(tqdm(train_loader_cache)):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    if blip_model:
                        blip_img_features = blip_model(images, caption=None, mode='image')[:,0,:].squeeze().half()
                        # image_features = 0.7 * image_features + 0.3 * blip_img_features # sum fusion
                        # print(image_features.shape)
                        # print(blip_img_features.shape)
                        image_features = torch.cat([image_features, blip_img_features], dim=1) # concat fusion
                    train_features.append(image_features)
                    if augment_idx == 0:
                        target = target.cuda()
                        cache_values.append(target)
                cache_keys.append(torch.cat(train_features, dim=0).unsqueeze(0))
            
        cache_keys = torch.cat(cache_keys, dim=0).mean(dim=0)
        cache_keys /= cache_keys.norm(dim=-1, keepdim=True)
        cache_keys = cache_keys.permute(1, 0)
        cache_values = F.one_hot(torch.cat(cache_values, dim=0)).half()

        torch.save(cache_keys, cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        torch.save(cache_values, cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    else:
        cache_keys = torch.load(cfg['cache_dir'] + '/keys_' + str(cfg['shots']) + "shots.pt")
        cache_values = torch.load(cfg['cache_dir'] + '/values_' + str(cfg['shots']) + "shots.pt")

    return cache_keys, cache_values


def pre_load_features(cfg, split, clip_model, loader, blip_model=None):

    if cfg['load_pre_feat'] == False:
        features, labels = [], []

        with torch.no_grad():
            for i, (images, target) in enumerate(tqdm(loader)):
                images, target = images.cuda(), target.cuda()
                image_features = clip_model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                if blip_model:
                    blip_img_features = blip_model(images, caption=None, mode='image')[:,0,:].squeeze().half()
                    blip_img_features /= blip_img_features.norm(dim=-1, keepdim=True)
                    image_features = torch.cat([image_features, blip_img_features], dim=1)
                features.append(image_features)
                labels.append(target)

        features, labels = torch.cat(features), torch.cat(labels)

        torch.save(features, cfg['cache_dir'] + "/" + split + "_f.pt")
        torch.save(labels, cfg['cache_dir'] + "/" + split + "_l.pt")
   
    else:
        features = torch.load(cfg['cache_dir'] + "/" + split + "_f.pt")
        labels = torch.load(cfg['cache_dir'] + "/" + split + "_l.pt")
    
    return features, labels


def search_hp(cfg, cache_keys, cache_values, features, labels, clip_weights, adapter=None, blip_weights=None):

    if cfg['search_hp'] == True:
    
        beta_list = [i * (cfg['search_scale'][0] - 0.1) / cfg['search_step'][0] + 0.1 for i in range(cfg['search_step'][0])]
        alpha_list = [i * (cfg['search_scale'][1] - 0.1) / cfg['search_step'][1] + 0.1 for i in range(cfg['search_step'][1])]
        if blip_weights is not None:
            gamma_list = [i * (cfg['search_scale'][2] - 0.1) / cfg['search_step'][2] + 0.1 for i in range(cfg['search_step'][2])]
        else:
            gamma_list = [1]

        best_acc = 0
        best_beta, best_alpha, best_gamma = 0, 0, 0

        for beta in beta_list:
            for alpha in alpha_list:
                for gamma in gamma_list:
                    if adapter:
                        affinity = adapter(features)
                    else:
                        affinity = features @ cache_keys

                    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
                    clip_logits = 100. * features[:, :1024] @ clip_weights
                    if blip_weights is not None:
                        blip_logits = 100. * features[:, 1024:] @ blip_weights
                        tip_logits = (blip_logits * (1-gamma) + clip_logits * gamma) + cache_logits * alpha
                    else:
                        tip_logits = clip_logits + cache_logits * alpha
                    acc = cls_acc(tip_logits, labels)
                
                    if acc > best_acc:
                        print("New best setting, beta: {:.2f}, alpha: {:.2f}, gamma: {:.2f}; accuracy: {:.2f}".format(beta, alpha, gamma, acc))
                        best_acc = acc
                        best_beta = beta
                        best_alpha = alpha
                        best_gamma = gamma

        print("\nAfter searching, the best accuarcy: {:.2f}.\n".format(best_acc))

    return best_beta, best_alpha, best_gamma
