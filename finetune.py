import argparse
import clip
import torch
from PIL import Image, ImageFilter
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import os
import pathlib
import json
# import flickr8k_example.generation_eval_utils as generation_eval_utils
import pprint
import warnings
from packaging import version
from clipscore import extract_all_captions, extract_all_images
import logging

#Citations: 
# CSV dataset from : https://github.com/mlfoundations/open_clip/blob/6ee59e10510ec9761b8b9871b9fd1eeb8e28627d/src/training/data.py#L445
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, IterableDataset, get_worker_info
from torch.utils.data import DataLoader, RandomSampler

# additional imports
import torch.optim as optim
import pandas as pd
import torch.nn as nn

# TODO: take this out if you don't use it
# adapted from github linked above -- took out transforms as parameter, use local (class) function as preprocessing/transforms
class CsvDataset(Dataset):
    def __init__(self, input_filename, img_key, caption_key, sep="\t", tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        df = pd.read_csv(input_filename, sep=sep)

        self.images = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        logging.debug('Done loading data.')
        self.preprocess = self._transforms(224)

        self.tokenize = tokenizer
        
    # taken to be preprocessing function from clipscore.py
    def _transforms(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.preprocess(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts

class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return c_data

    def __len__(self):
        return len(self.data)
    
class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = image.rotate(45)
        #image = image.filter(ImageFilter.GaussianBlur(radius=7.5))
        image = self.preprocess(image)
        return image

    def __len__(self):
        return len(self.data)

class ImageCaptionDataset(torch.utils.data.Dataset):
    def __init__(self, image_dataset, caption_dataset):
        self.images = image_dataset
        self.captions = caption_dataset

    def __getitem__(self, idx):
        image = self.images[idx]
        caption = self.captions[idx]
        return image, caption    

    def __len__(self):
        return len(self.images)
    
# adapted from CLIPScore compute_metrics.py
# function compute_human_correlation
def load_images_and_captions(input_json, image_directory):
    data = {}
    with open(input_json) as f:
        data.update(json.load(f))
    print('Loaded {} images'.format(len(data)))
    
    images = []
    candidates = []
    
    for k, v in list(data.items()):
        for human_judgement in v['human_judgement']:
            # if np.isnan(human_judgement['rating']):
            #     print('NaN')
            #     continue
            images.append(image_directory + '/' + v['image_path'])
            # refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
            candidates.append(' '.join(human_judgement['caption'].split()))
            # human_scores.append(human_judgement['rating'])
    
    return images, candidates
    
    
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        p.grad.data = p.grad.data.float() 

def train(model, optimizer, train_dataset, preprocess):
    # num_iters = 10000
    # for i in range(num_iters):
        
    # # save the model
    # if i % 1000:
    #     with open('./fine_tuned/' + 'clip_model_' + str(i) + '_.model', 'wb') as f:
    #             torch.save(model.state_dict(), f)
    
    BATCH_SIZE = 1024
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()
    
    clip.model.convert_weights(model)
    num_epochs = 10
    for i in range(num_epochs):
        print('Starting Epoch ', i)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=BATCH_SIZE,
            drop_last=True,
        )
        for j, batch in enumerate(train_dataloader):
            print('Starting Batch ', j)
            optimizer.zero_grad()
            images,texts = batch #list_images is list of image in numpy array(np.uint8)
            #images= torch.stack([preprocess(img) for img in list_image],dim=0)
            # images= torch.stack(list_image,dim=0)
            # # texts = clip.tokenize(list_txt)
            # texts = list_txt
            
            logits_per_image, logits_per_text = model(images.to(device), texts.to(device))

            ground_truth = torch.arange(BATCH_SIZE).to(device)
            total_loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2
            total_loss.backward()
            print(total_loss.item())

            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)
            
            # Save model every epoch
            with open('./fine_tuned_every_step/' + 'clip_model_' + str(j) + '_.model', 'wb') as f:
                torch.save(model.state_dict(), f)

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # load model from clip
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    clip.model.convert_weights(model)

    
     # put model in training mode
    model.train()
    
    # lower learning rate for fine tuning, as suggested by https://github.com/openai/CLIP/issues/83
    # Freezing all but the image embedding layer
    optimizer = optim.Adam([model.visual.class_embedding], lr=5e-5,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) #Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

    print('Loading images')
    images, candidates = load_images_and_captions('flickr8k_example/flickr8k/flickr8k.json', 'flickr8k_example/flickr8k/')
    
    print('Creating Dataset')
    image_dataset = CLIPImageDataset(images)
    caption_dataset = CLIPCapDataset(candidates)
    
    train_dataset = ImageCaptionDataset(image_dataset, caption_dataset)

    print('Starting Training')
    train(model, optimizer, train_dataset, transform)
    

    
    
    