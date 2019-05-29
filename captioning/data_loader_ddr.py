import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab_ddr import Vocabulary
#from pycocotools.coco import COCO
import pandas as pd
import json

class DDRDataset(data.Dataset):
    """DDR Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, csv, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            csv: DDR annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root 
        self.vocab = vocab
        self.transform = transform
        
        #We have pruned the data to use fewer songs (scores of 5-9)
        all_data = pd.read_csv(csv, index_col=False)
        all_data['json'] = all_data.apply(lambda x: x.to_json(), axis=1)
        all_data = [json.loads(row) for row in all_data['json']]
        
        training_data = []
        
        for row in all_data:
            
            #Change semicolon to space
            row['text'] = row['text'].split(';')
            row['text'] = ' '.join(row['text'])
            row['text'] = str(row['text'])
            
            training_data.append(row)
        
        self.ddr = training_data

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        ddr = self.ddr
        vocab = self.vocab
        #ann_id = self.ids[index]
        caption = ddr[index]['text']
        #img_id = coco.anns[ann_id]['image_id']
        path = ddr[index]['spectrogram_path2'].split('\\')[1]

        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(caption)
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ddr)
    

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, csv, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    ddr = DDRDataset(root=root,
                       csv=csv,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for DDR dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=ddr, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader