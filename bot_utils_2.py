# New bot utils for new models

import os
import requests

from fashion_clip.fashion_clip import FashionCLIP, FCLIPDataset
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation

import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes

# FashionCLIP
fclip = FashionCLIP('fashion-clip')

# Segmentation model
extractor = AutoFeatureExtractor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

def segment_image(img):
    inputs = extractor(images=img, return_tensors="pt")

    with torch.no_grad(): outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=img.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    mask = upsampled_logits.argmax(dim=1)[0]
    mask = relevant_map(mask)
    return mask

def parse_image(img):
    # Segment image
    mask = segment_image(img)

    obj_ids = mask.unique()
    masks = torch.BoolTensor(mask == obj_ids[:,None, None])
    boxes = masks_to_boxes(masks)

    subims = []
    subids = []

    resize = transforms.Resize((1500,1000))
    to_image = transforms.ToPILImage()

    image_tensor = resize(transforms.ToTensor()(img))

    for obind, obid in enumerate(obj_ids):
        if obid != 0:
            temp = image_tensor.clone()
            temp[:,mask!=obid] = 0 # Set everything but current clothing item to 0

            xmin = int(boxes[obind, 0])
            ymin = int(boxes[obind, 1])
            xmax =int(boxes[obind, 2])
            ymax = int(boxes[obind, 3])

            subim = temp[:, ymin:ymax, xmin:xmax]

            if debug == True:
                plt.imshow(subim.squeeze().permute(1,2,0))
                plt.title(f'Only {key_map[obid.item()]}')
                plt.show()
        
            subim = to_image(subim)
            subims.append(subim) # I think we need them in image format for Clip
            subids.append(obid)

    return subims, subids

def embed_image(img):
    # Take a PIL image and return FashionClip embedding
    image_embedding = fclip.encode_images([img], 1)
    # Normalize for dot product later
    image_embedding = image_embedding/np.linalg.norm(image_embedding, ord=2, axis=-1, keepdims=True)

    return image_embedding

def add_item_to_database(img, item = None, name = '', price = 100):
    # Segment an uploaded image and add it to the database of images
    # There are a few ways to do this, depending on if we want the entire image to be added, or take the Mode, etc
    # For now we use the old version which takes the mode
    mask = segment_image(img)

    obj_ids = mask.unique().tolist()[1:]

    print(mask.shape, obj_ids)
    masks = torch.BoolTensor(mask == obj_ids[:,None, None])
    boxes = masks_to_boxes(masks)

    value_counts = torch.bincount(torch.FloatTensor(mask).int().flatten())[1:].tolist()
    code, _ = max(list(enumerate(value_counts)), key=lambda x: x[1])
    #code = code+1 # this is sketchy and should probably be reconfigured. we want the max OTHER THAN background
    
    temp = image_tensor.clone()
    temp[:,:,mask!=code] = 0
    
    for obind, obid in enumerate(obj_ids):
        if obid != code:
            pass
        else:
            xmin = int(boxes[obind, 0])
            ymin = int(boxes[obind, 1])
            xmax =int(boxes[obind, 2])
            ymax = int(boxes[obind, 3])

    temp = temp[:,:, ymin:ymax, xmin:xmax]

    # Centroid should be given by augmentation+embedding on extracted item
    centroid = emb_centroid(temp.squeeze())
    
    #plt.imshow((temp.squeeze().permute(1,2,0)+1)/2)
    #plt.show()
    
    # Look in "database" directory and find the number of items:
    trymkdir(f'database/{code}/') # Make sure database directory exists
    dblist = os.listdir(f'database/{code}/') # Go to subdirectory of item type
    item_id = len(dblist)
    
    # Embed the CROPPED SUB image
    img_emb = embed_tensor(temp)
    print('This item emebdded to', img_emb.shape)

    # Make the database directory for this item under the code subdir
    item_path = f'database/{code}/{item_id}/'
    trymkdir(item_path)

    torch.save(img, f'{item_path}/img.pt')         # Original image PIL
    torch.save(temp, f'{item_path}/temp.pt')       # Scanned item cutout
    torch.save(img_emb, f'{item_path}/img_emb.pt') # Embedding of image
    torch.save(name, f'{item_path}/name.pt')       # Provided Name
    torch.save(price, f'{item_path}/price.pt')     # Provided Price

    return code, img, temp
    

def relevant_map(mask):
    # make the mask relevant by merging features
    # let's just have this return a mask with only clothing
    
    mask[mask == 10] = 9 # Right shoes become left shoes
    
    # Remove then following:
    mask[mask == 11] = 0 # face
    mask[mask == 2] = 0 # hair
    mask[mask == 14] = 0 # left arm
    mask[mask == 12] = 0 # left leg
    mask[mask == 15] = 0 # right arm
    mask[mask == 13] = 0 # right leg
    return mask

old_map = {
   "Background": 0,
   "Bag": 16,
   "Belt": 8,
   "Dress": 7,
   "Face": 11,
   "Hair": 2,
   "Hat": 1,
   "Left-arm": 14,
   "Left-leg": 12,
   "Left-shoe": 9,
   "Pants": 6,
   "Right-arm": 15,
   "Right-leg": 13,
   "Right-shoe": 10,
   "Scarf": 17,
   "Skirt": 5,
   "Sunglasses": 3,
   "Upper-clothes": 4
} # we only need to process through some of these keys - and can merge shoes, etc?

key_map = {v: k for k, v in old_map.items()}