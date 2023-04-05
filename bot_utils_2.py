# New bot utils for new models

import os
import requests

from fashion_clip.fashion_clip import FashionCLIP, FCLIPDataset
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation

import matplotlib.pyplot as plt

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes

from PIL import Image, ImageDraw, ImageFont, ImageOps
import time
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# FashionCLIP
fclip = FashionCLIP('fashion-clip')

# Segmentation model
extractor = AutoFeatureExtractor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Transforms

to_image = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

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

def parse_image(img, debug = False):
    # This is being depreciated in favor of image_to_subims, mode = 'all'

    # Segment image
    mask = segment_image(img)
    #mask = resize(mask)

    obj_ids = mask.unique()
    masks = torch.BoolTensor(mask == obj_ids[:,None, None])
    boxes = masks_to_boxes(masks)

    subims = []
    subids = []

    image_tensor = to_tensor(img)
    # Make the image tensor the mask's size:
    transforms.Resize((mask.shape[0],mask.shape[1]))(image_tensor)

    for obind, obid in enumerate(obj_ids):
        if obid != 0:
            temp = image_tensor.clone()
            temp[:,mask!=obid] = 0 # Set everything but current clothing item to 0

            xmin = int(boxes[obind, 0])
            ymin = int(boxes[obind, 1])
            xmax =int(boxes[obind, 2])
            ymax = int(boxes[obind, 3])

            subim = temp[:, ymin:ymax, xmin:xmax]

            if False:
                plt.imshow(subim.squeeze().permute(1,2,0))
                plt.title(f'Only {key_map[obid.item()]}')
                plt.show()
        
            subim = to_image(subim)
            subims.append(subim) # I think we need them in image format for Clip
            subids.append(obid.item())

    return subims, subids

def embed_image(img):
    if type(img) == list:
        image_embedding = fclip.encode_images(img, len(img))
        return image_embedding.squeeze()
    else:

        # Take a PIL image and return FashionClip embedding
        image_embedding = fclip.encode_images([img], 1)
        # Normalize for dot product later (I'm not crazy about the dot product idea - id prefer MSE?)
        #image_embedding = image_embedding/np.linalg.norm(image_embedding, ord=2, axis=-1, keepdims=True)

        return image_embedding.squeeze()

def trymkdir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def image_to_subims(img, mode, debug = False):
    # mode = 'biggest' or 'all'
    # biggest returns largest subimage but mask pixel count
    # all returns list of subims, subids
    # This should replace the logic in `parse` and `database`

    image_tensor = transforms.ToTensor()(img) # Not resizing here?

    mask = segment_image(img).cpu().numpy()
    obj_ids = np.unique(mask)[1:]

    masks = torch.BoolTensor(mask == obj_ids[:,None, None])
    boxes = masks_to_boxes(masks)

    if mode == 'biggest':
        value_counts = torch.bincount(torch.FloatTensor(mask).int().flatten())[1:].tolist()
        code, _ = max(list(enumerate(value_counts)), key=lambda x: x[1])
        code = code+1 # this is sketchy and should probably be reconfigured. we want the max OTHER THAN background
        
        temp = image_tensor.clone()

        temp[:,mask!=code] = 0

        if False:
            plt.imshow(temp.permute(1,2,0))
            plt.title(f'Masked image {key_map[code]}')
            plt.show()
        
        for obind, obid in enumerate(obj_ids):
            if obid != code:
                pass
            else:
                xmin = int(boxes[obind, 0])
                ymin = int(boxes[obind, 1])
                xmax =int(boxes[obind, 2])
                ymax = int(boxes[obind, 3])

        temp = temp[:, ymin:ymax, xmin:xmax]
        
        if False:
            plt.imshow(temp.permute(1,2,0))
            plt.title('Cropped image')
            plt.show()

        return to_image(temp), code, key_map[code]

    elif mode == 'all':
        transforms.Resize((mask.shape[0],mask.shape[1]))(image_tensor)

        subims = []
        boxims = []
        subids = []
        for obind, obid in enumerate(obj_ids):
            if obid != 0:
                temp = image_tensor.clone()

                xmin = int(boxes[obind, 0])
                ymin = int(boxes[obind, 1])
                xmax =int(boxes[obind, 2])
                ymax = int(boxes[obind, 3])

                boxim = temp.clone()[:, ymin:ymax, xmin:xmax]

                # 

                temp[:,mask!=obid] = 0 # Set everything but current clothing item to 0

                
                subim = temp[:, ymin:ymax, xmin:xmax]

                if False:
                    plt.imshow(subim.squeeze().permute(1,2,0))
                    plt.title(f'Only {key_map[obid.item()]}')
                    plt.show()
            
                subim = to_image(subim)
                subims.append(subim) # I think we need them in image format for Clip
                subids.append(obid.item())
                boxims.append(to_image(boxim))

        return subims, subids, boxims



def add_item_to_database(img, mode, name, price = 100, debug = False):
    # Segment an uploaded image and add it to the database of images
    # There are a few ways to do this, depending on if we want the entire image to be added, or take the Mode, etc
    # For now we use the old version which takes the mode

    if mode == 'all':
        # Take all constituent parts of uploaded image, send them all to database
        subims, subids, boxims = image_to_subims(img, 'all', debug)

        # Embed the CROPPED SUB image
        img_embs = embed_image(subims)

        for i, (subim, subid, boxim) in enumerate(zip(subims, subids, boxims)):
            # Look in "database" directory and find the number of items:
            trymkdir(f'database2/{key_map[subid]}/') # make sure item type directory exists

            item_path = f'database2/{key_map[subid]}/{name}/'
            trymkdir(item_path) # Make sure database directory exists

            img_emb = img_embs[i]

            # Make the database directory for this item under the code subdir
            img.save(f'{item_path}{name}_full_jpg.jpg')# Original image - pil
            subim.save(f'{item_path}{name}_subim_jpg.jpg')# Scanned item cutout - pil
            boxim.save(f'{item_path}{name}_box_im.jpg')
            #torch.save(pil, f'{newdir}img.pt') 
            #torch.save(temp, f'{newdir}temp.pt') 
            torch.save(img_emb, f'{item_path}emb.pt') # Embedding of image - vector
            torch.save(name, f'{item_path}item_name.pt') # Item string - str
            torch.save(price, f'{item_path}price.pt') # Provided Price - float

        return subims, subids


    elif mode == 'biggest':
        temp, code, item_str = image_to_subims(img, 'biggest', debug)
    
        # Look in "database" directory and find the number of items:
        trymkdir(f'database/{code}/') # Make sure database directory exists
        dblist = os.listdir(f'database/{code}/') # Go to subdirectory of item type
        item_id = len(dblist)
        
        # Embed the CROPPED SUB image
        img_emb = embed_image(temp)
        print('This item emebdded to', img_emb.shape)

        # Make the database directory for this item under the code subdir
        item_path = f'database/{code}/{item_id}/'
        trymkdir(item_path)

        # Make the database directory for this item under the code subdir
        img.save(f'{item_path}{item_name}_full_jpg.jpg')# Original image - pil
        temp.save(f'{item_path}{item_name}_subim_jpg.jpg')# Scanned item cutout - pil
        #torch.save(pil, f'{newdir}img.pt') 
        #torch.save(temp, f'{newdir}temp.pt') 
        torch.save(img_emb, f'{item_path}emb.pt') # Embedding of image - vector
        torch.save(name, f'{item_path}item_name.pt') # Item string - str
        torch.save(price, f'{item_path}price.pt') # Provided Price - float

        return temp, code, item_str

    
def nearest_items(subims, subids):
    ######
    # Take in a list of subimages, and thier id's
    # Embed each image
    # Take the distance between the embeding and the relevant subset of images
    ######

    imgs = []
    names = []
    prices = []
    temps = []
    boxims = []

    subim_embs = embed_image(subims)

    for i, (subid, subim) in enumerate(zip(subids, subims)):
        

        trymkdir(f'database2/{key_map[subid]}/')
        dir_list = os.listdir(f'database2/{key_map[subid]}/')
        if len(dir_list) > 0:
            subim_emb = subim_embs[i]

            distances = []
            min_dist = 1e9
            min_ind = None
            for i, val_dir in enumerate(dir_list):
                # While iterating, take distances
                print(i, val_dir)
                emb = torch.load(f'database2/{key_map[subid]}/{val_dir}/emb.pt')

                #distance = cosine_similarity(subim_emb, emb)

                print(subim_emb.shape, emb.shape)
                distance = euclidean_distances(subim_emb.reshape(1,-1), emb.reshape(1,-1))

                print(val_dir, distance)
                #print('Distance between items - i -valdir - dist', i, val_dir, distance)
                if distance < min_dist:
                    min_dist = distance
                    min_ind = i
                    print('new min is', min_dist, min_ind)

            # Now we have the nearest validation element for this subim
            imgs.append(Image.open(f'database2/{key_map[subid]}/{dir_list[min_ind]}/{dir_list[min_ind]}_full_jpg.jpg'))
            boxims.append(Image.open(f'database2/{key_map[subid]}/{dir_list[min_ind]}/{dir_list[min_ind]}_box_im.jpg'))
            temps.append(Image.open(f'database2/{key_map[subid]}/{dir_list[min_ind]}/{dir_list[min_ind]}_subim_jpg.jpg'))
            names.append(torch.load(f'database2/{key_map[subid]}/{dir_list[min_ind]}/item_name.pt'))
            prices.append(torch.load(f'database2/{key_map[subid]}/{dir_list[min_ind]}/price.pt'))

    return imgs, boxims, temps, names, prices

def draw_meme(tar_img, imgs, names, prices, id_list):
    ######
    # Draw the steal their look meme
    # tar_img in image form, imgs is list of images
    # Currently only works with 2 element inputs
    ######

    # Make images on white background:
    # new_imgs = []
    # for img in imgs:
    #     img = to_tensor(img)
    #     print(torch.min(img), torch.max(img))
    #     img[img<0.0001] = 0.9999
    #     #plt.imshow(img.permute(1,2,0))
    #     #plt.show()
    #     img = to_image(img)
    #     new_imgs.append(img)

    # imgs = new_imgs

    
    #### Make the canvas:
    # Determine the height of the combined images
    bias = 80
    height = 1200+bias
    
    image1 = tar_img
    width1 = int(image1.width * height / image1.height)
    image1 = image1.resize((width1, height), Image.ANTIALIAS)
    
    total_width = image1.width
    
    total_width += len(imgs)//2 * image1.width ## the total width should be the tar_img, and half the remaining scaled up
    
    # Create a new blank image to hold the combined images
    result = Image.new("RGB", (total_width, height), color = 'white')
    
    # plot the first image
    result.paste(image1, (0, bias))
    
    # Add text onto image
    fill = (0, 0, 0)
    draw = ImageDraw.Draw(result)

    big_font = ImageFont.truetype("arial.ttf", 80)
    small_font = ImageFont.truetype("arial.ttf", 35)

    # Draw captions
    midpt = height//2
    uppt = int(midpt - height/60)
    dnt = int(midpt + height/60)
    
    # Draw title
    title = 'Steal Their Look'
    tw, th = draw.textsize(title, big_font)
    draw.text(( (total_width-tw)//2, 0), 'Steal Their Look', font = big_font, fill = fill)
    
    
    #### Plot remaining images:
    xpos = width1
    ypos = bias

    secondary_bias = 0
    for i, image in enumerate(imgs):
        print('writing image', i, xpos, ypos)

        scale = 0.8
        #width = min(width, width1)
        width = int(scale*image.width * height / (image.height * 2))
        
        #image = image.resize((width, int((scale*height) // 2)), Image.ANTIALIAS
        image = ImageOps.contain(image, (image1.width//2, height//2))
        
        if i%2 == 0: # Even, above
            result.paste(image, (xpos, ypos+secondary_bias))
            draw.text((xpos, uppt+secondary_bias), f'{names[i]} - {key_map[id_list[i]]} - ${prices[i]}', font=small_font, fill=fill)
            
        else: # Odd, below
            result.paste(image, (xpos, ypos + height//2+secondary_bias))
            draw.text((xpos, dnt+secondary_bias), f'{names[i]} - {key_map[id_list[i]]} - ${prices[i]}', font=small_font, fill=fill)
            
            # Move the cursor to the right for new cols:
            xpos += image.width+bias

        secondary_bias += 15
            
        # I'd like to improve this code so that each of these images situates itself in the middle of its box
            
        
        
    # Save the result
    timestr = time.strftime("%Y%m%d-%H%M%S")
    savedir = f"memes/{timestr}.jpg"
    result.save(savedir)
    return result, savedir

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