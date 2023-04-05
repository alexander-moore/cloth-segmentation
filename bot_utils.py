# Utilities for StealTheirLook

import os
import time
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from cloth_utils import Normalize_image
from utils.saving_utils import load_checkpoint_mgpu

from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes

from networks import U2NET, emb_U2NET
import matplotlib.patches as patches
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

device = "cuda"

image_dir = "input_images"
result_dir = "output_images"
checkpoint_path = '../../Downloads/cloth_segm_u2net_latest.pth'
do_palette = True
debug_plotting = False

def get_palette(num_cls):
    """Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= ((lab >> 0) & 1) << (7 - i)
            palette[j * 3 + 1] |= ((lab >> 1) & 1) << (7 - i)
            palette[j * 3 + 2] |= ((lab >> 2) & 1) << (7 - i)
            i += 1
            lab >>= 3
    return palette

# Segmentation transform
resize = transforms.Resize((1024,1024), antialias = True)
transforms_list = []
transforms_list += [transforms.ToTensor()]
transforms_list += [Normalize_image(0.5, 0.5)]
transforms_list += [resize]
transform_rgb = transforms.Compose(transforms_list)

# Embedding IMAGE transform
embed_resize = transforms.Resize((224,224), antialias = True)
embed_transforms_list = []
embed_transforms_list += [transforms.ToTensor()]
embed_transforms_list += [Normalize_image(0.5, 0.5)]
embed_transforms_list += [embed_resize]
embed_transform_rgb = transforms.Compose(embed_transforms_list)

# Embedding TENSOR transform
embed_resize = transforms.Resize((224,224), antialias = True)
embed_transforms_list = []
embed_transforms_list += [Normalize_image(0.5, 0.5)]
embed_transforms_list += [embed_resize]
embed_tensor_transform_rgb = transforms.Compose(embed_transforms_list)

#### Global elements
# Segmentation model
net = U2NET(in_ch=3, out_ch=4)
net = load_checkpoint_mgpu(net, checkpoint_path)
net = net.to(device).eval()

# Embedding model
resnet152 = models.resnet152(pretrained=True)
modules=list(resnet152.children())[:-1]
resnet152=nn.Sequential(*modules)
for p in resnet152.parameters():
    p.requires_grad = False

embedding_model = resnet152
embedding_model = embedding_model.to(device).eval()

# Segmentation keys
key_strs = ['shirt', 'pants', 'dress', 'unknown?']

def parse_image(img):
    ######
    # Take an input image, and parse it into shirt and pants.
    # The input image should be an Image.open(path).convert("RGB") type?
    # Return the list of images which constitute the whole
    ######

    palette = get_palette(4)

    #img = Image.open(os.path.join(image_dir, image_name)).convert("RGB")
    
    image_tensor, mask_tensor = forward_image(img)
    print('output array shape in parse_image', image_tensor.shape, mask_tensor.shape)

    output_img = Image.fromarray(mask_tensor.astype("uint8"), mode="L")
    #plt.imshow(output_img)
    #plt.show()
    if do_palette:
        output_img.putpalette(palette)
    #output_img.save(os.path.join(result_dir, image_name[:-3] + "png"))
    
    # For each value in image:
    obj_ids = np.unique(mask_tensor)[1:]
    masks = torch.BoolTensor(mask_tensor == obj_ids[:,None, None])
    boxes = masks_to_boxes(masks)

    subims = []
    subids = []
    
    for obind, obid in enumerate(obj_ids):
        temp = image_tensor.clone()
        temp[:,:,mask_tensor!=obid] = 0

        xmin = int(boxes[obind, 0])
        ymin = int(boxes[obind, 1])
        xmax =int(boxes[obind, 2])
        ymax = int(boxes[obind, 3])

        subim = temp[:,:, ymin:ymax, xmin:xmax]
        subim = resize(subim)
        
        #plt.imshow((subim.squeeze().permute(1,2,0) + 1)/2)
        #plt.show()
        
        subims.append(subim)
        subids.append(obid)
        
        if debug_plotting == True:
            fig, ax = plt.subplots()
            ax.imshow(.5*(image_tensor.squeeze().permute(1,2,0)+1))

            # Create a Rectangle patch
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=5, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)

            plt.show()
            
            #plt.imshow(.5*(subim.squeeze().permute(1,2,0)+1))
            #plt.show()

    return subims, subids

def forward_image(img):
    # Forward call of segmentation model on image (RGB)
    image_tensor = transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    
    with torch.no_grad(): output_tensor = net(image_tensor.to(device))
    output_tensor = F.log_softmax(output_tensor[0], dim=1)
    output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_tensor = torch.squeeze(output_tensor, dim=0)
    output_arr = output_tensor.cpu().numpy()

    return image_tensor, output_arr

def embed_image(img):
    # Activations from some layer of segmentation model on image
    image_tensor = embed_transform_rgb(img)
    image_tensor = torch.unsqueeze(image_tensor, 0)
    
    with torch.no_grad(): output_tensor = embedding_model(image_tensor.to(device))
    output_arr = output_tensor.cpu().numpy()

    return output_arr

def embed_tensor(image_tensor):
    # Activations from some layer of segmentation model on image
    image_tensor = embed_tensor_transform_rgb(image_tensor)
    
    print('input of embedding model', image_tensor.shape, type(image_tensor))
    with torch.no_grad(): output_tensor = embedding_model(image_tensor.to(device))
    print('output of embedding model', output_tensor.shape, type(image_tensor))

    output_arr = output_tensor.cpu().numpy()

    return output_arr


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

    for subid, subim in zip(subids, subims):

        print('subim in nearest', subim.shape)
        subim_emb = emb_centroid(subim.squeeze(), 16).cpu()

        dir_list = os.listdir(f'database/{subid}/')
        distances = []
        min_dist = 1e6
        min_ind = None
        for i, val_dir in enumerate(dir_list):
            # While iterating, take distances
            emb = torch.load(f'database/{subid}/{i}/centroid.pt')

            subim_emb = subim_emb.reshape(1, -1)
            emb = emb.reshape(1, -1)

            #distance = cosine_similarity(subim_emb, emb)
            distance = euclidean_distances(subim_emb, emb)

            print(val_dir, distance)
            #print('Distance between items - i -valdir - dist', i, val_dir, distance)
            if distance < min_dist:
                min_dist = distance
                min_ind = i

        # Now we have the nearest validation element for this subim
        imgs.append(torch.load(f'database/{subid}/{dir_list[min_ind]}/img.pt'))
        temps.append(torch.load(f'database/{subid}/{dir_list[min_ind]}/temp.pt'))
        names.append(torch.load(f'database/{subid}/{dir_list[min_ind]}/name.pt'))
        prices.append(torch.load(f'database/{subid}/{dir_list[min_ind]}/price.pt'))

    return imgs, temps, names, prices


def trymkdir(path):
    if os.path.exists(path):
        pass
    else:
        os.mkdir(path)

def str_to_code(string):
    if string == 'shirt':
        return 1

    elif string == 'pants':
        return 2

    elif string == 'dress':
        return 3

    else:
        return False

def add_item_to_database(img, name, price):
    # Parse the input
    image_tensor, mask_tensor = forward_image(img)
    print('output array shape in parse_image: image_tensor.shape, mask_tensor.shape', image_tensor.shape, mask_tensor.shape)
    obj_ids = np.unique(mask_tensor)[1:]

    print(type(mask_tensor))

    print('obj_ids, mask_tensor.shape', obj_ids, mask_tensor.shape)
    print(type(obj_ids), type(obj_ids[0]))
    masks = torch.BoolTensor(mask_tensor == obj_ids[:,None, None])
    boxes = masks_to_boxes(masks)
    
    #plt.imshow(mask_tensor)
    #plt.show()

    value_counts = torch.bincount(torch.FloatTensor(mask_tensor).int().flatten())[1:].tolist()
    code, _ = max(list(enumerate(value_counts)), key=lambda x: x[1])
    code = code+1

    
    temp = image_tensor.clone()
    temp[:,:,mask_tensor!=code] = 0
    
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

    torch.save(img, f'{item_path}/img.pt')
    torch.save(temp, f'{item_path}/temp.pt')
    torch.save(img_emb, f'{item_path}/img_emb.pt')
    torch.save(name, f'{item_path}/name.pt')
    torch.save(price, f'{item_path}/price.pt')
    torch.save(centroid, f'{item_path}/centroid.pt')

    return code, img, temp


def draw_meme(tar_img, imgs, names, prices):
    ######
    # Draw the steal their look meme
    # tar_img in image form, imgs is list of images
    # Currently only works with 2 element inputs
    ######

    multi = True if len(imgs) > 1 else False
    # Unpack images
    image1 = tar_img
    image2 = imgs[0]

    if multi:
        image3 = imgs[1]
    
    # Determine the height of the combined images
    bias = 50
    height = 1200+bias

    # Resize image 1 to have the same height as the combined images
    width1 = int(image1.width * height / image1.height)
    image1 = image1.resize((width1, height), Image.ANTIALIAS)

    # Resize image 2 and 3 to have half the height of the combined images
    scale = 0.8
    width2 = int(scale*image2.width * height / (image2.height * 2))
    image2 = image2.resize((width2, int((scale*height) // 2)), Image.ANTIALIAS)

    if multi:
        width3 = int(scale*image3.width * height / (image3.height * 2))
        image3 = image3.resize((width3, int((scale*height) // 2)), Image.ANTIALIAS)

    # Determine the width of the blank image
    if multi:
        width = width1 + max(width2, width3)
    else:
        width = width1 + width2

    # Create a new blank image to hold the combined images
    result = Image.new("RGB", (width, height), color = 'white')

    # Paste the images onto the new image
    #half_mean_width = (width2+width3)//4
    result.paste(image1, (0, bias))

    #pos = int(height/2 - height/20)
    result.paste(image2, (width1, bias))

    if multi:
        pos = int(height/2 + height/20)
        result.paste(image3, (width1, pos+bias))

    # this is image size: int((scale*height) // 2))
    # put in center of area of size: height//2
    # which is int(((scale*hegiht)/2 + height/2)/2)

    # Add text onto image
    fill = (0, 0, 0)
    draw = ImageDraw.Draw(result)

    big_font = ImageFont.truetype("arial.ttf", 80)
    small_font = ImageFont.truetype("arial.ttf", 35)

    # Draw captions
    midpt = height//2
    uppt = int(midpt - height/60)
    dnt = int(midpt + height/60)
    
    title = 'Steal Their Look'
    tw, th = draw.textsize(title, big_font)
    draw.text(( (width-tw)//2, 0), 'Steal Their Look', font = big_font, fill = fill)
    draw.text((width1, uppt), f'{names[0]} - ${prices[0]}', font=small_font, fill=fill)

    if multi:
        draw.text((width1, dnt), f'{names[1]} - ${prices[1]}', font=small_font, fill=fill)

    # Save the result
    timestr = time.strftime("%Y%m%d-%H%M%S")
    savedir = f"memes/{timestr}.jpg"
    result.save(savedir)
    return result, savedir

key_map = {
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