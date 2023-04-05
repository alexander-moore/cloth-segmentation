import os
import io
import discord
import openai
import argparse

from discord.ext import commands
import requests
from PIL import Image

import torch
import torchvision.transforms as transforms

import bot_utils_2 as butils
import matplotlib.pyplot as plt

with open('keys/openai_key.txt', 'r') as file:
    openai_key = file.read().rstrip()
print('running on openai_key : ', openai_key)

intents = discord.Intents.all()

bot = commands.Bot(intents = intents, command_prefix='!')

# transform = transforms.Compose([
#                 transforms.Resize((224, 224)),  # resize the image to 224x224 pixels
#                 transforms.ToTensor()  # convert the PIL Image to a PyTorch tensor
#             ])

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

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')


@bot.command(pass_context = True)
async def add(ctx, *, user_input):
    
    # Parse user input:
    print('user input:', user_input)
    if '-debug' in user_input:
        user_input = user_input[6:]
        debug = True
        print('debugging')
        await ctx.channel.send(f"Entering debug mode for !add call")
    else:
        debug = False

    # Get database arguments:
    mode, name, price = user_input.split(', ')
    if '/' in name:
        await ctx.channel.send(f"No slashes in name please")
        return

    # Validate items:
    try:
        float(price)
    except ValueError:
        print("Not a float")
        await ctx.channel.send(f"Invalid price - use a float")
        return

    # Look for image attachment
    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        if attachment.url.endswith('.png') or attachment.url.endswith('.jpg') or attachment.url.endswith('.jpeg'):

            image_data = await attachment.read()  # read the attachment data
            image = Image.open(io.BytesIO(image_data)).convert("RGB")

            # Add parsed message items to database
            print('Found image,', mode, name, price, debug)
            if mode == ' all':
                print('Entered all mode')
                subims, subids = butils.add_item_to_database(image, 'all', name, price, debug)

                if debug:

                    for im, code in zip(subims, subids):
                        # Visualize image by converting back to PIL, save to png, send to disc
                        
                        file_obj = io.BytesIO()
                        im.save(file_obj, 'PNG')
                        file_obj.seek(0)

                        file = discord.File(file_obj, filename = 'my_image.png')
                        await ctx.channel.send(f'Successfully added {name}, {code}, {key_map[code]}',file = file)
                else:
                    await ctx.channel.send(f'Successfully added {name}, {code}, {key_map[code]}')

            if mode == 'biggest':
                temp, code, item_str = butils.add_item_to_database(image, 'biggest', name, price, debug)


                if debug:
                    # Visualize image by converting back to PIL, save to png, send to disc
                    
                    file_obj = io.BytesIO()
                    item_image.save(file_obj, 'PNG')
                    file_obj.seek(0)

                    file = discord.File(file_obj, filename = 'my_image.png')
                    await ctx.channel.send(f'Successfully added {name}, {code}, {key_map[code]}',file = file)
                else:
                    await ctx.channel.send(f'Successfully added {name}, {code}, {key_map[code]}')

    else:
        await ctx.channel.send(f"Did not find an image on database request")

@bot.command(pass_context = True)
async def steal(ctx, *, user_input):

    if '-debug' in user_input:
        user_input = user_input[9:]
        debug = True
        print('debugging')
        await ctx.channel.send(f"Entering debug mode for !steal call")

    else:
        debug = False

    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        if attachment.url.endswith('.png') or attachment.url.endswith('.jpg') or attachment.url.endswith('.jpeg'):

            image_data = await attachment.read()  # read the attachment data
            image = Image.open(io.BytesIO(image_data)).convert("RGB")  # create a PIL (RGB?) Image object from the image data
            

            # Parse input image into list of tensor sub-images
            subimage_list, id_list, _ = butils.image_to_subims(image, mode='all', debug = debug)
            print('Validating Meme:')
            print('subimage_list', len(subimage_list))
            print('id_list', id_list)

            if len(subimage_list) == 0:
                await ctx.channel.send("Failed to find clothing")
                return

            if debug:
                await ctx.channel.send("Found the following clothing items:")
                for im, id_ in zip(subimage_list, id_list):
                    file_obj = io.BytesIO()
                    im.save(file_obj, 'PNG')
                    file_obj.seek(0)

                    file = discord.File(file_obj, filename = 'my_image.png')
                    await ctx.channel.send(f'Subimage item {key_map[id_]}:',file = file)

            # Get nearest validation set items for subimages
            imgs, boxims, temps, names, prices = butils.nearest_items(subimage_list, id_list)

            if debug:
                await ctx.channel.send("Found the nearest paired items:")
                for im, id_ in zip(temps, id_list):
                    # im = (im+1)/2
                    # item_image = transforms.ToPILImage()(im.squeeze())
                    file_obj = io.BytesIO()
                    im.save(file_obj, 'PNG')
                    file_obj.seek(0)

                    file = discord.File(file_obj, filename = 'my_image.png')
                    await ctx.channel.send(f'Subimage item {key_map[id_]}:',file = file)

            # Build a meme with these images, prices, names, and id's
            

            print('imgs', len(imgs), type(imgs[0]))
            print('prices', prices)
            print('names', names)

            meme_image, meme_dir = butils.draw_meme(image, box_ims, names, prices, id_list)

            file_obj = io.BytesIO()
            meme_image.save(file_obj, 'PNG')
            file_obj.seek(0)

            file = discord.File(file_obj, filename = 'my_image.png')
            await ctx.channel.send(file = file)

    else:
        await ctx.channel.send(f"Did not find an image")

with open('keys/discord_bot_key.txt', 'r') as file:
    bot_key = file.read().rstrip()
print('running on bot key: ', bot_key)
bot.run(bot_key)