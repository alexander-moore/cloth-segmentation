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

import bot_utils as butils
import matplotlib.pyplot as plt

openai.api_key = "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"

intents = discord.Intents.all()

bot = commands.Bot(intents = intents, command_prefix='!')

# transform = transforms.Compose([
#                 transforms.Resize((224, 224)),  # resize the image to 224x224 pixels
#                 transforms.ToTensor()  # convert the PIL Image to a PyTorch tensor
#             ])

key_strs = ['None', 'Shirt', 'Pant', 'Dress', 'unknown?']

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')


@bot.command(pass_context = True)
async def db(ctx, *, user_input):
    
    # Parse user input:
    print('user input:', user_input)
    if '-debug' in user_input:
        user_input = user_input[6:]
        debug = True
        print('debugging')
        await ctx.channel.send(f"Entering debug mode for !db call")
    else:
        debug = False

    # Get database arguments:
    name, price = user_input.split(', ')

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
            code, img, temp = butils.add_item_to_database(image, name, price)
            print('caught added item', code, key_strs[code])

            if debug:
                # Visualize image by converting back to PIL, save to png, send to disc
                temp = (temp+1)/2
                item_image = transforms.ToPILImage()(temp.squeeze())
                file_obj = io.BytesIO()
                item_image.save(file_obj, 'PNG')
                file_obj.seek(0)

                file = discord.File(file_obj, filename = 'my_image.png')
                await ctx.channel.send(f'Successfully added {name}, {code}, {key_strs[code]}',file = file)
            else:
                await ctx.channel.send(f'Successfully added {name}, {code}, {key_strs[code]}')

    else:
        await ctx.channel.send(f"Did not find an image on database request")

@bot.command(pass_context = True)
async def steal(ctx, *, user_input):

    if '-debug' in user_input:
        user_input = user_input[9:]
        debug = True
        print('debugging')
        await ctx.channel.send(f"Entering debug mode for !steal call")

    if ctx.message.attachments:
        attachment = ctx.message.attachments[0]
        if attachment.url.endswith('.png') or attachment.url.endswith('.jpg') or attachment.url.endswith('.jpeg'):

            image_data = await attachment.read()  # read the attachment data
            image = Image.open(io.BytesIO(image_data)).convert("RGB")  # create a PIL (RGB?) Image object from the image data
            
            #tensor = transform(image)
            #print('Tensor of shape', tensor.shape)

            # Parse input image into list of tensor sub-images
            subimage_list, id_list = butils.parse_image(image)
            print('Validating Meme:')
            print('subimage_list', len(subimage_list))
            print('id_list', id_list)

            if len(subimage_list) == 0:
                await ctx.channel.send("Failed to find clothing")
                return

            if debug:
                await ctx.channel.send("Found the following clothing items:")
                for im, id_ in zip(subimage_list, id_list):
                    im = (im+1)/2
                    item_image = transforms.ToPILImage()(im.squeeze())
                    file_obj = io.BytesIO()
                    item_image.save(file_obj, 'PNG')
                    file_obj.seek(0)

                    file = discord.File(file_obj, filename = 'my_image.png')
                    await ctx.channel.send(f'Subimage item {key_strs[id_]}:',file = file)

            # Get nearest validation set items for subimages
            imgs, temps, names, prices = butils.nearest_items(subimage_list, id_list)

            if debug:
                await ctx.channel.send("Found the nearest paired items:")
                for im, id_ in zip(temps, id_list):
                    im = (im+1)/2
                    item_image = transforms.ToPILImage()(im.squeeze())
                    file_obj = io.BytesIO()
                    item_image.save(file_obj, 'PNG')
                    file_obj.seek(0)

                    file = discord.File(file_obj, filename = 'my_image.png')
                    await ctx.channel.send(f'Subimage item {key_strs[id_]}:',file = file)

            # Build a meme with these images, prices, names, and id's
            

            print('imgs', len(imgs), type(imgs[0]))
            print('prices', prices)
            print('names', names)

            meme_image, meme_dir = butils.draw_meme(image, imgs, names, prices)

            #await ctx.channel.send(f"{user_input}: Made list of {len(subimage_list)} tensors of shape {subimage_list[0].shape}")
            #for img, price, name, id_ in zip(imgs, names, prices, id_list):
                #outstr = f'{name} {key_strs[id_]}-${price}'
            file_obj = io.BytesIO()
            meme_image.save(file_obj, 'PNG')
            file_obj.seek(0)

            file = discord.File(file_obj, filename = 'my_image.png')
            await ctx.channel.send(file = file)

    else:
        await ctx.channel.send(f"Did not find an image")

bot.run('MTA3MjMzNDg1MTIwNzg3NjY2OA.Gdavkm.BhByEDQiTp6S-GERlM4Ys28XanGQtV7ROe7XeE')