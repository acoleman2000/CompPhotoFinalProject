#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################

'''CLI tools for Comp Photo'''
################################################################################

__version__ = "0.0.0"
__status__ = "Development"


import sys
import argparse
import numpy as np
import skimage
import requests
import urllib
from io import BytesIO
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw
import math
import cv2
color_mapping = {
'protanopia': np.array([
    [0,1.05118294, -0.05116099],
    [0,1,0],
    [0,0,1]
]
),
'deuteranopia':np.array(
    [
    [1,0,0],
    [0.9513092,0,0.04866992],
    [0,0,1]
    ]
),
'tritanopia':np.array(
    [
    [1,0,0],
    [0,1,0],
    [-0.86744736, 1.86727089,0]
    ]
)
}

def usr_args():
    """
    functional arguments for process
    https://stackoverflow.com/questions/27529610/call-function-based-on-argparse
    """

    # initialize parser
    parser = argparse.ArgumentParser()

    # set usages options
    parser = argparse.ArgumentParser(
        prog='vision',
        usage='%(prog)s [options]')

    # version
    parser.add_argument(
        '-v', '--version',
        action='version',
        version='%(prog)s ' + __version__)

    # create subparser objects
    subparsers = parser.add_subparsers()

    # Create parent subparser. Note `add_help=False` & creation via `argparse.`
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-i', '--image',
                               required=True,
                               help="Local path to image or url")

    parent_parser.add_argument('-o', '--output',
                                required=True,
                                help="Name of output image")



    # Create a functions subcommand
    parser_listapps = subparsers.add_parser('functions',
                                            help='List all available functions.')
    parser_listapps.set_defaults(func=list_apps)

    # Create the bco_license
    parser_color_blindness = subparsers.add_parser('color_blindness',
                                           parents=[parent_parser],
                                           help='Convert RGB of an image to what a colorblind person would see.')
    parser_color_blindness.set_defaults(func=color_blindness)
    parser_color_blindness.add_argument('-c', '--color_type',
                              nargs=1, choices=['protanopia','deuteranopia', 'tritanopia', 'all'],
                              help="Type of color-blindness",required=True)

    # Create a validate subcommand
    parser_sightedness = subparsers.add_parser('sightedness',
                                            parents=[parent_parser],
                                            help="See an image as what a near-sighted/far-sighted person would see")
    parser_sightedness.set_defaults(func=sightedness)
    parser_sightedness.add_argument('-s', '--sightedness',
                              nargs=1, choices =["near", "far"],
                              help="Near or far sightedness", required=True)


    parser_tunnel_vision = subparsers.add_parser('tunnel_vision',
                                            parents=[parent_parser],
                                            help="See an image as what someone with tunnel vision would see")
    parser_tunnel_vision.set_defaults(func=tunnel_vision)


    parser_stigmatisim = subparsers.add_parser('stigmatisim',
                                            parents=[parent_parser],
                                            help="See an image as what someone with a stigmatism would see")
    parser_stigmatisim.set_defaults(func=stigmatism)
    parser_stigmatisim.add_argument("-in", "-intensity",
                                    type=int)

    parser_cartaracts = subparsers.add_parser('cartaracts',
                                            parents=[parent_parser],
                                            help="See an image as what somewone with cartacts would see")
    parser_cartaracts.set_defaults(func=cataracts)
    parser_cartaracts.add_argument("-t", "--type",
                                    nargs=1, choices=['yellow', 'ghost','blur','all'])


    # Print usage message if no args are supplied.
    if len(sys.argv) <= 1:
        sys.argv.append('--help')

    # Run the appropriate function
    options = parser.parse_args()
    if options.func is list_apps:
        options.func(parser)
    else:
        options.func(options)



def list_apps(parser: argparse.ArgumentParser):
    """
    List all functions and options available in app
    https://stackoverflow.com/questions/7498595/python-argparse-add-argument-to-multiple-subparsers
    """

    print('Function List')
    subparsers_actions = [
        # pylint: disable=protected-access
        action for action in parser._actions
        # pylint: disable=W0212
        if isinstance(action, argparse._SubParsersAction)]
    # there will probably only be one subparser_action,
    # but better safe than sorry
    for subparsers_action in subparsers_actions:
        # get all subparsers and print help
        for choice, subparser in subparsers_action.choices.items():
            print("Function: '{}'".format(choice))
            print(subparser.format_help())
    # print(parser.format_help())


def color_blindness(options: dict):
    if(options.color_type[0] == "all"):
        image = load_image(options.image)
        old_im = image.copy()
        fig = plt.figure(figsize=(10, 7))
        ax=fig.add_subplot(2, 2, 1)
        skimage.io.imshow(old_im)
        ax.title.set_text('Default')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        i = 2
        for key in color_mapping:
            old_im = image.copy()
            apply_matrix_to_image(old_im, color_mapping[key])


            ax=fig.add_subplot(2, 2, i)
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            i = i + 1

            skimage.io.imshow(old_im)
            ax.title.set_text(key)
        plt.savefig(options.output)
        plt.show()
    
    else:
        image = load_image(options.image)
        fig = plt.figure(figsize=(10, 7))
        ax=fig.subplot(2, 1, 1)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        skimage.io.imshow(old_im)
        ax.title.set_text('Default')

        color_matrix = color_mapping[options.color_type[0]]
        apply_matrix_to_image(image, color_matrix)


        fig.subplot(2, 1, 2)
        ax1=skimage.io.imshow(image)
        ax1.title.set_title(options.color_type[0])
        ax1.axes.xaxis.set_visible(False)
        ax1.axes.yaxis.set_visible(False)
        plt.savefig(options.output)
        plt.show()

def apply_matrix_to_image(image, matrix):
    nrows, ncols, colors = image.shape
    for i in range(0, nrows):
        for j in range(0, ncols):
            vec = np.dot(matrix, image[i, j,:])
            vec = [min(item, 255) for item in vec]
            image[i,j,:] = vec


def tunnel_vision(options: dict):
    image = load_image(options.image)
    g_dict = {
        "Early": 1,
        "Advanced":1.25,
        "Extreme":1.5
    }
    fig = plt.figure(figsize=(10, 7))
    ax=fig.add_subplot(2, 2, 1)
    skimage.io.imshow(image)
    ax.title.set_text('Default')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    nrows, ncols, colors = image.shape
    k = 2
    for key,value in g_dict.items():
        scale = 1/math.dist([nrows, ncols], [nrows/2, ncols/2]) * value
        
        new_image  = image.copy()
        for i in range(0, nrows):
            for j in range(0, ncols):
                dist = min(1, math.dist([i,j], [nrows/2, ncols/2]) * scale)
                imscale = 1 - dist
                new_image[i,j,:]=min(1, dist)*np.array([0,0,0]) + imscale * image[i,j,:]
        ax=fig.add_subplot(2, 2, k)
        k=k+1
        skimage.io.imshow(new_image)
        ax.title.set_text(key)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    
    plt.savefig(options.output)
    plt.show()

def sightedness(options: dict):
    return 0

def stigmatism(options: dict):
    return 0

def load_image(path: str) -> np.array:
    """_summary_

    Args:
        path (str): _description_

    Returns:
        np.array: _description_
    """
    #TODO: resize images 
    try:
        im = skimage.io.imread(fname= path)
    except Exception as e: 
        try: 
            req = urllib.request.Request(path, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req)
            buf = BytesIO(response.read())
            im = np.array(Image.open(buf))
        except Exception as e:
            im = skimage.io.imread(fname="default.jpg")
    return im

def cataracts(options):

    im = load_image(options.image)
    if(options.type == "yellow"):
        im1 = apply_yellow_tint(im)
        skimage.io.imshow(im1)
        plt.show()
    elif(options.type == "ghost"):
        im2 = generate_visual_aura(im, 1.5)
        skimage.io.imshow(im2)
        plt.show()
    elif(options.type == "blur"):
        im3 = make_ghost_image(im, 75, 0.7)
        skimage.io.imshow(im3)
        plt.show()
    elif(options.type[0] == "all"):
        im1 = apply_yellow_tint(im)
        im2 = generate_visual_aura(im, 1.5)
        im3 = make_ghost_image(im, 75, 0.7)
        fig = plt.figure(figsize=(10, 7))

        ax=fig.add_subplot(2, 2, 1)
        skimage.io.imshow(im)
        ax.title.set_text('Default')
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        ax=fig.add_subplot(2, 2, 2)
        skimage.io.imshow(im1)
        ax.title.set_text("Yellow tinted vision")
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        ax=fig.add_subplot(2, 2, 3)
        skimage.io.imshow(im2)
        ax.title.set_text('Blurry vision with visual aura')  
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
 
        ax=fig.add_subplot(2, 2, 4)
        skimage.io.imshow(im3)
        ax.title.set_text("Ghost vision")
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.savefig(options.output)
        plt.show()
  
    
def apply_yellow_tint(image):
    yellow_matrix = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0.6]
    ]
    )
    new_image = image.copy()
    apply_matrix_to_image(new_image, yellow_matrix)
    return new_image


def make_ghost_image(image : np.array, shift : int, intensity : float) -> np.array:
    nrows, ncols, colors = image.shape
    new_image = image.copy()
    for i in range(0, nrows):
        for j in range(0, ncols):
            new_image[i,j,:] = image[i,j,:] * intensity +  image[i,min(j + shift, ncols - 1),:] * (1 - intensity)
    return new_image
 


def generate_visual_aura(image : np.array, intensity : float) -> np.array: 

    nrows, ncols, colors = image.shape

    scale = 1/math.dist([nrows, ncols], [nrows/2, ncols/2]) * intensity
    new_image = image.copy()
    for i in range(0, nrows):
        for j in range(0, ncols):
            dist = min(1, math.dist([i,j], [nrows/2, ncols/2]) * scale)
            imscale = 1 - dist
            new_image[i,j,:]=dist*image[i,j,:]+ imscale * np.array([255, 255, 255])
    
    return new_image

def main():
    usr_args()


if __name__ == "__main__":
    main()