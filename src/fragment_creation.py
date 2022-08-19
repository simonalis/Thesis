#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
This is to generate fragments of size 512 bytes from fifty 512_4 dataset
-------------------------------------------------------------------------------
    Variables:
    
        folder_name = the folder where you have all files
        tosave_path = where you save the first 512 bytes
        c = fragment size in bytes. default is given 512,
            change is to 4096 if needed
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np
import math
import os
import mmap
import random

"""
find chunk of data from within a block. 
    start: start location of the block
    length: size of the chunk
"""
absolute_path = os.path.dirname(os.path.realpath(__file__))
default_size = True
if default_size == True:
    sizes = [100]
else:
    sizes = list(np.arange(5, 105, 5))  # vector length

def get_foreign_chunk(filename, start, length):
    fobj = open(filename, 'r+b')
    m = mmap.mmap(fobj.fileno(), 0)
    return m[start:start + length]


"""
function for splitting the file data into blocks of size n=4096 bytes
"""


def split_by_blocks(infile, n, marker, save_path):
    f = open(infile, 'rb')
    l = len(f.read())
    split_count = math.ceil(l / n)  # consider the ceiling of the division as an integer.
    # print("Blocks:", split_count)
    s = np.arange(split_count)

    # a folder with mixed types of files to use as part of the last fragment
    loc = absolute_path + "/512_4/mixed"

    with open(infile, 'rb') as f:
        k = 0
        for chunk in iter(lambda: f.read(n), b''):
            i = s[k]
            j = len(chunk)

            if (i == split_count - 1) and (j < n):
                print(infile, " - Shorter files will not be created!", "length = ", j)
                # # list the files
                # filelist = os.listdir(loc)
                # # generate a random integer between 1 and 100
                # index = random.randint(1, 100)
                # # select that random file from filelist
                # randfile = filelist[index]
                # randfile = os.path.join(loc, randfile)
                # # get the missing chunk from location j till n
                # foreign_chunk = get_foreign_chunk(randfile, j, n - j)
                # # append with current chunk
                # chunk = chunk + foreign_chunk
                # # print("Lenth of the last chunk is:"+str(len(chunk))+"kb")
                # print(str(marker) + "_partial_" + str(i + 1) + "." + str(infile.rpartition('.')[-1]))
                # out_file = str(marker) + "_partial_" + str(i + 1) + "." + str(infile.rpartition('.')[-1])
            else:
             #   print(str(marker) + "_full_" + str(i + 1) + "." + str(infile.rpartition('.')[-1]))
                out_file = str(marker) + "_full_" + str(i + 1) + "." + str(infile.rpartition('.')[-1])

                out_file = os.path.join(save_path, out_file)

                with open(out_file, 'wb') as ofile:
                    ofile.write(chunk)
              #  print(n, " of ", out_file, " first fragment created!")
                return
            k += 1


"""
function for splitting the file based on specific marker other than newline (\n)
this function is not used, its for future improvement.
"""


def split_by_marker(f, marker="-MARKER-", block_size=4096):
    print("Start")
    current = ''
    while True:
        block = f.read(block_size)
        if not block:  # end-of-file
            yield current
            return
        current += block
        while True:
            markerpos = current.find(marker)
            if markerpos < 0:
                break
            yield current[:markerpos]
            current = current[markerpos + len(marker):]

    print(current)


if __name__ == "__main__":

    # run it on local or server?
    local = 1  # 0=online, 1=local

    if local == 1:
        folder_name = absolute_path + '/512_4/000'
        tosave_path = absolute_path + '/512_4/dump'
    else:
        folder_name = '/govdocs_files_unzipped/'  # location of unzipped files from govdocs1
        tosave_path = '/fragment_data/'  # location to save fragments

    # this is to generate all 1000 folder names
    # folders = list()
    # for i in range(1000):
    #     # 000 directory has problem in files, start from 001
    #     if 0 < i <= 10:
    #         # print(str('00')+str(i))
    #         folders.append(str('00') + str(i))
    #     if 10 < i <= 100:
    #         # print(str('0')+str(i))
    #         folders.append(str('0') + str(i))
    #     if 100 < i <= 1000:
    #         # print(i)
    #         folders.append(str(i))

    locs = folder_name
    c = 512
    save_path = tosave_path
    i = 0
    #for i in range(len(locs)):
    os.system("mkdir " + str(save_path) + str(locs[i]))  # create a folder
    path = folder_name + str(locs[i])
    save_path = str(save_path) + str(locs[i]) #+ str('/')  # directory to save individual fragments of a folder
    print(path)
    for f in os.listdir(path):
        file = os.path.join(path, f)
        marker = os.path.splitext(f)[-2]
        extension = os.path.splitext(file)[-1]
        fileType = extension.upper()
        split_by_blocks(file, c, marker, save_path)
    save_path = tosave_path  # reset the save path
    print("Program ended running..")
