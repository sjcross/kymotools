from ctraptools.kymos.detect import gauss_1D, get_raw_profile
from lumicks import pylake
from matplotlib.colors import hsv_to_rgb

import csv
import ctraptools.fileutils as fu
import ctraptools.imageutils as iu
import ctraptools.kymos.io as kio
import imageio as io
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tifffile

def save_kymo(kymo, output_filename, output_range=None):
    # Getting image from kymo
    image = kymo.get_image()
            
    # Applying range normalisation
    image = iu.apply_normalisation(image, output_range)

    # Getting calibration
    time_res = 1/kymo.line_time_seconds
    spat_res = 1/kymo.pixelsize_um[1] if len(kymo.pixelsize_um) > 1 else 1/kymo.pixelsize_um[0]

    # Converting output_range to text
    or_str = iu.get_output_range_string(output_range)
    
    # Saving imag
    tifffile.imwrite(
        output_filename, 
        image.astype(np.single),
        resolution=(time_res,spat_res,"MICROMETER"),
        metadata={"OutputRange":or_str}
    )

def batch_save_kymos(input_path, output_path, output_range=None, verbose=False):
    # Creating output folder (if it doesn't already exist)
    os.makedirs(output_path) if not os.path.exists(output_path) else None      

    # Iterating over all files in the input folder
    for filename in os.listdir(input_path):

        # Processing all .h5 files (doesn't matter if they have 'Kymograph' in the name)
        if filename.endswith(".h5"):
            print("Importing "+filename)
            file = pylake.File(input_path + filename)

            # Getting kymographs and saving one-by-one
            kymos = file.kymos
            for kymo_id in kymos:
                kymo = kymos.get(kymo_id)    
                output_filename = output_path + fu.strip_ext(filename) + "_kymo" + kymo_id + ".tiff"
                save_kymo(kymo, output_filename, output_range=output_range)

def read_image(path,channel,x_range=None):
    image = io.imread(path)
    if len(image.shape) == 3:
        image = image[:,:,channel]
    
    if x_range is not None:
        image = image[x_range[0]:x_range[1],:]

    return image

def write_change_points(tracks, filepath):
    with open(filepath+".csv", 'w', newline='') as file:
        writer = csv.writer(file)

        # Adding header row
        writer.writerow(['TrackID','Number of steps','Step positions (timpoint) ->'])

        for track in tracks.values():
            row = []
            row.append(str(track.ID))
            row.append(str(len(track.steps)))
            for step in track.steps:
                row.append(step)
            writer.writerow(row)

def write_traces(tracks, filepath):
    for track in tracks.values():
        with open(filepath+"_ID"+str(track.ID)+".csv", 'w', newline='') as file:
            writer = csv.writer(file)

            # Adding header row
            writer.writerow(['Timepoint','Intensity','Step'])

            for timepoint in track.intensity.keys():
                row = []
                row.append(timepoint)
                row.append(track.intensity.get(timepoint))
                row.append(track.step_trace.get(timepoint))
                writer.writerow(row)

def save_overlay(tracks, image, filepath):
    fig = plt.figure()
    
    plt.imshow(image)
    plt.set_cmap('gray')

    for track in tracks.values():
        # Setting colour to this track with some transparency
        random.seed(track.ID)
        colour = hsv_to_rgb([random.random(),1,1])    
        colour_line = np.append(colour,[0.05])       

        # Drawing each point
        for peak in track.peaks.values():
            plt.plot(peak.t,peak.b,'.',color=colour_line)

        # Adding a label to the centre of each line       
        cent = list(track.peaks.values())[math.floor(len(track.peaks)/2)]
        plt.text(cent.t,cent.b,str(track.ID),fontsize=24,color=colour,horizontalalignment='center',verticalalignment='center')

    ax = fig.get_axes()[0]
    ax.set_axis_off()
    fig.set_size_inches(image.shape[1]/50,image.shape[0]/50)

    plt.savefig(filepath+".png",bbox_inches=0)

def save_plots(tracks, filepath):
    for track in tracks.values():
        fig = plt.figure(figsize=(8,6))
        plt.plot(list(track.intensity.keys()),list(track.intensity.values()))
        plt.plot(list(track.step_trace.keys()),list(track.step_trace.values()))
        plt.xlabel('Timepoint')
        plt.ylabel('Intensity')

        plt.savefig(filepath+"_ID"+str(track.ID)+".png")

def plot_gauss_for_frame(peaks, frame, image):
    plt.figure(figsize=(14,8))

    x, vals = get_raw_profile(image,frame,3)
    plt.plot(x,vals,color="black",linewidth=4)

    for peak in peaks.values():
        if peak.t == frame:
            g0 = gauss_1D(x,peak.a,peak.b,peak.c)
            plt.plot(x,g0)
    
    plt.xlabel('Position (px)')
    plt.ylabel('Intensity')
    plt.show()