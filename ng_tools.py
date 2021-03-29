#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 20:52:32 2021

@author: gabrielemilioherreraoropeza
"""

import sys, os, subprocess, json
from os import listdir
from os.path import isfile, join
from collections import Counter
from math import pi, sqrt


### --- Define install to install missing modules

def install(package):
    """
    Installs any module that is not currently installed in the system
    """
    print(f"Installing {package}...", end = " ")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print("DONE")


### --- Import modules

try:
    import czifile as zis
except ImportError:
    install('czifile')
    import czifile as zis
    
try:
    import numpy as np
except ImportError:
    install('numpy')
    import numpy as np
    
try:
    import xmltodict
except ImportError:
    install('xmltodict')
    import xmltodict
    
try:
    from cellpose import models, plot
except ImportError:
    install('cellpose')
    from cellpose import models, plot
    
try:
    import matplotlib.pyplot as plt
except ImportError:
    install('matplotlib')
    import matplotlib.pyplot as plt
    
try:
    from skimage.measure import regionprops
    from skimage.filters import (threshold_otsu, threshold_isodata, threshold_li, threshold_mean,
                                 threshold_minimum, threshold_triangle, threshold_yen, threshold_sauvola,
                                 gaussian)
except ImportError:
    install("scikit-image")
    from skimage.measure import regionprops
    from skimage.filters import (threshold_otsu, threshold_isodata, threshold_li, threshold_mean,
                                 threshold_minimum, threshold_triangle, threshold_yen, threshold_sauvola,
                                 gaussian)
    
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    install('plotly')
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
try:
    from tqdm import tqdm
except ImportError:
    install('tqdm')
    from tqdm import tqdm
    
try:
    import cv2
except ImportError:
    install('opencv-python')
    import cv2
    
try:
    import pandas as pd
except ImportError:
    install("pandas")
    import pandas as pd
    
try:
    import compress_json
except ImportError:
    install("compress-json")
    import compress_json
    
try:
    from photutils.detection import find_peaks
    from photutils.segmentation import detect_threshold
    from photutils.background import Background2D
except ImportError:
    install("photutils")
    from photutils.detection import find_peaks
    from photutils.segmentation import detect_threshold
    from photutils.background import Background2D
    
try:
    from spatialentropy import leibovici_entropy
except ImportError:
    install("spatialentropy")
    from spatialentropy import leibovici_entropy
    
try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
except ImportError:
    install("scikit-learn")
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler
    
try:
    import umap.umap_ as umap
except ImportError:
    install("umap-learn")
    import umap.umap_ as umap
    
    
#############################################
#     Functions & Classes | Segmentation    #
#############################################


def get_array_czi(
        filename,
        replacevalue=False,
        remove_HDim=True,
        return_addmd=False
        ):
    """
    Get the pixel data of the CZI file as multidimensional NumPy.Array
    :param filename: filename of the CZI file
    :param replacevalue: replace arrays entries with a specific value with Nan
    :param remove_HDim: remove the H-Dimension (Airy Scan Detectors)
    :param return_addmd: read the additional metadata
    :return: cziarray - dictionary with the dimensions and its positions
    :return: metadata - dictionary with CZI metadata
    :return: additional_metadata_czi - dictionary with additional CZI metadata
    """

    # get metadata    
    metadata = get_metadata_czi(filename)

    # get additional metadata    
    additional_metadata_czi = get_additional_metadata_czi(filename)
        
    # get CZI object and read array
    czi = zis.CziFile(filename)
    cziarray = czi.asarray()

    # check for H dimension and remove
    if remove_HDim and metadata['Axes'][0] == 'H':
        metadata['Axes'] = metadata['Axes'][1:]
        cziarray = np.squeeze(cziarray, axis=0)

    # get additional information about dimension order etc.
    dim_dict, dim_list, numvalid_dims = get_dimorder(metadata['Axes'])
    metadata['DimOrder CZI'] = dim_dict

    if cziarray.shape[-1] == 3:
        pass
    else:
        cziarray = np.squeeze(cziarray, axis=len(metadata['Axes']) - 1)

    if replacevalue:
        cziarray = replace_value(cziarray, value=0)

    # close czi file
    czi.close()

    return cziarray, metadata, additional_metadata_czi


def get_metadata_czi(filename, dim2none=False):
    """
    Returns a dictionary with CZI metadata.
    Information CZI Dimension Characters:
    '0': 'Sample',  # e.g. RGBA
    'X': 'Width',
    'Y': 'Height',
    'C': 'Channel',
    'Z': 'Slice',  # depth
    'T': 'Time',
    'R': 'Rotation',
    'S': 'Scene',  # contiguous regions of interest in a mosaic image
    'I': 'Illumination',  # direction
    'B': 'Block',  # acquisition
    'M': 'Mosaic',  # index of tile for compositing a scene
    'H': 'Phase',  # e.g. Airy detector fibers
    'V': 'View',  # e.g. for SPIM
    :param filename: filename of the CZI image
    :param dim2none: option to set non-existing dimension to None
    :return: metadata - dictionary with the relevant CZI metainformation
    """

    # get CZI object and read array
    czi = zis.CziFile(filename)
    #mdczi = czi.metadata()

    # parse the XML into a dictionary
    metadatadict_czi = xmltodict.parse(czi.metadata())
    metadata = create_metadata_dict()

    # get directory and filename etc.
    try:
        metadata['Directory'] = os.path.dirname(filename)
    except:
        metadata['Directory'] = 'Unknown'
    try:
        metadata['Filename'] = os.path.basename(filename)
    except:
        metadata['Filename'] = 'Unknown'
    metadata['Extension'] = 'czi'
    metadata['ImageType'] = 'czi'

    # add axes and shape information
    metadata['Axes'] = czi.axes
    metadata['Shape'] = czi.shape

    # determine pixel type for CZI array
    metadata['NumPy.dtype'] = str(czi.dtype)

    # check if the CZI image is an RGB image depending on the last dimension entry of axes
    if czi.axes[-1] == 3:
        metadata['isRGB'] = True

    metadata['Information'] = metadatadict_czi['ImageDocument']['Metadata']['Information']
    try:
        metadata['PixelType'] = metadata['Information']['Image']['PixelType']
    except KeyError as e:
        print('Key not found:', e)
        metadata['PixelType'] = None

    metadata['SizeX'] = np.int(metadata['Information']['Image']['SizeX'])
    metadata['SizeY'] = np.int(metadata['Information']['Image']['SizeY'])

    try:
        metadata['SizeZ'] = np.int(metadata['Information']['Image']['SizeZ'])
    except:
        if dim2none:
            metadata['SizeZ'] = None
        if not dim2none:
            metadata['SizeZ'] = 1

    try:
        metadata['SizeC'] = np.int(metadata['Information']['Image']['SizeC'])
    except:
        if dim2none:
            metadata['SizeC'] = None
        if not dim2none:
            metadata['SizeC'] = 1

    channels = []
    for ch in range(metadata['SizeC']):
        try:
            channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                            ['Channels']['Channel'][ch]['ShortName'])
        except:
            try:
                channels.append(metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
                                                ['Channels']['Channel']['ShortName'])
            except:
                channels.append(str(ch))

    metadata['Channels'] = channels

    try:
        metadata['SizeT'] = np.int(metadata['Information']['Image']['SizeT'])
    except:
        if dim2none:
            metadata['SizeT'] = None
        if not dim2none:
            metadata['SizeT'] = 1

    try:
        metadata['SizeM'] = np.int(metadata['Information']['Image']['SizeM'])
    except:
        if dim2none:
            metadata['SizeM'] = None
        if not dim2none:
            metadata['SizeM'] = 1

    try:
        metadata['SizeB'] = np.int(metadata['Information']['Image']['SizeB'])
    except:

        if dim2none:
            metadata['SizeB'] = None
        if not dim2none:
            metadata['SizeB'] = 1

    try:
        metadata['SizeS'] = np.int(metadata['Information']['Image']['SizeS'])
    except:
        if dim2none:
            metadata['SizeS'] = None
        if not dim2none:
            metadata['SizeS'] = 1

    try:
        metadata['Scaling'] = metadatadict_czi['ImageDocument']['Metadata']['Scaling']
        metadata['XScale'] = float(metadata['Scaling']['Items']['Distance'][0]['Value']) * 1000000
        metadata['YScale'] = float(metadata['Scaling']['Items']['Distance'][1]['Value']) * 1000000
        metadata['XScale'] = np.round(metadata['XScale'], 3)
        metadata['YScale'] = np.round(metadata['YScale'], 3)
        try:
            metadata['XScaleUnit'] = metadata['Scaling']['Items']['Distance'][0]['DefaultUnitFormat']
            metadata['YScaleUnit'] = metadata['Scaling']['Items']['Distance'][1]['DefaultUnitFormat']
        except:
            metadata['XScaleUnit'] = None
            metadata['YScaleUnit'] = None
        try:
            metadata['ZScale'] = float(metadata['Scaling']['Items']['Distance'][2]['Value']) * 1000000
            metadata['ZScale'] = np.round(metadata['ZScale'], 3)
            try:
                metadata['ZScaleUnit'] = metadata['Scaling']['Items']['Distance'][2]['DefaultUnitFormat']
            except:
                metadata['ZScaleUnit'] = metadata['XScaleUnit']
        except:
            if dim2none:
                metadata['ZScale'] = metadata['XScaleUnit']
            if not dim2none:
                # set to isotropic scaling if it was single plane only
                metadata['ZScale'] = metadata['XScale']
    except:
        metadata['Scaling'] = None

    # try to get software version
    try:
        metadata['SW-Name'] = metadata['Information']['Application']['Name']
        metadata['SW-Version'] = metadata['Information']['Application']['Version']
    except KeyError as e:
        print('Key not found:', e)
        metadata['SW-Name'] = None
        metadata['SW-Version'] = None

    try:
        metadata['AcqDate'] = metadata['Information']['Image']['AcquisitionDateAndTime']
    except KeyError as e:
        print('Key not found:', e)
        metadata['AcqDate'] = None

    try:
        metadata['Instrument'] = metadata['Information']['Instrument']
    except KeyError as e:
        print('Key not found:', e)
        metadata['Instrument'] = None

    if metadata['Instrument'] is not None:

        # get objective data
        try:
            metadata['ObjName'] = metadata['Instrument']['Objectives']['Objective']['@Name']
        except:
            metadata['ObjName'] = None

        try:
            metadata['ObjImmersion'] = metadata['Instrument']['Objectives']['Objective']['Immersion']
        except:
            metadata['ObjImmersion'] = None

        try:
            metadata['ObjNA'] = np.float(metadata['Instrument']['Objectives']['Objective']['LensNA'])
        except:
            metadata['ObjNA'] = None

        try:
            metadata['ObjID'] = metadata['Instrument']['Objectives']['Objective']['@Id']
        except:
            metadata['ObjID'] = None

        try:
            metadata['TubelensMag'] = np.float(metadata['Instrument']['TubeLenses']['TubeLens']['Magnification'])
        except:
            metadata['TubelensMag'] = None

        try:
            metadata['ObjNominalMag'] = np.float(metadata['Instrument']['Objectives']['Objective']['NominalMagnification'])
        except KeyError as e:
            print('Key not found:', e)
            metadata['ObjNominalMag'] = None

        try:
            metadata['ObjMag'] = metadata['ObjNominalMag'] * metadata['TubelensMag']
        except:
            metadata['ObjMag'] = None

        # get detector information
        try:
            metadata['DetectorID'] = metadata['Instrument']['Detectors']['Detector']['@Id']
        except:
            metadata['DetectorID'] = None

        try:
            metadata['DetectorModel'] = metadata['Instrument']['Detectors']['Detector']['@Name']
        except:
            metadata['DetectorModel'] = None

        try:
            metadata['DetectorName'] = metadata['Instrument']['Detectors']['Detector']['Manufacturer']['Model']
        except:
            metadata['DetectorName'] = None

        # delete some key from dict
        del metadata['Instrument']

    # check for well information

    metadata['Well_ArrayNames'] = []
    metadata['Well_Indices'] = []
    metadata['Well_PositionNames'] = []
    metadata['Well_ColId'] = []
    metadata['Well_RowId'] = []
    metadata['WellCounter'] = None

    try:
        allscenes = metadata['Information']['Image']['Dimensions']['S']['Scenes']['Scene']
        for s in range(metadata['SizeS']):
            well = allscenes[s]
            metadata['Well_ArrayNames'].append(well['ArrayName'])
            metadata['Well_Indices'].append(well['@Index'])
            metadata['Well_PositionNames'].append(well['@Name'])
            metadata['Well_ColId'].append(well['Shape']['ColumnIndex'])
            metadata['Well_RowId'].append(well['Shape']['RowIndex'])

        # count the content of the list, e.g. how many time a certain well was detected
        metadata['WellCounter'] = Counter(metadata['Well_ArrayNames'])
        # count the number of different wells
        metadata['NumWells'] = len(metadata['WellCounter'].keys())

    except KeyError as e:
        print('Key not found:', e)
        print('No Scence or Well Information detected:')


    # for getting binning

    try:
        channels = metadatadict_czi['ImageDocument']['Metadata']['Information']['Image']['Dimensions']['Channels']['Channel']
        for channel in range(len(channels)):
            cuch = channels[channel]
            metadata['Binning'].append(cuch['DetectorSettings']['Binning'])
            
    except KeyError as e:
        print('Key not found:', e)
        print('No Binning Found')

    del metadata['Information']
    del metadata['Scaling']
    
    # close CZI file
    czi.close()

    return metadata


def get_additional_metadata_czi(filename):
    """
    Returns a dictionary with additional CZI metadata.
    :param filename: filename of the CZI image
    :return: additional_czimd - dictionary with the relevant OME-TIFF metainformation
    """

    # get CZI object and read array
    czi = zis.CziFile(filename)

    # parse the XML into a dictionary
    metadatadict_czi = xmltodict.parse(czi.metadata())
    additional_czimd = {}

    try:
        additional_czimd['Experiment'] = metadatadict_czi['ImageDocument']['Metadata']['Experiment']
    except:
        additional_czimd['Experiment'] = None

    try:
        additional_czimd['HardwareSetting'] = metadatadict_czi['ImageDocument']['Metadata']['HardwareSetting']
    except:
        additional_czimd['HardwareSetting'] = None

    try:
        additional_czimd['CustomAttributes'] = metadatadict_czi['ImageDocument']['Metadata']['CustomAttributes']
    except:
        additional_czimd['CustomAttributes'] = None

    try:
        additional_czimd['DisplaySetting'] = metadatadict_czi['ImageDocument']['Metadata']['DisplaySetting']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['DisplaySetting'] = None

    try:
        additional_czimd['Layers'] = metadatadict_czi['ImageDocument']['Metadata']['Layers']
    except KeyError as e:
        print('Key not found:', e)
        additional_czimd['Layers'] = None

    # close CZI file
    czi.close()

    return additional_czimd


def create_metadata_dict():
    """
    A Python dictionary will be created to hold the relevant metadata.
    :return: metadata - dictionary with keys for the relevant metadata
    """

    metadata = {'Directory': None,
                'Filename': None,
                'Extension': None,
                'ImageType': None,
                'Name': None,
                'AcqDate': None,
                'TotalSeries': None,
                'SizeX': None,
                'SizeY': None,
                'SizeZ': None,
                'SizeC': None,
                'SizeT': None,
                'Sizes BF': None,
                'DimOrder BF': None,
                'DimOrder BF Array': None,
                'DimOrder CZI': None,
                'Axes': None,
                'Shape': None,
                'isRGB': None,
                'ObjNA': None,
                'ObjMag': None,
                'ObjID': None,
                'ObjName': None,
                'ObjImmersion': None,
                'XScale': None,
                'YScale': None,
                'ZScale': None,
                'XScaleUnit': None,
                'YScaleUnit': None,
                'ZScaleUnit': None,
                'DetectorModel': [],
                'DetectorName': [],
                'DetectorID': None,
                'InstrumentID': None,
                'Channels': [],
                'ImageIDs': [],
                'NumPy.dtype': None,
                'Binning': []
                }

    return metadata


def replace_value(data, value=0):
    """
    Replace specifc values in array with NaN
    :param data: NumPy.Array
    :param value: value inside array to be replaced with Nan
    :return: data - array with new values
    """

    data = data.astype('float')
    data[data == value] = np.nan

    return data


def get_dimorder(dimstring):
    """
    Get the order of dimensions from dimension string
    :param dimstring: string containing the dimensions
    :return: dims_dict - dictionary with the dimensions and its positions
    :return: dimindex_list - list with indices of dimensions
    :return: numvalid_dims - number of valid dimensions
    """

    dimindex_list = []
    dims = ['B', 'S', 'T', 'C', 'Z', 'Y', 'X', '0']
    dims_dict = {}

    for d in dims:

        dims_dict[d] = dimstring.find(d)
        dimindex_list.append(dimstring.find(d))

    numvalid_dims = sum(i > 0 for i in dimindex_list)

    return dims_dict, dimindex_list, numvalid_dims


def check_channels(lst_of_lsts):
    """
    Checks that the channels for every image are the same

    Parameters
    ----------
    lst_of_lsts : list
        List containing a list of the channels for each image.

    Returns
    -------
    test : bool
        True if all channels for all images are the same.
        False if not.

    """
    test = True
    for n in range(len(lst_of_lsts)-1):
        if Counter(lst_of_lsts[n]) != Counter(lst_of_lsts[n+1]):
            test = False
    return test


def wk_array(array, axes):
    """
    Converts the raw array into a proper format to perform the analysis.

    Parameters
    ----------
    array : array
        Array of the image.
    axes : string
        Order of the axes.

    Returns
    -------
    working_array : array
        Array in the proper format to perform the analysis.

    """
    
    if axes == 'SCYX0':
        working_array = array[0]
        
    elif axes == 'BCYX0':
        working_array = array[0]
        
    elif axes == 'CYX0':
        working_array = array

    elif axes == 'YX' or axes == 'XY':
        working_array = array
        
    #elif axes == 'BVCTZYX0':
    #    working_array = array[0,0,:,0,0,:,:]
        
    #elif axes == 'VBTCZYX0':
    #    working_array = array[0,0,0,:,13,:,:]
        
    return working_array


def _cellpose(image, diameter = None, GPU = None):
    """
    Run cellpose for nuclear segmentation

    Parameters
    ----------
    image : array
        Image to segment as array.
    diameter : integer, optional
        Aproximate average diameter. The default is None.
    GPU : bool, optional
        Use GPU if available. The default is None.

    Returns
    -------
    masks : array
        Masks obtained.
    flows : TYPE
        DESCRIPTION.

    """
            
    model = models.Cellpose(gpu = GPU, model_type = 'nuclei')
    channels = [0, 0]
    masks, flows, _, _ = model.eval(image, diameter = diameter, channels = channels)

    return masks, flows


def nucleus_layers(image, mask, cell_no, xscale):
    """
    Generates masks corresponding to nucleus layers.

    Parameters
    ----------
    image : array
        Image of the nuclei as array.
    mask : array
        Masks of the nuclei as array.
    cell_no : int
        Mask number of nucleus of interest.
    xscale : float
        Number of um equivalent to one pixel.

    Returns
    -------
    core : array
        Mask of the core of the nucleus of interest.
    internal : array
        Mask of the internal ring of the nucleus of interest.
    external : array
        Mask of the external ring of the nucleus of interest.

    """

    image = image.copy()
    mask = mask.copy()
    
    iter_1um = round(1/float(xscale)) # Number of iterations equivalent to 1um
    kernel = np.ones((3, 3), np.uint8)
    
    bin_mask = np.zeros(image.shape)
    bin_mask[mask == cell_no] = 1
    
    eroded_1um = cv2.erode(bin_mask, kernel, iterations = iter_1um)
    dilated_1um = cv2.dilate(bin_mask, kernel, iterations = iter_1um)

    eroded_1um = np.array(eroded_1um)
    dilated_1um = np.array(dilated_1um)

    internal = bin_mask - eroded_1um
    external = dilated_1um - bin_mask
    
    bin_mask = np.uint16(bin_mask)
    
    props = regionprops(bin_mask, intensity_image = image)
    for p in props:
        if p['label'] == 1:     
                area_0 = p['area']
    
    area_n = area_0
    kernel = np.ones((3, 3), np.uint16)
    
    iter_bin_mask = bin_mask.copy()
    
    while (area_0/2) < area_n:
        iter_bin_mask = cv2.erode(iter_bin_mask, kernel, iterations = 1)
        props = regionprops(iter_bin_mask, intensity_image = image)
        for p in props:
            if p['label'] == 1:     
                area_n = p['area']
                    
    core = iter_bin_mask
    
    internal[internal == 1] = cell_no
    core[core == 1] = cell_no
    external[external == 1] = cell_no

    internal = internal.astype(np.uint16)
    core = core.astype(np.uint16)
    external = external.astype(np.uint16)
    
    return core, internal, external


def find_avg_intensity(image, mask, cell_no):
    """
    Finds average pixel intensity from an intensity image by using a mask.

    Parameters
    ----------
    image : array-like
        Array of intensity image .
    mask : array-like
        Array of mask.
    cell_no : int
        Number of cell of interest.

    Returns
    -------
    avg_intensity : int
        Average pixel intensity.

    """
    image = np.array(image.copy())
    mask = np.array(mask.copy())
    avg_intensity = np.average(image[mask == cell_no])
    try:
        avg_intensity = round(avg_intensity)
    except:
        avg_intensity = avg_intensity

    return avg_intensity


def find_sum_intensity(image, mask, cell_no):
    """
    Calculates sum of pixel intensity from an intensity image by using a mask.

    Parameters
    ----------
    image : array-like
        Array of intensity image .
    mask : array-like
        Array of mask.
    cell_no : int
        Number of cell of interest.

    Returns
    -------
    sum_intensity : int
        Sum of pixel intensity of the area covered by the mask.

    """
    image = np.array(image.copy())
    mask = np.array(mask.copy())
    sum_intensity = np.sum(image[mask == cell_no])

    return sum_intensity


def get_threshold(image, thresh_option):
    """
    Generate threshold image

    Parameters
    ----------
    image : array-like image
        Intensity image.
    thresh_option : str
        Thresholding option.

    Returns
    -------
    thresh_img : array-like
        Threshold image.

    """
    ### --- Adaptive Otsu
    if thresh_option.lower() == 'adaptive_otsu':
        fltd = gaussian(image, 3)
        th = threshold_otsu(fltd)
        thresh_img = fltd > th

    ### --- Otsu
    elif thresh_option.lower() == 'otsu':
        th = threshold_otsu(image)
        thresh_img = image > th

    ### --- Isodata
    elif thresh_option.lower() == 'isodata':
        th = threshold_isodata(image)
        thresh_img = image > th

    ### --- Li
    elif thresh_option.lower() == 'li':
        th = threshold_li(image)
        thresh_img = image > th

    ### --- Mean
    elif thresh_option.lower() == 'mean':
        th = threshold_mean(image)
        thresh_img = image > th

    ### --- Minimum
    elif thresh_option.lower() == 'minimum':
        th = threshold_minimum(image)
        thresh_img = image > th

    ### --- Triangle
    elif thresh_option.lower() == 'triangle':
        th = threshold_triangle(image)
        thresh_img = image > th

    ### --- Yen
    elif thresh_option.lower() == 'yen':
        th = threshold_yen(image)
        thresh_img = image > th

    ### --- Sauvola
    elif thresh_option.lower() == 'sauvola':
        th = threshold_sauvola(image)
        thresh_img = image > th

    return thresh_img


def binary_img(thresh_img):
    """
    Generates binary image and closes small holes.

    Parameters
    ----------
    thresh_img : Boolean array
        Output of thresholding.

    Returns
    -------
    bin_img : array-like
        Binary image.

    """
    bin_img = np.uint8(thresh_img)

    # Close small holes inside foreground objects
    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    return bin_img
        

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)
    
        
class NuclearGame_Segmentation(object):
    
    def __init__(self, arg):
        """
        Start Nuclear Game.
        Parameters
        ----------
        arg : string
            Is the path to the folder where all the microscope images that will be analysed
            are found.

        Returns
        -------
        None.

        """
        
        self.path_read = arg
        
        # TODO: add compatibility to more formats.
        formats = [".czi", 
                   #".tiff", 
                   #".tif"
                   ]
        
        # Generate dictionary that will contain all the generated data
        self.data = {}
        self.data["files"] = {}
        
        files = [f for f in listdir(self.path_read) if isfile(join(self.path_read, f))]
        
        isFormat = False
        for file in files:
            if file.lower()[-4:] in formats:
                isFormat = True
        
        if isFormat == False:
            raise ValueError("Ops... No valid format found in the given path!")
        
        # Creat out folder in the same path
        try:
            os.mkdir(self.path_read + 'out_ng/')
            self.path_save = self.path_read + 'out_ng/'
        except OSError:
            n = 1
            while True:
                try:
                    os.mkdir(self.path_read + f'out_ng ({n})/')
                    self.path_save = self.path_read + f'out_ng ({n})/'
                    break
                except:
                    n += 1
                    pass
                
                
    def get_file_name(self, _format = ".czi"):
        """
        Gets the file names in a given path.
        Parameters
        ----------
        _format : string
                Is the format of the files that will be analysed
        
        Returns
        -------
        None.

        """
        
        self.image_format = _format
        
        files = [f for f in listdir(self.path_read) if isfile(join(self.path_read, f)) and f.lower().endswith(self.image_format)]
        
        while True:
            no_files = input(f'\nAnalyse all ({len(files)}) {self.image_format} files or select one (all/one)? ')
            if no_files.lower() == "all" or no_files.lower() == "one":
                break
            else:
                print(f"The input {no_files} is not valid, try again...")
            
        if no_files.lower() == "one":
            print("\n")
            for file in files:
                print(file)
            while True:
                new_files = input("\nEnter name of file to analyse: ")
                if new_files in files:
                    files = [new_files]
                    break
                else:
                    print(f"The given file name '{new_files}' is not valid, try again...")
        
        print("\nFiles to be analysed: \n")
        for file in files:
            _file = file.replace(self.image_format, "")
            self.data["files"][_file] = {}
            self.data["files"][_file]["path"] = self.path_read + file
            print(_file, f"(format: {self.image_format.upper()})")
            
            
    def read_files(self):
        """
        Reads every file, generates arrays, and obtains metadata
        
        Returns
        -------
        None.

        """
        
        if self.image_format == ".czi":
            for file in self.data["files"]:
                self.data["files"][file]["array"], self.data["files"][file]["metadata"], self.data["files"][file]["add_metadata"] = get_array_czi(filename = self.data["files"][file]["path"])
        
        # TODO: support TIFF files
        elif self.image_format == ".tiff" or self.image_format == ".tif":
            pass
        
        
    def identify_channels(self):
        """
        Assign a name to each channel

        Returns
        -------
        None.

        """
        
        if self.image_format == ".czi":
            if len(self.data["files"]) > 1:
                lsts_ch = [self.data["files"][file]['metadata']['Channels'] for file in self.data["files"]] 
                test_ch = check_channels(lsts_ch)
                if test_ch == False:
                    raise ValueError("Channels of files are different!")    
                else:
                    self.data["channels_info"] = {}
                    for n, channel in enumerate(lsts_ch[0]):
                        marker = input(f"Insert name of marker in channel {channel}: ")
                        self.data["channels_info"][marker] = n
            elif len(self.data["files"]) == 1:
                self.data["channels_info"] = {}
                for file in self.data["files"]:
                    for n, channel in enumerate(self.data["files"][file]['metadata']['Channels']):
                        self.data["channels_info"][input(f"Insert name of marker in channel {channel}: ")] = n
            
            while True:
                self.data["dna_marker"] = input(f"\nWhich marker is the DNA marker ({'/'.join(self.data['channels_info'].keys())})? ")
                if self.data["dna_marker"] in list(self.data['channels_info'].keys()):
                    break
                else:
                    print(f"{self.data['dna_marker']} is not in the list of markers! Try again...")
                    
        elif self.image_format == ".tiff" or self.image_format == ".tif":
            pass
        
        
    def nuclear_segmentation(self, diameter = None):
        """
        Perform nuclear segmentation.

        Parameters
        ----------
        diameter : Integer, optional
            Approximate nuclear diameter. The default is None.

        Returns
        -------
        None.

        """
        
        for n, file in enumerate(self.data["files"]):
            print(f"\nPerforming segmentation on file {n+1} of {len(self.data['files'])} \n")
            self.data["files"][file]['working_array'] = wk_array(self.data["files"][file]['array'], 
                                                                 self.data["files"][file]['metadata']['Axes'])
        
            self.data["files"][file]["masks"], self.data["files"][file]["flows"] = _cellpose(self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]], 
                                                                                   diameter = diameter, 
                                                                                   GPU = False)
            
    
    def show_segmentation(self, file):
        """
        Shows nuclear segmentation.

        Returns
        -------
        fig : plot
            Image plot of nuclear segmentation.

        """
                
        image = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]]
        mask = self.data["files"][file]["masks"]
        flows = self.data["files"][file]["flows"]
        channels = [0, 0]
        
        fig = plt.figure(figsize=(15,6))
        plot.show_segmentation(fig, image, mask, flows[0], channels = channels)
        
        plt.tight_layout()
        
        return fig
    
      
    def nuclear_features(self):
        """
        Measure first pool of nuclear features.

        Returns
        -------
        None.

        """
        
        for file in tqdm(self.data["files"]):
            
            self.data["files"][file]["nuclear_features"] = self.generate_dict_nf()
            
            if self.data["files"][file]['metadata']['XScale'] == 0.0:
                while True:
                    try:
                        self.data["files"][file]['metadata']['XScale'] = float(input("Enter pixel for X axis (e.g. 0.454): "))
                        break
                    except:
                        print("Invalid input! Try again...\n")
            if self.data["files"][file]['metadata']['YScale'] == 0.0:
                while True:
                    try:
                        self.data["files"][file]['metadata']['YScale'] = float(input("Enter pixel for Y axis (e.g. 0.454): "))
                        break
                    except:
                        print("Invalid input! Try again...\n")
            
            mask = self.data["files"][file]["masks"]
            image = self.data["files"][file]['working_array'][self.data["channels_info"][self.data["dna_marker"]]]
            
            props = regionprops(mask, intensity_image = image)
            
            for p in props:
                if p['label'] > 0:
                    
                    self.data["files"][file]["nuclear_features"]["cell_no"].append(p['label'])
                    
                    area = p['area'] * (self.data["files"][file]['metadata']['XScale'] * self.data["files"][file]['metadata']['YScale'])
                    self.data["files"][file]["nuclear_features"]["nuclear_area"].append(round(area))
                    
                    self.data["files"][file]["nuclear_features"][f"avg_intensity_{self.data['dna_marker']}"].append(round(p['mean_intensity']))
                    
                    perimeter = p['perimeter'] * self.data["files"][file]['metadata']['XScale']
                    self.data["files"][file]["nuclear_features"]["nuclear_perimeter"].append(round(perimeter))
                    
                    circularity = 4 * pi * (area / perimeter ** 2)
                    self.data["files"][file]["nuclear_features"]["circularity"].append(round(circularity, 3))
                    
                    self.data["files"][file]["nuclear_features"]["eccentricity"].append(round(p['eccentricity'], 3))
                    
                    self.data["files"][file]["nuclear_features"]["solidity"].append(round(p['solidity'], 3))
                    
                    major_axis = p['major_axis_length'] * self.data["files"][file]['metadata']['XScale']
                    self.data["files"][file]["nuclear_features"]["major_axis"].append(round(major_axis, 1))
                    
                    minor_axis = p['minor_axis_length'] * self.data["files"][file]['metadata']['XScale']
                    self.data["files"][file]["nuclear_features"]["minor_axis"].append(round(minor_axis, 1))                  
        
                    axes_ratio = minor_axis / major_axis
                    self.data["files"][file]["nuclear_features"]["axes_ratio"].append(round(axes_ratio, 3))
                    
                    cY, cX = p['centroid']
                    self.data["files"][file]["nuclear_features"]["x_pos"].append(round(cX))
                    self.data["files"][file]["nuclear_features"]["y_pos"].append(round(cY))
                    
                    self.data["files"][file]["nuclear_features"]["source"].append(file)
            
            
    def generate_dict_nf(self):
        """
        Creates dictionary for the nuclear features

        Returns
        -------
        dct_df : dict
            Dictionary that will containd the values of the nuclear features.

        """
        
        dct_df = {
            'cell_no': [], 
            f'avg_intensity_{self.data["dna_marker"]}': [], 
            'nuclear_area': [], 
            'nuclear_perimeter': [], 
            'major_axis': [], 
            'minor_axis': [],
            'axes_ratio': [], 
            'circularity': [], 
            'eccentricity': [],
            'solidity': [], 
            'x_pos': [], 
            'y_pos': [],
            'source': []
            }
        
        return dct_df
    
    
    def plot_boxplot_hist(self, feature = "nuclear_area"):
        """
        Generates boxplot and histogram for a desired nuclear feature.

        Parameters
        ----------
        feature : string, optional
            Desired nuclear feature to show. The default is "nuclear_area".

        Returns
        -------
        fig : plot
            Boxplot-Histogram.

        """
        
        ft2show = [l for file in self.data["files"] for l in self.data["files"][file]["nuclear_features"][feature]]
        
        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True)
        fig.add_trace(go.Box(x = ft2show, boxpoints = 'suspectedoutliers', fillcolor = 'rgba(7,40,89,0.5)', 
                             line_color = 'rgb(7,40,89)', showlegend = False, name = ''), row = 1, col = 1)
        fig.add_trace(go.Histogram(x = ft2show, histnorm = 'probability', marker_color = 'rgb(7,40,89)',
                                  name = feature, showlegend = False), row = 2, col = 1)
        fig.update_layout(title = f"{feature} distribution", 
                          xaxis = dict(autorange = True, showgrid = True, zeroline = True, gridwidth=1), width = 1000, 
                          height = 400, template = "plotly_white")
        
        return fig
    
    
    def print_features(self):
        """
        Prints measured nuclear features.

        Returns
        -------
        None.

        """
        for file in self.data["files"]:
            for ft in self.data["files"][file]["nuclear_features"]:
                if ft != "cell_no" and ft != "source":
                    print(ft)
            break
        
        
    def print_files(self):
        """
        Prints files.

        Returns
        -------
        None.

        """
        for file in self.data["files"]:
            print(file)
        
        
    def add_nf(self):
        """
        Generate a list containing additional nuclear features.

        Returns
        -------
        fts2add : list
            List of additional nuclear features.

        """
        fts2add = []
        
        for ch in self.data["channels_info"]:
            if ch == self.data["dna_marker"]:
                fts2add.append(f"avg_intensity_core_{ch}")
                fts2add.append(f"avg_intensity_internal_ring_{ch}")
                fts2add.append(f"avg_intensity_external_ring_{ch}")
                fts2add.append(f"total_intensity_core_{ch}")
                fts2add.append(f"total_intensity_internal_ring_{ch}")
                fts2add.append(f"total_intensity_external_ring_{ch}")
                fts2add.append(f"total_intensity_{ch}")
            else:
                fts2add.append(f"avg_intensity_{ch}")
                fts2add.append(f"total_intensity_{ch}")
                
        return fts2add
    
    
    def add_nf2file(self):
        """
        Add additional nuclear features to dictionaries of files.

        Returns
        -------
        None.

        """
        add_nf = self.add_nf()
        for file in self.data["files"]:
            for ft in add_nf:
                self.data["files"][file]["nuclear_features"][ft] = []
    
    
    def add_nuclear_features(self):
        """
        Measure additional nuclear features.

        Returns
        -------
        None.

        """
        self.add_nf2file()
        
        for file in tqdm(self.data["files"]):
            for cell in self.data["files"][file]["nuclear_features"]["cell_no"]:
                for ch in self.data["channels_info"]:
                    mask = self.data["files"][file]["masks"]
                    image = self.data["files"][file]['working_array'][self.data["channels_info"][ch]]
                    if ch == self.data["dna_marker"]:
                        core, internal_ring, external_ring  = nucleus_layers(image,
                                                                             mask, 
                                                                             cell_no = cell, 
                                                                             xscale = self.data["files"][file]['metadata']['XScale'])
                        self.data["files"][file]["nuclear_features"][f"avg_intensity_core_{ch}"].append(find_avg_intensity(image, core, cell_no = cell))
                        self.data["files"][file]["nuclear_features"][f"avg_intensity_internal_ring_{ch}"].append(find_avg_intensity(image, internal_ring, cell_no = cell))
                        self.data["files"][file]["nuclear_features"][f"avg_intensity_external_ring_{ch}"].append(find_avg_intensity(image, external_ring, cell_no = cell))
                        self.data["files"][file]["nuclear_features"][f"total_intensity_{ch}"].append(find_sum_intensity(image, mask, cell_no = cell))
                        self.data["files"][file]["nuclear_features"][f"total_intensity_core_{ch}"].append(find_sum_intensity(image, core, cell_no = cell))
                        self.data["files"][file]["nuclear_features"][f"total_intensity_internal_ring_{ch}"].append(find_sum_intensity(image, internal_ring, cell_no = cell))
                        self.data["files"][file]["nuclear_features"][f"total_intensity_external_ring_{ch}"].append(find_sum_intensity(image, external_ring, cell_no = cell))    
                    else:
                        self.data["files"][file]["nuclear_features"][f"avg_intensity_{ch}"].append(find_avg_intensity(image, mask, cell_no = cell))
                        self.data["files"][file]["nuclear_features"][f"total_intensity_{ch}"].append(find_sum_intensity(image, mask, cell_no = cell))
                        
                        
    def positive2marker(self, frac_covered = 0.8, thresh_method = "triangle"):
        """
        Identify whether a cell is positive to a nuclear marker.

        Parameters
        ----------
        frac_covered : float, optional
            Fraction of the nucleus covered by marker. The default is 0.8.
        thresh_option : str, optional
            Thresholding method. The default is "triangle".

        Returns
        -------
        None.

        """
        for ch in tqdm(self.data["channels_info"]):
            if ch == self.data["dna_marker"]:
                continue
            for file in self.data["files"]:
                
                image = self.data["files"][file]['working_array'][self.data["channels_info"][ch]]
                th_img = get_threshold(image, thresh_method)
                bin_img = binary_img(th_img)
                
                mask = self.data["files"][file]["masks"]
                
                self.data["files"][file]["nuclear_features"][f"{ch}_positive"] = []
                
                for cell in self.data["files"][file]["nuclear_features"]["cell_no"]:
                    if cell != 0:
                        avg_binary = np.average(bin_img[mask == cell])
                        if avg_binary >= frac_covered:
                            self.data["files"][file]["nuclear_features"][f"{ch}_positive"].append(True)
                        else:
                            self.data["files"][file]["nuclear_features"][f"{ch}_positive"].append(False)
                   
                            
    def get_lst_features(self):
        """
        Gets list of measured features.
        
        Returns
        -------
        lst_fts : list
            list of features measured.

        """
        lst_fts = []
        
        for file in self.data["files"]:
            for ft in self.data["files"][file]["nuclear_features"]:
                lst_fts.append(ft)
            break
        
        return lst_fts
           
                            
    def export_csv(self, filename = "raw_output.csv"):
        """
        Export data generated as CSV

        Parameters
        ----------
        filename : str, optional
            Name of output file. The default is "output.csv".

        Returns
        -------
        None.

        """
        if not filename.endswith(".csv"):
            filename = filename + ".csv"
        
        lst_fts = self.get_lst_features()
        
        dct_df = {}
        
        for ft in lst_fts:
            dct_df[ft] = [l for file in self.data["files"] for l in self.data["files"][file]["nuclear_features"][ft]]
        
        df_out = pd.DataFrame.from_dict(data = dct_df)
        df_out.to_csv(self.path_save + filename, index = False)
        print(f"CSV file saved as: {self.path_save + filename}")
        
    
    def toJSON(self, filename = "output.json.gz"):
        """
        Save all data (including metadata) as compressed a JSON file

        Parameters
        ----------
        filename : str, optional
            Name of output file. The default is "output.json.gz".

        Returns
        -------
        None.

        """
        if not filename.endswith(".json.gz"):
            filename = filename + ".json.gz"
            
        print("This step might take some minutes...")
        
        compress_json.dump(self.data, 
                           self.path_save + filename, 
                           json_kwargs = {"sort_keys": True, 
                                          "indent": 4, 
                                          "cls": NumpyEncoder
                                          }
                           )
    
        print(f"\nData saved as: {self.path_save + filename}")
    

#########################################
#     Functions & Classes | Analysis    #
#########################################

def selection2df(table):
    """
    Generates pandas dataframe from selection table.

    Parameters
    ----------
    table : Plotly Table
        Nuclear Features Data as Plotly Table.

    Returns
    -------
    df_out : pandas dataframe
        DataFrame from selection.

    """
    d = table.to_dict()
    df_out = pd.DataFrame(d['data'][0]['cells']['values'], index = d['data'][0]['header']['values']).T
    
    return df_out


class NuclearGame_Analysis(object):
    
    def __init__(self, arg):
        """
        Start NuclearGame Analysis

        Parameters
        ----------
        arg : str
            Path to compressed JSON file generated in segmentation.

        Raises
        ------
        ValueError
            If path to JSON file is not correct.

        Returns
        -------
        None.

        """
        if not arg.lower().endswith(".json.gz"):
            raise ValueError("Ops! Invalid format...")
            
        self.path2json = arg
        self.path_save = os.path.dirname(self.path2json) + "/"
        
        print("Reading JSON file, this step might take some minutes...", end = "  ")
        self.data = compress_json.load(self.path2json)
        print("DONE")
        
        self.df_raw = self.generate_nf_df()
        
        for file in self.data["files"]:
            self.data["files"][file]["array"] = np.asarray(self.data["files"][file]["array"])
            self.data["files"][file]['working_array'] = np.asarray(self.data["files"][file]['working_array'])
            self.data["files"][file]["masks"] = np.asarray(self.data["files"][file]["masks"])
            
            
    def get_lst_features(self):
        """
        Gets list of measured features.
        
        Returns
        -------
        lst_fts : list
            list of features measured.

        """
        lst_fts = []
        
        for file in self.data["files"]:
            for ft in self.data["files"][file]["nuclear_features"]:
                lst_fts.append(ft)
            break
        
        return lst_fts
    
    
    def generate_nf_df(self):
        """
        Creates DataFrame with raw data.

        Returns
        -------
        df : pandas DataFrame
            DataFrame containing nuclear features raw data.

        """
        lst_fts = self.get_lst_features()
        
        dct_df = {}
        
        for ft in lst_fts:
            dct_df[ft] = [l for file in self.data["files"] for l in self.data["files"][file]["nuclear_features"][ft]]
        
        df = pd.DataFrame.from_dict(data = dct_df)
        
        return df

    
    def scatter_widget(self, feature1, feature2, df = None, xlog = False, ylog = False):
        """
        Generates Scatter Widget for data selection

        Parameters
        ----------
        feature1 : str
            Nuclear feature 1.
        feature2 : str
            Nuclear feature 2.
        xlog : bool, optional
            True for X axis to be in log scale. The default is False.
        ylog : TYPE, optional
            True for Y axis to be in log scale. The default is False.

        Returns
        -------
        f : Figure Widget
            Scatterplot Widget of selescted nuclear features.

        """
        
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw
            
        f = go.FigureWidget([go.Scatter(y = df[feature1], 
                                        x = df[feature2], 
                                        mode = 'markers')])
        
        if xlog == False and ylog == False:
            
            f.update_layout(
                title = f'Distribution of {feature1} VS {feature2}',
                xaxis = dict(title = feature2),
                yaxis = dict(title = feature1),
                autosize = True,
                hovermode = 'closest',
                template = "plotly_white"
                )   
        
        elif xlog == True and ylog == False:
            
            f.update_layout(
                title = f'Distribution of {feature1} VS {feature2}',
                xaxis = dict(title = feature2, type = "log"),
                yaxis = dict(title = feature1),
                autosize = True,
                hovermode = 'closest',
                template = "plotly_white"
                )   
            
        elif xlog == False and ylog == True:
            
            f.update_layout(
                title = f'Distribution of {feature1} VS {feature2}',
                xaxis = dict(title = feature2),
                yaxis = dict(title = feature1, type = "log"),
                autosize = True,
                hovermode = 'closest',
                template = "plotly_white"
                )
        
        elif xlog == True and ylog == True:
            
            f.update_layout(
                title = f'Distribution of {feature1} VS {feature2}',
                xaxis = dict(title = feature2, type = "log"),
                yaxis = dict(title = feature1, type = "log"),
                autosize = True,
                hovermode = 'closest',
                template = "plotly_white"
                )
        
        scatter = f.data[0]
        
        t = go.FigureWidget([go.Table(
        header=dict(values = df.columns,
                    fill = dict(color='#C2D4FF'),
                    align = ['left'] * 5),
        cells=dict(values=[df[col] for col in df.columns],
                   fill = dict(color='#F5F8FF'),
                   align = ['left'] * 5))])
        
        def selection_fn(trace,points,selector):
            t.data[0].cells.values = [df.loc[points.point_inds][col] for col in df.columns]
            
        scatter.on_selection(selection_fn)
        
        return f, t
    
    
    def print_features(self):
        """
        Prints measured nuclear features.

        Returns
        -------
        None.

        """
        for file in self.data["files"]:
            for ft in self.data["files"][file]["nuclear_features"]:
                if ft != "cell_no" and ft != "source":
                    print(ft)
            break


    def plot_boxplot_hist(self, feature = "nuclear_area", df = None):
        """
        Generates boxplot and histogram for a desired nuclear feature.

        Parameters
        ----------
        feature : string, optional
            Desired nuclear feature to show. The default is "nuclear_area".

        Returns
        -------
        fig : plot
            Boxplot-Histogram.

        """
        
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw
            
        ft2show = df[feature].to_list()
        
        fig = make_subplots(rows = 2, cols = 1, shared_xaxes = True)
        fig.add_trace(go.Box(x = ft2show, boxpoints = 'suspectedoutliers', fillcolor = 'rgba(7,40,89,0.5)', 
                             line_color = 'rgb(7,40,89)', showlegend = False, name = ''), row = 1, col = 1)
        fig.add_trace(go.Histogram(x = ft2show, histnorm = 'probability', marker_color = 'rgb(7,40,89)',
                                  name = feature, showlegend = False), row = 2, col = 1)
        fig.update_layout(title = f"{feature} distribution", 
                          xaxis = dict(autorange = True, showgrid = True, zeroline = True, gridwidth=1), width = 1000, 
                          height = 400, template = "plotly_white")
        
        return fig
    
    
    def filter_data(self, feature, min_value = None, max_value = None, df = None):
        """
        Filter data with lower and higher thresholds.

        Parameters
        ----------
        feature : str
            Nuclear feature.
        min_value : float, optional
            Minimum value required for the given nuclear feature. The default is None.
        max_value : TYPE, optional
            Maximum value required for the given nuclear feature. The default is None.
        df : pandas dataframe, optional
            DataFrame containing the nuclear features. The default is None.

        Returns
        -------
        df_out : pandas dataframe
            Filtered DataFrame containing the nuclear features.

        """
        if not isinstance(df, pd.DataFrame):
            df = self.df_raw
        
        before = len(df)
        
        if min_value != None and max_value != None:
            df_out = df[(df[feature] >= min_value) & (df[feature] <= max_value)]
        elif min_value == None and max_value != None:
            df_out = df[(df[feature] <= max_value)]
        elif min_value != None and max_value == None:
            df_out = df[(df[feature] >= min_value)]
        elif min_value == None and max_value == None:
            df_out = df
        
        after = len(df_out)
        
        print(f"{before - after} cells were removed with the filter.")
        
        return df_out
    
    
    def find_dna_dots(self, df, box_size = 10):
        """
        Finds number of DNA dots

        Parameters
        ----------
        df : pandas dataframe
            DataFrame containing nuclear features.
        box_size : int, optional
            Side size (px) of box for finding high intensity dots. The default is 10.

        Returns
        -------
        df : pandas dataframe
            DataFrame containing nuclear features and number of DNA dots.

        """
        lst_dna_dots = []
        
        n = 1
        
        for index, row in df.iterrows():
        
            masks = self.data["files"][row["source"]]["masks"].copy()
            masks[masks != row['cell_no']] = 0
            masks[masks == row['cell_no']] = 1
            
            nucleus = self.data["files"][row["source"]]['working_array'][self.data["channels_info"][self.data["dna_marker"]]].copy()
            nucleus[masks != 1] = 0
            
            cY = int(row['y_pos'])
            cX = int(row['x_pos'])
            cY_low = cY - 150
            cY_high = cY + 150
            cX_low = cX - 150
            cX_high = cX + 150
            if (cY-150) < 0:
                cY_low = 0
            if (cY+150) > len(nucleus):
                cY_high = len(nucleus)
            if (cX-150) < 0:
                cX_low = 0
            if (cX+150) > len(nucleus[0]):
                cX_high = len(nucleus[0])
            nucleus = nucleus[cY_low:cY_high, cX_low:cX_high]
            masks = masks[cY_low:cY_high, cX_low:cX_high]
                    
            ignore_mask = np.zeros(masks.shape)
            ignore_mask[masks == 0] = True
            ignore_mask[masks != 0] = False
            ignore_mask = ignore_mask.astype(bool)
            
            bkg = Background2D(nucleus, 3, mask = ignore_mask)
    
            th = detect_threshold(data = nucleus, nsigma = 0, mask_value = 0, background = bkg.background)
        
            peak_tb = find_peaks(data = nucleus, threshold = th, mask = ignore_mask, box_size = box_size)
            
            try: 
                peak_df = peak_tb.to_pandas()
                lst_remove = []
    
                for index2, row2 in peak_df.iterrows():
                    x_up = row2["x_peak"] + 5
                    x_down = row2["x_peak"] - 5
                    y_up = row2["y_peak"] + 5
                    y_down = row2["y_peak"] -5
                    temp_df = peak_df[((peak_df["x_peak"] > x_down) & (peak_df["x_peak"] < x_up)) & ((peak_df["y_peak"] > y_down) & (peak_df["y_peak"] < y_up))]
                    if len(temp_df) > 1:
                        sorted_df = temp_df.sort_values(by = "peak_value", ascending = False)
                        flag = True
                        for index3, row3 in sorted_df.iterrows():
                            if flag == True:
                                flag = False
                                pass
                            elif flag == False:
                                lst_remove.append(index3)
    
                peak_df_fltd = peak_df.drop(lst_remove)
    
                no = len(peak_df_fltd)
                lst_dna_dots.append(no)
                
            except:
                lst_dna_dots.append(0)
            
            print(f"Progress: {n} / {len(df)}")
            n += 1
            
        df["dna_dots"] = lst_dna_dots
            
        return df
    
    
    def spatial_entropy(self, df, d = 5):
        """
        Finds spatial entropy for each nucleus.

        Parameters
        ----------
        df : pandas dataframe
            DataFrame containing nuclear features.
        d : int, optional
            Side size (px) of box for finding co-occurrences. The default is 5.

        Returns
        -------
        df : pandas dataframe
            DataFrame containing nuclear features and spatial entropy.

        """
        lst_entropies = []
        
        n = 1
        
        for index, row in df.iterrows():
            
            masks = self.data["files"][row["source"]]["masks"].copy()
            masks[masks != row['cell_no']] = 0
            masks[masks == row['cell_no']] = 1
            
            nucleus = self.data["files"][row["source"]]['working_array'][self.data["channels_info"][self.data["dna_marker"]]].copy()
            nucleus[masks != 1] = 0
            
            cY = int(row['y_pos'])
            cX = int(row['x_pos'])
            cY_low = cY - 150
            cY_high = cY + 150
            cX_low = cX - 150
            cX_high = cX + 150
            if (cY-150) < 0:
                cY_low = 0
            if (cY+150) > len(nucleus):
                cY_high = len(nucleus)
            if (cX-150) < 0:
                cX_low = 0
            if (cX+150) > len(nucleus[0]):
                cX_high = len(nucleus[0])
            nucleus = nucleus[cY_low:cY_high, cX_low:cX_high]
            masks = masks[cY_low:cY_high, cX_low:cX_high]
            
            pp = np.array([[nx, ny] for ny in range(len(nucleus)) for nx in range(len(nucleus[ny])) if nucleus[ny][nx] != 0])
            
            lst_types = []
            
            p10, p20, p30, p40, p50, p60, p70, p80, p90 = np.percentile(nucleus[nucleus != 0], [10, 20, 30, 40, 50, 60, 70, 80, 90])
            
            for l in pp:
                x, y = l
                if nucleus[y][x] < p10:
                    lst_types.append("1")
                elif nucleus[y][x] >= p10 and nucleus[y][x] < p20:
                    lst_types.append("2")
                elif nucleus[y][x] >= p20 and nucleus[y][x] < p30:
                    lst_types.append("3")
                elif nucleus[y][x] >= p30 and nucleus[y][x] < p40:
                    lst_types.append("4")  
                elif nucleus[y][x] >= p40 and nucleus[y][x] < p50:
                    lst_types.append("5")
                elif nucleus[y][x] >= p50 and nucleus[y][x] < p60:
                    lst_types.append("6")
                elif nucleus[y][x] >= p60 and nucleus[y][x] < p70:
                    lst_types.append("7") 
                elif nucleus[y][x] >= p70 and nucleus[y][x] < p80:
                    lst_types.append("8")
                elif nucleus[y][x] >= p80 and nucleus[y][x] < p90:
                    lst_types.append("9") 
                else:
                    lst_types.append("10")
            
            types = np.array(lst_types)
            
            lb_ent = leibovici_entropy(pp, types, d)
            
            lst_entropies.append(round(lb_ent.entropy, 3))
        
            print(f"Progress: {n} / {len(df)}")
            n += 1
            
        df["spatial_entropy"] = lst_entropies
            
        return df


    def toCSV(self, df, filename = "filtered_output.csv"):
        """
        Export data generated as CSV

        Parameters
        ----------
        filename : str, optional
            Name of output file. The default is "filtered_output.csv".

        Returns
        -------
        None.

        """
        if not filename.endswith(".csv"):
            filename = filename + ".csv"
        
        df.to_csv(self.path_save + filename, index = False)
        print(f"CSV file saved as: {self.path_save + filename}")
        


###########################################
#     Functions & Classes | Clustering    #
###########################################
    


def calculate_wcss(data, show_plot = False):
    """
    Calculate Sum Squared Distances

    Parameters
    ----------
    data : pandas dataframe values
        Nuclear Features values.

    Returns
    -------
    wcss : lst
        list of kmeans inertia.

    """
    wcss = []
    K = range(2, 21)
    for n in K:
        kmeans = KMeans(n_clusters=n)
        kmeans.fit(X=data)
        wcss.append(kmeans.inertia_)
    
    if show_plot == True:
        plt.plot(K, wcss, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    return wcss


def optimal_number_of_clusters(wcss):
    """
    Find optimal number of clusters with Kmeans

    Parameters
    ----------
    wcss : lst
        list of kmeans inertia.

    Returns
    -------
    int
        Optimal number of clusters.

    """
    x1, y1 = 2, wcss[0]
    x2, y2 = 21, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2


class NuclearGame_Clustering(object):
    
    def __init__(self, arg):
        """
        Start NuclearGame - Clustering

        Parameters
        ----------
        arg : str
            Path to CSV file containing nuclear features.

        Returns
        -------
        None.

        """
        self.path = arg
        self.data = pd.read_csv(self.path)
        
        
    def scale_data(self, features, method = "StandardScaler"):
        """
        Scale data

        Parameters
        ----------
        features : list
            List containing nuclear features to be considered for dimension reduction.
        method : str, optional
            Method for scaling (StandardScaler/MinMaxScaler/MaxAbsScaler/None). 
            The default is "StandardScaler".

        Returns
        -------
        None.

        """

        self.features = features
        
        # Copy DataFrame
        self.data_notna = self.data.copy()
        
        # Remove cells that contain NAN values
        for ft in self.features:
            self.data_notna = self.data_notna[self.data_notna[ft].notna()]
        
        # Obtain relevant DataSet  
        self.rel_data = self.data_notna.loc[:, self.features]
        
        # Obtain values of relevant DataSet
        self.values = self.rel_data.values
        
        # Scale data
        if method.lower() == "standardscaler":
            self.scaled_data = StandardScaler().fit_transform(self.values) # StandardScaler
        elif method.lower() == "minmaxscaler":
            self.scaled_data = MinMaxScaler().fit_transform(self.values) # MinMaxScaler
        elif method.lower() == "maxabsscaler":
            self.scaled_data = MaxAbsScaler().fit_transform(self.values) # MaxAbsScaler
        else:
            self.scaled_data = self.values
            
    
    def umap_reduction(self, show_plot = False, size = 10):
        """
        Perform UMAP dimension reduction.

        Parameters
        ----------
        show_plot : bool, optional
            True for showing UMAP plot. The default is False.

        Returns
        -------
        None.

        """
        reducer = umap.UMAP()

        embedding = reducer.fit_transform(self.scaled_data)
        
        self.principalDf = pd.DataFrame(data = embedding, columns = ['UMAP 1', 'UMAP 2'])
        
        if show_plot == True:
            plt.scatter(self.principalDf["UMAP 1"], self.principalDf["UMAP 2"], s = size, cmap='Spectral')
            plt.xlabel("UMAP 2"); plt.ylabel("UMAP 1"); plt.title("UMAP")
            plt.show()
            
    
    def optimalClusters(self, show_plot = False):
        """
        Calculate optimal number of clusters

        Parameters
        ----------
        show_plot : bool, optional
            Show plot. The default is False.
            
        Returns
        -------
        None.

        """
        # Calculating the within clusters sum-of-squares for 20 cluster amounts
        sum_of_squares = calculate_wcss(self.scaled_data, show_plot)
        
        # Calculating the optimal number of clusters
        self.no_clusters = optimal_number_of_clusters(sum_of_squares)
        print(f"\nOptimal number of clusters: {self.no_clusters}")
        
        
    def clusterableEmbedding(self, n_neighbors = 50, min_dist = 0.0, n_components = 2):
        """
        Obtain a clusterable embedding

        Parameters
        ----------
        n_neighbors : int, optional
            Number of neighbors. The default is 50.
        min_dist : float, optional
            Minimum distance. The default is 0.0.
        n_components : int, optional
            Number of components. The default is 2.

        Returns
        -------
        None.

        """
        self.clusterable_embedding = umap.UMAP(
            n_neighbors = n_neighbors,
            min_dist = min_dist,
            n_components = n_components,
            ).fit_transform(self.scaled_data)
        
    
    def kmeans_clustering(self, show_plot = False, size = 10):
        """
        Perform kmeans clustering

        Parameters
        ----------
        show_plot : bool, optional
            Show plot with clustering. The default is False.

        Returns
        -------
        None.

        """
        self.kmeans_labels = KMeans(n_clusters = self.no_clusters).fit_predict(self.clusterable_embedding)
        cdict = {0: 'grey', 1: 'red', 2: 'blue', 3: 'green', 4: 'pink', 5: 'orange', 6: 'yellow', 7: 'saddlebrown', 8: 'purple',
                9: 'magenta'}
        
        if show_plot == True:
            fig, ax = plt.subplots(figsize = (6.4, 4.3))
            for g in np.unique(self.kmeans_labels):
                ix = np.where(self.kmeans_labels == g)
                ax.scatter(np.array(self.principalDf["UMAP 1"])[ix], np.array(self.principalDf["UMAP 2"])[ix], c = cdict[g], 
                           label = g, s = size)
            ax.legend(bbox_to_anchor = (1.02, 1), loc = 'upper left'); ax.set_xlabel("UMAP 2"); ax.set_ylabel("UMAP 1"); ax.set_title("kMeans Clustering")
            plt.tight_layout()
            plt.show()
        
        
    def assignCluster(self):
        """
        Assigns cluster number to cells.

        Returns
        -------
        pandas dataframe
            Nuclear features dataframe containing cluster numbers.

        """
        self.data_notna["cluster"] = [c for c in self.kmeans_labels]
        
        return self.data_notna
    
    
    def scatter_heatmap(self, feature, size = 10, cmap = "RdBu_r"):
        """
        Generate UMAP scatter plot heatmap.

        Parameters
        ----------
        feature : str
            Nuclear Feature to plot as Scatter plot heatmap.
        size : int, optional
            Circles size. The default is 10.
        cmap : Color map, optional
            Color map for heatmap. The default is "RdBu_r".

        Returns
        -------
        fig : matplot fig
            UMAP scatter plot heatmap for input feature.

        """
        fig, ax = plt.subplots()
        
        q3, q1 = np.percentile(self.data_notna[feature], [75 ,25])
        iqr = q3 - q1
        _max = q3 + (1.5 * iqr)
        _min = q1 - (1.5 * iqr)
        
        norm = plt.Normalize(_min, _max, clip = True)
        sca = ax.scatter(self.principalDf["UMAP 1"], 
                         self.principalDf["UMAP 2"], 
                         c = self.data_notna[feature],
                         s = size,
                         cmap = cmap,
                         norm = norm
                         )
        ax.set_xlabel("UMAP 2"); ax.set_ylabel("UMAP 1"); ax.set_title(f"UMAP | {feature}")
        position = fig.add_axes([1.02, 0.80, 0.012, 0.15])
        fig.colorbar(sca, cax = position)
        #plt.tight_layout()
        
        return fig
        
    
        