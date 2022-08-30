''' ctfeatures computes the color and texture features of an image or set of images '''

# Import libraries
import os
from numpy import mean, var, reshape, uint8, empty, array, append
from pandas import DataFrame
from scipy.stats import skew, kurtosis
from skimage.color import rgb2hsv, rgb2gray
from sklearn.decomposition import PCA
from mahotas.features import haralick


###################################################################################################################################

def get_color_features(image):

    ''' Get color features 
        f1: red mean
        f2: red variance
        f3: red skewness
        f4: red kurtosis
        f5: green mean
        f6: green variance
        f7: green skewness
        f8: green kurtosis
        f9: blue mean
        f10: blue variance
        f11: blue skewness
        f12: blue kurtosis
        f13: hue mean
        f14: hue variance
        f15: hue skewness
        f16: hue kurtosis
        f17: saturation mean
        f18: saturation variance
        f19: saturation skewness
        f20: saturation kurtosis
        f21: value mean
        f22: value variance
        f23: value skewness
        f24: value kurtosis
        f25: first component 1
        f26: first component 2
        f27: first component 3
        f28: second component 1
        f29: second component 2
        f30: second component 3 '''

    # RGB channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]
    
    # HSV channels
    hsv_image = rgb2hsv(image)
    hue_channel = hsv_image[:, :, 0]
    saturation_channel = hsv_image[:, :, 1]
    value_channel = hsv_image[:, :, 2]
    
    # Red channel statistics
    f1 = mean(red_channel)
    f2 = var(red_channel)
    f3 = skew(red_channel, axis=None)
    f4 = kurtosis(red_channel, axis=None)
    
    # Green channel statistics
    f5 = mean(green_channel)
    f6 = var(green_channel)
    f7 = skew(green_channel, axis=None)
    f8 = kurtosis(green_channel, axis=None)
    
    # Blue channel statistics
    f9 = mean(blue_channel)
    f10 = var(blue_channel)
    f11 = skew(blue_channel, axis=None)
    f12 = kurtosis(blue_channel, axis=None)
    
    # Hue channel statistics
    f13 = mean(hue_channel)
    f14 = var(hue_channel)
    f15 = skew(hue_channel, axis=None)
    f16 = kurtosis(hue_channel, axis=None)
    
    # Saturation channel statistics
    f17 = mean(saturation_channel)
    f18 = var(saturation_channel)
    f19 = skew(saturation_channel, axis=None)
    f20 = kurtosis(saturation_channel, axis=None)
    
    # Value channel statistics
    f21 = mean(value_channel)
    f22 = var(value_channel)
    f23 = skew(value_channel, axis=None)
    f24 = kurtosis(value_channel, axis=None)
    
    # PCA
    new_shape = (image.shape[0] * image.shape[1], image.shape[2])
    data = reshape(image, new_shape)
    pca = PCA(n_components=image.shape[2])
    pca.fit(data)
    components = pca.components_
    (f25, f26, f27) = components[0, :]
    (f28, f29, f30) = components[1, :]
    
    # Feature vector
    color_features = array([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, 
                      f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28, f29, f30])

    return color_features

###################################################################################################################################

def get_texture_features(image):
    
    # Transform RGB image to grayscale
    
    gray_image = uint8(rgb2gray(image) * 255)
    haralick_features = haralick(gray_image)
    texture_features = haralick_features.mean(axis=0)
    
    return texture_features

###################################################################################################################################

def get_image_data(image, classification, dimensions):
    
    # Numero de pixeles de la imagen y de las subimagenes
    image_height = image.shape[0]
    image_width = image.shape[1]
    c = image.shape[2]
    (subimage_height, subimage_width) = dimensions
    
    # Cortar los pixeles que sobran de cada dimension para que la imagen sea divisible
    height_remainder = image_height % subimage_height
    width_remainder = image_width % subimage_width
    image = image[height_remainder:, width_remainder:, :]
    
    # Nuevas dimensiones
    image_height = image.shape[0]
    image_width = image.shape[1]
    
    # Crear un array de tamano (n_subimage, subimage_height, subimage_width)
    num_subimages = int(image_height / subimage_height * image_width / subimage_width)
    num_features = 43
    X = empty((num_subimages, num_features))
    y = [''] * num_subimages
    index = 0
    for i in range(int(image_height / subimage_height)):
        for j in range(int(image_width / subimage_width)):
            x_start = i * subimage_height
            x_end = (i + 1) * subimage_height
            y_start = j * subimage_width
            y_end = (j + 1) * subimage_width
            subimage = image[x_start:x_end, y_start:y_end, :]
            
            color_features = get_color_features(subimage)
            texture_features = get_texture_features(subimage)
            
            X[index, 0:30] = color_features
            X[index, 30:43] = texture_features
            y[index] = classification
            
            index += 1
            
    # Create Data Frame to store the color and texture features of the image
    columns = ['red mean', 'red var', 'red skew', 'red kurt', 'green mean', 'green var', 'green skew', 'green kurt', 
               'blue mean', 'blue var', 'blue skew', 'blue kurt', 'hue mean', 'hue var', 'hue skew', 'hue kurt',
               'sat mean', 'sat var', 'sat skew', 'sat kurt', 'val mean', 'val var', 'val skew', 'val kurt',
               'pc1-1', 'pc1-2', 'pc1-3', 'pc2-1', 'pc2-2', 'pc2-3', 'asm', 'contrast',
               'correlation', 'variance', 'inv diff moment', 'sum average', 'sum variance', 'sum entropy', 'entropy', 'diff var',
               'diff entropy', 'imc 1', 'imc 2']
    df = DataFrame(X, columns=columns)
    df['class'] = y
    
    return df

###################################################################################################################################

def count_subimages(path):
    n = 0
    image_classes = os.listdir(path)
    if '.DS_Store' in image_classes:
        image_classes.remove('.DS_Store')
    for image_class in image_classes:
        class_path = path + image_class + '/'
        images = os.listdir(class_path)
        if '.DS_Store' in images:
            images.remove('.DS_Store')
        n += len(images)
    return n
