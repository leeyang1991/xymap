# coding=utf-8
# from lytools import *
import xycmap
import numpy as np
import PIL.Image as Image
from matplotlib import pyplot as plt
from osgeo import osr
from osgeo import gdal

class Bivariate_plot:


    def __init__(self):
        pass

    def run(self):
        n = (16, 16)
        corner_colors = ("#DD9FC5", '#798AAB', "#F3F3F3", "#8ECCA5")
        zcmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)
        # plt.imshow(zcmap)
        # plt.show()
        tif1 = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/Moving_window_single_correlation/moving_window_trend/early/early_CCI_SM.tif'
        tif2 = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/Moving_window_single_correlation/over_all_corr/early/early_early_CCI_SM.tif'
        arr1,originX, originY, pixelWidth, pixelHeight = ToRaster().raster2array(tif1)
        arr2 = ToRaster().raster2array(tif2)[0]
        arr1[arr1<-999] = np.nan
        arr2[arr2<-999] = np.nan
        max1 = np.nanmax(arr1)
        min1 = np.nanmin(arr1)
        max2 = np.nanmax(arr2)
        min2 = np.nanmin(arr2)
        # min1 = -0.05
        # max1 = 0.05
        # min2 = -0.5
        # max2 = 0.5
        bin1 = np.linspace(min1,max1,n[0]+1)
        bin2 = np.linspace(min2,max2,n[1]+1)

        spatial_dict1 = DIC_and_TIF().spatial_arr_to_dic(arr1)
        spatial_dict2 = DIC_and_TIF().spatial_arr_to_dic(arr2)
        dict_all = {'arr1':spatial_dict1,'arr2':spatial_dict2}
        df = T.spatial_dics_to_df(dict_all)
        # print(df)
        # exit()


        blend_arr = []
        for r in range(len(arr1)):
            temp = []
            for c in range(len(arr1[0])):
                val1 = arr1[r][c]
                val2 = arr2[r][c]
                if np.isnan(val1) or np.isnan(val2):
                    temp.append(np.array([1,1,1,1]))
                    continue
                for i in range(len(bin1)-1):
                    if val1 >= bin1[i] and val1 <= bin1[i+1]:
                        for j in range(len(bin2)-1):
                            if val2 >= bin2[j] and val2 <= bin2[j+1]:
                                # print(zcmap[i][j])
                                color = zcmap[i][j] * 255
                                # print(color)
                                temp.append(color)
            temp = np.array(temp)
            blend_arr.append(temp)
        blend_arr = np.array(blend_arr)
        print(np.shape(blend_arr))
        # exit()
        newRasterfn = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/Moving_window_single_correlation/test.tif'
        img = Image.fromarray(blend_arr.astype('uint8'), 'RGBA')
        img.save(newRasterfn)
        # define a projection and extent
        raster = gdal.Open(newRasterfn)
        geotransform = raster.GetGeoTransform()
        raster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        raster.SetProjection(outRasterSRS.ExportToWkt())
        plt.imshow(zcmap)
        plt.show()

class ToRaster:
    def __init__(self):

        pass

    def raster2array(self, rasterfn):
        '''
        create array from raster
        Agrs:
            rasterfn: tiff file path
        Returns:
            array: tiff data, an 2D array
        '''
        raster = gdal.Open(rasterfn)
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        array = np.asarray(array)
        del raster
        return array, originX, originY, pixelWidth, pixelHeight





