# coding=utf-8
'''
todo list:
1. provide color ramps options
2. add band palette to tif # https://gis.stackexchange.com/questions/325615/store-geotiff-with-color-table-python
3. add user defined color ramps
4. interpolate color ramps
5. vector format support
6. triangle mesh support
'''

# from lytools import *
import xycmap
import numpy as np
import PIL.Image as Image
from matplotlib import pyplot as plt
from osgeo import osr
from osgeo import gdal
import pandas as pd
from os.path import *


class Bivariate_plot:

    def __init__(self):

        pass

    def gen_zcmap(self, n):
        corner_colors = ("#1D2D69", '#A1FF64', "#E9FF64", "#BD2ECC")
        zcmap = xycmap.custom_xycmap(corner_colors=corner_colors, n=n)
        return zcmap
        pass

    def plot_bivariate_map(self, tif1, tif2, tif1_label, tif2_label, min1, max1, min2, max2, outf,
                           n=(5,5), n_legend=(101,101), zcmap=None, legend_title=''):
        '''
        :param tif1: input tif1
        :param tif2: input tif2
        :param tif1_label: tif1 label
        :param tif2_label: tif2 label
        :param min1: min value of tif1
        :param max1: max value of tif1
        :param min2: min value of tif2
        :param max2: max value of tif2
        :param outf: output file
        :param n: number of colors
        :param legend_title: legend title
        :output: bivariate map
        '''
        if zcmap is None:
            zcmap = self.gen_zcmap(n)
            n = (zcmap.shape[0], zcmap.shape[1])
            zcmap_legend = self.gen_zcmap(n_legend)
        else:
            zcmap_legend = zcmap

        arr1 = GDAL_func().raster2array(tif1)
        arr2 = GDAL_func().raster2array(tif2)
        arr1 = np.array(arr1)
        arr2 = np.array(arr2)
        arr1[arr1 < -99999] = np.nan
        arr2[arr2 < -99999] = np.nan

        bin1 = np.linspace(min1, max1, n[0] + 1)
        bin2 = np.linspace(min2, max2, n[1] + 1)

        spatial_dict1 = GDAL_func().spatial_arr_to_dic(arr1)
        spatial_dict2 = GDAL_func().spatial_arr_to_dic(arr2)
        dict_all = {'arr1': spatial_dict1, 'arr2': spatial_dict2}
        df = GDAL_func().spatial_dics_to_df(dict_all)

        blend_arr = []

        for r in range(len(arr1)):
            temp = []
            for c in range(len(arr1[0])):
                val1 = arr1[r][c]
                val2 = arr2[r][c]
                if val1 > max1:
                    val1 = max1
                if val1 < min1:
                    val1 = min1
                if val2 > max2:
                    val2 = max2
                if val2 < min2:
                    val2 = min2
                if np.isnan(val1) or np.isnan(val2):
                    temp.append(np.array([1, 1, 1, 1]))
                    continue
                for i in range(len(bin1) - 1):
                    if val1 >= bin1[i] and val1 <= bin1[i + 1]:
                        for j in range(len(bin2) - 1):
                            if val2 >= bin2[j] and val2 <= bin2[j + 1]:
                                color = zcmap[j][i] * 255
                                temp.append(color)
            temp = np.array(temp)
            blend_arr.append(temp)
        blend_arr = np.array(blend_arr)
        # print(np.shape(blend_arr))
        # exit()
        # newRasterfn = '/Volumes/NVME2T/greening_project_redo/Result/late_lai_ml/tif/Moving_window_single_correlation/test.tif'
        img = Image.fromarray(blend_arr.astype('uint8'), 'RGBA')
        img.save(outf)
        # define a projection and extent
        raster = gdal.Open(outf)
        geotransform = raster.GetGeoTransform()
        originX, originY, pixelWidth, pixelHeight = GDAL_func().get_raster_transformations(tif1)
        raster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outRasterSRS = osr.SpatialReference()
        projection = GDAL_func().get_raster_projections(tif1)
        # outRasterSRS.ImportFromEPSG(4326)
        # outRasterSRS.ImportFromEPSG(projection)
        # raster.SetProjection(outRasterSRS.ExportToWkt())
        raster.SetProjection(projection)

        x_ticks = []
        y_ticks = []
        bin1 = np.linspace(min1, max1, n_legend[0] + 1)
        bin2 = np.linspace(min2, max2, n_legend[1] + 1)
        for i in range(len(bin1) - 1):
            for j in range(len(bin2) - 1):
                x_ticks.append((bin1[i] + bin1[i + 1]) / 2)
                y_ticks.append((bin2[j] + bin2[j + 1]) / 2)
        x_ticks = list(set(x_ticks))
        y_ticks = list(set(y_ticks))
        x_ticks.sort()
        y_ticks.sort()
        x_ticks = [round(x, 2) for x in x_ticks]
        y_ticks = [round(y, 2) for y in y_ticks]
        zcmap_255 = zcmap_legend * 255
        zcmap_255 = zcmap_255.astype('uint8')
        zcmap_legend = zcmap_255
        plt.imshow(zcmap_legend)
        if len(x_ticks) > 100:
            plt.xticks(list(range(len(x_ticks)))[::10], x_ticks[::10], rotation=90)
            plt.yticks(list(range(len(y_ticks)))[::10], y_ticks[::10])
        elif len(x_ticks) > 50:
            plt.xticks(list(range(len(x_ticks)))[::5], x_ticks[::5], rotation=90)
            plt.yticks(list(range(len(y_ticks)))[::5], y_ticks[::5])
        else:
            plt.xticks(list(range(len(x_ticks))), x_ticks, rotation=90)
            plt.yticks(list(range(len(y_ticks))), y_ticks)
        plt.xlabel(tif1_label)
        plt.ylabel(tif2_label)
        plt.title(legend_title)
        plt.tight_layout()
        plt.savefig(outf.replace('.tif', '.pdf'))

class GDAL_func:
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
        band = raster.GetRasterBand(1)
        array = band.ReadAsArray()
        array = np.array(array)
        return array

    def get_raster_transformations(self, rasterfn):
        '''
        get raster transformation
        Agrs:
            rasterfn: tiff file path
        Returns:
            originX: x coordinate of the origin
            originY: y coordinate of the origin
            pixelWidth: width of the pixel
            pixelHeight: height of the pixel
        '''
        raster = gdal.Open(rasterfn)
        geotransform = raster.GetGeoTransform()
        originX = geotransform[0]
        originY = geotransform[3]
        pixelWidth = geotransform[1]
        pixelHeight = geotransform[5]
        return originX, originY, pixelWidth, pixelHeight

    def get_raster_projections(self, rasterfn):
        '''
        get raster projection
        Agrs:
            rasterfn: tiff file path
        Returns:
            projection: projection of the raster
        '''
        raster = gdal.Open(rasterfn)
        projection = raster.GetProjection()
        return projection

    def spatial_arr_to_dic(self, arr):

        pix_dic = {}
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                pix = (i, j)
                val = arr[i][j]
                pix_dic[pix] = val

        return pix_dic

    def spatial_dics_to_df(self, spatial_dic_all):
        unique_keys = []
        for var_name in spatial_dic_all:
            dic_i = spatial_dic_all[var_name]
            for key in dic_i:
                unique_keys.append(key)
        unique_keys = list(set(unique_keys))
        unique_keys.sort()
        dic_all_transform = {}
        for key in unique_keys:
            dic_all_transform[key] = {}
        var_name_list = []
        for var_name in spatial_dic_all:
            var_name_list.append(var_name)
            dic_i = spatial_dic_all[var_name]
            for key in dic_i:
                val = dic_i[key]
                dic_all_transform[key].update({var_name: val})
        df = self.dic_to_df(dic_all_transform, 'pix')
        valid_var_name_list = []
        not_valid_var_name_list = []
        for var_name in var_name_list:
            if var_name in df.columns:
                valid_var_name_list.append(var_name)
            else:
                not_valid_var_name_list.append(var_name)
        df = df.dropna(how='all', subset=valid_var_name_list)
        not_valid_var_name_list.sort()
        for var_name in not_valid_var_name_list:
            df[var_name] = np.nan
        return df


    def dic_to_df(self, dic, key_col_str='__key__', col_order=None):
        '''
        :param dic:
        {
        row1:{col1:val1, col2:val2},
        row2:{col1:val1, col2:val2},
        row3:{col1:val1, col2:val2},
        }
        :param key_col_str: define a Dataframe column to store keys of dict
        :return: Dataframe
        '''
        data = []
        columns = []
        index = []
        if col_order == None:
            all_cols = []
            for key in dic:
                vals = dic[key]
                for col in vals:
                    all_cols.append(col)
            all_cols = list(set(all_cols))
            all_cols.sort()
        else:
            all_cols = col_order
        for key in dic:
            vals = dic[key]
            if len(vals) == 0:
                continue
            vals_list = []
            col_list = []
            vals_list.append(key)
            col_list.append(key_col_str)
            for col in all_cols:
                if not col in vals:
                    val = np.nan
                else:
                    val = vals[col]
                vals_list.append(val)
                col_list.append(col)
            data.append(vals_list)
            columns.append(col_list)
            index.append(key)
        # df = pd.DataFrame(data=data, columns=columns[0], index=index)
        df = pd.DataFrame(data=data, columns=columns[0])
        return df

def main():
    pass

if __name__ == '__main__':
    main()
