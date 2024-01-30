# coding=utf-8
'''
todo list:
1. provide color ramps options
2. add band palette to tif # https://gis.stackexchange.com/questions/325615/store-geotiff-with-color-table-python
3. add user defined color ramps
4. interpolate color ramps
5. vector format support
6. triangle mesh support [Done]
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
from scipy.interpolate import griddata

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
        y_ticks = y_ticks[::-1]
        zcmap_255 = zcmap_legend * 255
        zcmap_255 = zcmap_255.astype('uint8')
        zcmap_legend = zcmap_255
        zcmap_legend = zcmap_legend[::-1]
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


class Ternary_plot:

    def __init__(self,
                 res = 1000,
                 top_color = (255, 0, 0), # red
                 left_color = (0, 255, 0), # green
                 right_color = (0, 0, 255), # blue
                 center_color = (255, 255, 255), # white
                 ):
        self.res = res
        self.top_color = top_color
        self.left_color = left_color
        self.right_color = right_color
        self.center_color = center_color
        self.rgb_arr = self.grid_triangle_legend()
        pass

    def run(self):
        self.test()

    def grid_triangle_legend(self):
        top_color = self.top_color
        left_color = self.left_color
        right_color = self.right_color
        center_color = self.center_color
        res = self.res

        top_pos = (0.5, np.cos(np.pi / 6))
        left_pos = (0, 0)
        right_pos = (1, 0)
        center_pos = (0.5, 0.5 * np.tan(np.pi / 6))

        x = [top_pos[0], left_pos[0], right_pos[0], center_pos[0]]
        y = [top_pos[1], left_pos[1], right_pos[1], center_pos[1]]
        band1 = [top_color[0], left_color[0], right_color[0], center_color[0]]
        band2 = [top_color[1], left_color[1], right_color[1], center_color[1]]
        band3 = [top_color[2], left_color[2], right_color[2], center_color[2]]

        grid_x, grid_y = np.mgrid[min(x):max(x):complex(0, res), min(y):max(y):complex(0, res) * np.cos(np.pi / 6)]

        grid_band1 = griddata((x, y), band1, (grid_x, grid_y), method='cubic') / 255
        grid_band2 = griddata((x, y), band2, (grid_x, grid_y), method='cubic') / 255
        grid_band3 = griddata((x, y), band3, (grid_x, grid_y), method='cubic') / 255

        grid_band1[np.isnan(grid_band1)] = 1
        grid_band2[np.isnan(grid_band2)] = 1
        grid_band3[np.isnan(grid_band3)] = 1

        grid_band1[grid_band1 < 0] = 0
        grid_band2[grid_band2 < 0] = 0
        grid_band3[grid_band3 < 0] = 0
        grid_band1[grid_band1 > 1] = 1
        grid_band2[grid_band2 > 1] = 1
        grid_band3[grid_band3 > 1] = 1

        grid_band1 = np.array(grid_band1, dtype=float)
        grid_band2 = np.array(grid_band2, dtype=float)
        grid_band3 = np.array(grid_band3, dtype=float)

        grid_band1_T = grid_band1.T[::-1]
        grid_band2_T = grid_band2.T[::-1]
        grid_band3_T = grid_band3.T[::-1]

        rgb_arr = np.dstack((grid_band1_T, grid_band2_T, grid_band3_T))
        rgb_arr = np.array(rgb_arr, dtype=float)

        return rgb_arr

    def get_point_position(self,x,y,z):
        res = self.res
        point = (x, y, z)
        h = res * np.sin(np.pi / 3)
        point_y = (1 - point[0]) * h
        x_start = res / 2 * x
        x_end = res / 2 + (res / 2 - x_start)
        x_delta = x_end - x_start
        if point[1] + point[2] == 0:
            point_x = x_start
        else:
            point_x = x_start + x_delta * (point[2] / (point[1] + point[2]))
        return point_x, point_y

    def get_color(self,x,y,z):
        rgb_arr = self.rgb_arr
        sum_x_y_z = x + y + z
        if round(sum_x_y_z, 3) != 1:
            raise ValueError(f'sum of x,y,z should be 1\ninput x,y,z: {x}, {y}, {z}')
        point_x, point_y = self.get_point_position(x,y,z)
        r = int(point_y)-1
        c = int(point_x)-1
        if r < 0:
            r = 1
        if c < 0:
            c = 1
        color = rgb_arr[r][c]
        # plt.scatter([int(point_x)], [int(point_y)], c=[color], s=100, edgecolors='gray', zorder=100)
        # plt.text(point_x, point_y, str(point))
        # plt.imshow(rgb_arr)
        # plt.axis('equal')
        # plt.axis('off')
        # plt.show()
        return color

    def test(self):
        x = 1
        y = 0
        z = 0
        color = self.get_color(x, y, z)
        point_x, point_y = self.get_point_position(x, y, z)
        rgb_arr = self.rgb_arr
        plt.imshow(rgb_arr)
        plt.scatter([int(point_x)], [int(point_y)], c=[color], s=1000, edgecolors='k', zorder=100, lw=2, marker="s")
        plt.show()
        pass

class Test:
    def __init__(self):

        pass

    def run(self):
        tif1 = '/Volumes/SSD1T/temp_files/early_peak/early_peak/MCD_trend.tif'
        tif2 = '/Volumes/SSD1T/temp_files/early_peak/late/MCD_trend.tif'
        min1 = -0.0
        max1 = 0.03
        min2 = -0.05
        max2 = 0.05
        outf = '/Volumes/SSD1T/temp_files/early_peak/bivariate_map.tif'
        n = (2, 2)
        n_legend = (2, 2)
        Bivariate_plot().plot_bivariate_map(tif1, tif2, 'early_peak', 'late', min1, max1, min2, max2, outf,
                                            n=n, n_legend=n_legend)

        pass

def main():
    Ternary_plot().test()
    pass

if __name__ == '__main__':
    main()
