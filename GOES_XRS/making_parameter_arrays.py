import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from matplotlib import pyplot as plt
from scipy import stats as st
import math

class MakingParamArrays:
    
    flare_fits = 'GOES_XRS_historical.fits'
    columns = []
    column_names = []
    diff_columns = []
    diff_column_names = []
    
    def __init__(self):
        fitsfile = fits.open(self.flare_fits)
        self.data = Table(fitsfile[1].data)[:]
        self.header = fitsfile[1].header
        self.xrsb = self.data['xrsb'][:]
        self.xrsa = self.data['xrsa'][:]
        
    def save_xrsb_rise_above_background(self):
        bk = self.data['background flux'][:]
        self.increase = self.xrsb - bk
        self.increase_pct = (self.xrsb-bk)/bk
        self.columns.append(self.increase)
        self.columns.append(self.increase_pct)
        self.column_names.append('Increase above Background')
        self.column_names.append('Increase above Background Fraction')
    
    def save_differences_between_further_points(self, n):
        diff_xrsb_list = []
        diff_xrsb_pct_list = []
        for arr in self.xrsb:
            diff_xrsb = arr[n:] - arr[:-n]
            diff_xrsb = np.concatenate([np.full(n, math.nan), diff_xrsb]) #appending the right amount of zeros to front to make the indices correct
            diff_xrsb_list.append(diff_xrsb)
            diff_xrsb_pct = []
            for i, diffy in enumerate(diff_xrsb):
                diff_xrsb_pct.append(diffy/arr[i]*100)
            diff_xrsb_pct_list.append(diff_xrsb_pct)
        self.columns.append(diff_xrsb_list)
        self.column_names.append(f'XRSB {n}-min Differences')
        self.columns.append(diff_xrsb_pct_list)
        self.column_names.append(f'XRSB {n}-min Differences %')
        diff_xrsa_list = []
        diff_xrsa_pct_list = []
        for arr in self.xrsa:
            diff_xrsa = arr[n:] - arr[:-n]
            diff_xrsa = np.concatenate([np.full(n, math.nan), diff_xrsa])
            diff_xrsa_list.append(diff_xrsa)
            diff_xrsa_pct = []
            for i, diffy in enumerate(diff_xrsa):
                diff_xrsa_pct.append(diffy/arr[i]*100)
            diff_xrsa_pct_list.append(diff_xrsa_pct)
        self.columns.append(diff_xrsa_list)
        self.column_names.append(f'XRSA {n}-min Differences')
        self.columns.append(diff_xrsa_pct_list)
        self.column_names.append(f'XRSA {n}-min Differences %')
        #also doing temp differences:
        temp_diff_list = []
        for i, arr in enumerate(diff_xrsa_list):
            temp = arr/diff_xrsb_list[i]
            temp_diff_list.append(temp)
        self.columns.append(temp_diff_list)
        self.column_names.append(f'Temp {n}-min Differences')
        
    def save_temp(self):
        temp = self.xrsa/self.xrsb
        self.columns.append(temp)
        self.column_names.append('Temperature (xrsa/xrsb)')
        
    def make_table(self):
        self.t = Table(self.columns, names=self.column_names)
        print(self.t.info)
        self.t.write('GOES_computed_parameters.fits', overwrite=True)

       
def add_c5_10min_bool_and_em():
    ''' adding function to add on the C5 flux for 10 min or longer bool. This is what we will use for the true/false
    in the new ROC curves for deciding if the flare is viable or not. (will change the C5 to C5 10 min or longer)
    '''
    flare_fits = 'GOES_XRS_historical_testerem.fits'
    fitsfile = fits.open(flare_fits)
    data = Table(fitsfile[1].data)[:]
    print(data.columns)
    header = fitsfile[1].header
    xrsb = data['xrsb'][:]
    
    c5_10min_bool_list = []
    c5_thresh = 5e-6
    minutes = 10
    for flare in xrsb:
        above_c5_arr = np.flatnonzero(np.convolve(flare>c5_thresh, np.ones(minutes, dtype=int),'valid')>=minutes)
        if len(above_c5_arr) > 0:
            c5_10min_bool_list.append(True)
        else:
            c5_10min_bool_list.append(False)
    
    #adding column to fits: 
    data.add_column(c5_10min_bool_list, name='above C5 10min', index=9)
    print(data.columns)
    #data.write('GOES_XRS_historical_finalversion.fits', overwrite=True)
    
    em = data['em'][:]
    
    for n in [1,2,3,4,5]:
        diff_em_list = []
        diff_em_pct_list = []
        for arr in em:
            diff_em = arr[n:] - arr[:-n]
            diff_em = np.concatenate([np.full(n, math.nan), diff_em]) #appending the right amount of zeros to front to make the indices correct
            diff_em_list.append(diff_em)
            diff_em_pct = []
            for i, diffy in enumerate(diff_em):
                diff_em_pct.append(diffy/arr[i]*100)
            diff_em_pct_list.append(diff_em_pct)
        data.add_column(diff_em_list, name=f'{n}-min em diff')
        data.add_column(diff_em_pct_list, name=f'{n}-min em diff %')
        print(data.columns)
    data.write('GOES_XRS_historical_finalversion.fits', overwrite=True)
    
# def add_em_differences(n):
#     flare_fits = 'GOES_XRS_historical_testerem.fits'
#     fitsfile = fits.open(flare_fits)
#     data = Table(fitsfile[1].data)[:]
#     print(data.columns)
#     header = fitsfile[1].header
#     em = data['em'][:]
#
#     diff_em_list = []
#     diff_em_pct_list = []
#     for arr in em:
#         diff_em = arr[n:] - arr[:-n]
#         diff_em = np.concatenate([np.full(n, math.nan), diff_em]) #appending the right amount of zeros to front to make the indices correct
#         diff_em_list.append(diff_em)
#         diff_em_pct = []
#         for i, diffy in enumerate(diff_em):
#             diff_em_pct.append(diffy/arr[i]*100)
#         diff_em_pct_list.append(diff_em_pct)
#     data.add_column(diff_em_list, name=f'{n}-min em diff')
#     data.add_column(diff_em_pct_list, name=f'{n}-min em diff %')
#     print(data.columns)
#     data.write('GOES_XRS_historical_finalversion.fits', overwrite=True)
    #print(data[f'{n}-min em diff'][0:10])
    
    

    
    
def testing_out():
    flare_fits = 'GOES_XRS_historical.fits'
    fitsfile = fits.open(flare_fits)
    data = Table(fitsfile[1].data)[:]
    print(data.columns)
    print(data['above C5 10min'][260:270])
        
        
        
if __name__ == '__main__':
    '''Uncomment these for making the new FITS file with calculated parameters
    '''
    # t = MakingParamArrays()
    # t.save_xrsb_rise_above_background()
    # print('increase done')
    # t.save_temp()
    # print('temp done')
    # t.save_differences_between_further_points(1)
    # print('1 min done')
    # t.save_differences_between_further_points(2)
    # print('2 min done')
    # t.save_differences_between_further_points(3)
    # print('3 min done')
    # t.save_differences_between_further_points(4)
    # print('4 min done')
    # t.save_differences_between_further_points(5)
    # print('5 min done')
    # t.make_table()
    # print('fits saved')
    ''' Uncomment this for adding the C5 for 10min or longer bool to the original .fits file
    '''
    add_c5_10min_bool_and_em()
    # add_em_differences(1)
    # add_em_differences(2)
    # add_em_differences(3)
    # add_em_differences(4)
    # add_em_differences(5)

    
    