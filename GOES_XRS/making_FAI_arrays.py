''''
This is specifically making FAI parameters to compare to Hughs work!!
'''

import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table, Column
from matplotlib import pyplot as plt
from scipy import stats as st
import math
import temp_em as tem

class MakingFAIArrays:
    ''' We need a few columns: 1, 3, 5 minute differences and then the temp and ems for those differences! (6 columns total)
    '''
        
    flare_fits = 'GOES_XRS_historical.fits'
    columns = []
    column_names = []
    
    def __init__(self):
        fitsfile = fits.open(self.flare_fits)
        self.data = Table(fitsfile[1].data)[:]
        self.header = fitsfile[1].header
        self.xrsb = self.data['xrsb'][:]
        self.xrsa = self.data['xrsa'][:]
        
        
    def save_differences_between_further_points(self, n):
        ''' Saves the n-minute difference between XRSA and XRSB points, including the correct amount of zeros at the
        beginning of each array to account for the correct timting.
        '''
        diff_xrsb_list = []
        for arr in self.xrsb:
            diff_xrsb = arr[n:] - arr[:-n]
            diff_xrsb = np.concatenate([np.full(n, math.nan), diff_xrsb]) #appending the right amount of zeros to front to make the indices correct
            diff_xrsb_list.append(diff_xrsb)
        self.columns.append(diff_xrsb_list)
        self.column_names.append(f'XRSB {n}-min Differences')
        diff_xrsa_list = []
        for arr in self.xrsa:
            diff_xrsa = arr[n:] - arr[:-n]
            diff_xrsa = np.concatenate([np.full(n, math.nan), diff_xrsa])
            diff_xrsa_list.append(diff_xrsa)
        self.columns.append(diff_xrsa_list)
        self.column_names.append(f'XRSA {n}-min Differences')
        
    def make_diff_table(self):
       self.t = Table(self.columns, names=self.column_names)
       print(self.t.info)
        
    def save_temp_em_fromdiffs(self, n):
        ''' This will save a temp and emission measure column for a previously saved column of differences. SAVE DF FIRST!
        '''
        tem.download_latest_goes_response()
        temp_list = []
        em_list = []
        for i, arr in enumerate(self.t[f'XRSB {n}-min Differences']):
            temp_arr, em_arr = tem.get_tem(arr, self.t[f'XRSA {n}-min Differences'][i])
            temp_list.append(temp_arr)
            em_list.append(em_arr)
        self.t[f'Temp {n}-min Differences'] = temp_list
        self.t[f'EM {n}-min Differences'] = em_list
        
    def write_fits(self):
        self.t.write('GOES_computed_FAI.fits', overwrite=True)


if __name__ == '__main__':           
    t = MakingFAIArrays()
    t.save_differences_between_further_points(1)
    print('1 min done')
    t.save_differences_between_further_points(3)
    print('3 min done')
    t.save_differences_between_further_points(5)
    print('5 min done')
    t.make_diff_table()
    print('diff table made')
    t.save_temp_em_fromdiffs(1)
    print('1 min temp em done')
    t.save_temp_em_fromdiffs(3)
    print('3 min temp em done')
    t.save_temp_em_fromdiffs(5)
    print('5 min temp em done')
    t.write_fits()
    


        
    