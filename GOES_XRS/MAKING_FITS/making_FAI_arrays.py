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
import emission_measure as em

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
        
    def save_xrsb_rise_above_background(self):
        bk = self.data['background flux'][:]
        self.increase = self.xrsb - bk
        self.increase_pct = (self.xrsb-bk)/bk
        # self.columns.append(self.increase)
        # self.columns.append(self.increase_pct)
        # self.column_names.append('XRSB Increase above Background')
        # self.column_names.append('XRSB Increase above Background Fraction')
        self.data['XRSB Increase above Background'] = self.increase
        self.data['XRSB Increase above Background Fraction'] = self.increase_pct
        
    def save_temp_em(self):
        ''' Calculates the temperature and emission measure for each flare.
        '''
        em_list = []
        temp_list = []
        for i, arr in enumerate(self.xrsb):
            em_arr, temp_arr = em.compute_goes_emission_measure(arr, self.xrsa[i], 16)
            em_list.append(em_arr)
            temp_list.append(temp_arr)
        self.data['Temperature'] = temp_list
        self.data['Emission Measure'] = em_list
        
    def save_differences_between_further_points(self, n):
        ''' Saves the n-minute difference between XRSA and XRSB points, including the correct amount of zeros at the
        beginning of each array to account for the correct timting.
        '''
        diff_xrsb_list = []
        for arr in self.xrsb:
            diff_xrsb = arr[n:] - arr[:-n]
            diff_xrsb = np.concatenate([np.full(n, math.nan), diff_xrsb]) #appending the right amount of zeros to front to make the indices correct
            diff_xrsb_list.append(diff_xrsb)
        # self.columns.append(diff_xrsb_list)
        # self.column_names.append(f'XRSB {n}-min Differences')
        self.data[f'XRSB {n}-min Differences'] = diff_xrsb_list
        diff_xrsa_list = []
        for arr in self.xrsa:
            diff_xrsa = arr[n:] - arr[:-n]
            diff_xrsa = np.concatenate([np.full(n, math.nan), diff_xrsa])
            diff_xrsa_list.append(diff_xrsa)
        # self.columns.append(diff_xrsa_list)
        # self.column_names.append(f'XRSA {n}-min Differences')
        self.data[f'XRSA {n}-min Differences'] = diff_xrsa_list
        
    def save_tem_differences_between_further_points(self,n):
        '''Saves the actual temperature and emission measure differences (as opposed to the differences from XRS flux
        increases that were being used for the previous searches.)'''
        diff_temp_list = []
        diff_em_list = []
        for arr in self.data['Temperature']:
            diff_temp = arr[n:] - arr[:-n]
            diff_temp = np.concatenate([np.full(n, math.nan), diff_temp]) #appending the right amount of zeros to front to make the indices correct
            diff_temp_list.append(diff_temp)
        self.data[f'{n}-minute Temperature Difference'] = diff_temp_list
        for arr in self.data['Emission Measure']:
            diff_em = arr[n:] - arr[:-n]
            diff_em = np.concatenate([np.full(n, math.nan), diff_em]) #appending the right amount of zeros to front to make the indices correct
            diff_em_list.append(diff_em)
        self.data[f'{n}-minute Emission Measure Difference'] = diff_em_list
        
        
    # def make_diff_table(self):
    #    self.t = Table(self.columns, names=self.column_names)
    #    print(self.t.info)
        
    def save_temp_em_fromdiffs(self, n):
        ''' This will save a temp and emission measure column for a previously saved column of differences.
        Since these are calculated from XRS n-minute differences, there may be many NAN values. This is because the 
        temperature and emission measure are calcualted from an XRSA/XRSB ratio. Therefore, it doesn't make sense if one/both
        of the values are negative and a NAN is returned instead.
        '''
        temp_list = []
        em_list = []
        for i, arr in enumerate(self.data[f'XRSB {n}-min Differences']):
            em_arr, temp_arr = em.compute_goes_emission_measure(arr, self.data[f'XRSA {n}-min Differences'][i], 16)
            temp_list.append(temp_arr)
            em_list.append(em_arr)
        self.data[f'Temp (XRS {n}-min Differences)'] = temp_list
        self.data[f'EM (XRS {n}-min Differences)'] = em_list
        
    def write_fits(self):
        self.data.write('GOES_XRS_computed_params.fits', overwrite=True)


if __name__ == '__main__':           
    t = MakingFAIArrays()
    t.save_xrsb_rise_above_background()
    print('saving rise above background done')
    t.save_temp_em()
    print('saving temperature and emission measure done')
    t.save_differences_between_further_points(1)
    print('1 min done')
    t.save_differences_between_further_points(3)
    print('3 min done')
    t.save_differences_between_further_points(5)
    print('5 min done')
    # t.make_diff_table()
    # print('diff table made')
    t.save_temp_em_fromdiffs(1)
    print('1 min temp em done')
    t.save_temp_em_fromdiffs(3)
    print('3 min temp em done')
    t.save_temp_em_fromdiffs(5)
    print('5 min temp em done')
    t.save_tem_differences_between_further_points(1)
    print('1 min temp em diffs done')
    t.save_tem_differences_between_further_points(3)
    print('3 min temp em diffs done')
    t.save_tem_differences_between_further_points(5)
    print('5 min temp em diffs done')
    t.write_fits()
    print('done!')
    


        
    