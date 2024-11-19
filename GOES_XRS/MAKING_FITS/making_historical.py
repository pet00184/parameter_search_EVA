import numpy as np
import pandas as pd
import netCDF4 as nc
from astropy.time import Time
import astropy.units as u
from astropy.table import QTable, Table
from astropy.io import fits
from sunpy import timeseries as ts
from sunkit_instruments import goes_xrs

flare_summary_file = 'sci_xrsf-l2-flsum_g16_s20170209_e20241115_v2-2-0.nc'
xrs_file = 'sci_xrsf-l2-avg1m_g16_s20170207_e20241115_v2-2-0.nc'

class MakingHistoricalDataframe:
    
    midnight_jan1 = 946684800.0
    noon_jan1 = midnight_jan1 + 43200
    
    xrsa_list = []
    xrsb_list = []
    time_list = []
    UTC_time_list = []
    peak_time_list = []
    UTC_peak_time_list = []
    peak_flux_list = []
    flare_class_list = []
    stp_time_list = []
    c5_list = []
    background_list = []
    
    
    def __init__(self, flare_summary_file, xrs_file):
        self.xrs_file = xrs_file
        self.xrs_data = nc.Dataset(xrs_file)
        self.fs_data = nc.Dataset(flare_summary_file)
        
    def make_flare_tuple(self):
        ''' Saves the start and end index for each flare'''
        flare_ids = self.fs_data['flare_id'][:]
        self.flare_id_arr = sorted(set(flare_ids))
        self.flare_tuple = []
        for f in self.flare_id_arr:
            this_flare = np.where(f==flare_ids)[0]
            self.flare_tuple.append([this_flare[0], this_flare[-1]])
            
    def parse_xrs_data(self):
        ''' Parses out the GOES XRSA, XRSB and time lightcurves for each flare.'''
        for i, index_range in enumerate(self.flare_tuple):
            start = index_range[0]
            end = index_range[1]
            xrs_start = np.where(self.xrs_data['time']==self.fs_data['time'][start])[0][0]
            xrs_end = np.where(self.xrs_data['time']==self.fs_data['time'][end])[0][0]
            self.xrsb_list.append(np.array(self.xrs_data['xrsb_flux_observed'][xrs_start-15:xrs_end+15].data))
            self.xrsa_list.append(np.array(self.xrs_data['xrsa_flux_observed'][xrs_start-15:xrs_end+15].data))
            self.time_list.append(np.array(self.xrs_data['time'][xrs_start-15:xrs_end+15].data))
            
    def include_flare_class(self):
        for i, index_range in enumerate(self.flare_tuple):
            peak = index_range[0]+1
            self.flare_class_list.append(self.fs_data['flare_class'][peak])
            
    def include_background_flux(self):
        for i, index_range in enumerate(self.flare_tuple):
            bkgrnd = index_range[0]
            self.background_list.append(self.fs_data['background_flux'][bkgrnd].data)
            
    def include_peak_flux(self):
        for i, index_range in enumerate(self.flare_tuple):
            peak = index_range[0]+1
            self.peak_flux_list.append(np.array(self.fs_data['xrsb_flux'][peak].data))
            
    def include_peak_time(self):
        for i, index_range in enumerate(self.flare_tuple):
            peak = index_range[0]+1
            self.peak_time_list.append(np.array(self.fs_data['time'][peak].data))
            self.UTC_peak_time_list.append(Time(self.fs_data['time'][peak] + self.noon_jan1, format='unix').isot)
            
    def include_start_to_peak_time(self):
        for i, index_range in enumerate(self.flare_tuple):
            start = index_range[0]
            peak = index_range[0]+1
            diff = (self.fs_data['time'][peak] - self.fs_data['time'][start]) 
            self.stp_time_list.append(diff/60.)
            
    def include_c5_boolean(self):
        for i, index_range in enumerate(self.flare_tuple):
            if self.peak_flux_list[i] > 5e-6:
                self.c5_list.append(True)
            else: 
                self.c5_list.append(False)
        
       
    def try_plain_table(self):
        self.t = Table([self.xrsa_list, self.xrsb_list, self.time_list, self.flare_class_list, self.peak_flux_list, self.peak_time_list, self.UTC_peak_time_list, self.stp_time_list, self.c5_list, self.flare_id_arr, self.background_list],
                        names=('xrsa', 'xrsb', 'time', 'class', 'peak flux', 'peak time', 'UTC peak time', 'start to peak time', 'above C5', 'flare ID', 'background flux'))
        print(self.t.info)
        self.t.write('GOES_XRS_historical.fits', overwrite=True)
        
        

if __name__ == '__main__':
    test = MakingHistoricalDataframe(flare_summary_file, xrs_file)
    test.make_flare_tuple()
    print('flare tuple done')
    test.parse_xrs_data()
    print('parsing xrs data done')
    test.include_flare_class()
    print('including flare class done')
    test.include_peak_flux()
    print('including peak flux done')
    test.include_peak_time()
    print('including peak time done')
    test.include_start_to_peak_time()
    print('including stp done')
    test.include_c5_boolean()
    print('flare boolean done. trying to save fits now.')
    test.include_background_flux()
    print('including background flux done')

    test.try_plain_table()
    print('done!!')

        