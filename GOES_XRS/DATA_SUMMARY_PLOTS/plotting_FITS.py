import pandas as pd
import numpy as np
from astropy.io import fits
from matplotlib import pyplot as plt
from scipy import stats as st
import os


flare_fits = '../GOES_XRS_historical.fits'

class FITS_plots:
    
    def __init__(self, flare_fits):
        fitsfile = fits.open(flare_fits)
        self.data = fitsfile[1].data
        self.header = fitsfile[1].header
        
    def plot_one(self, i):
        xrsa = self.data['xrsa'][i]
        xrsb = self.data['xrsb'][i]
        time = self.data['time'][i]
        x_axis = np.arange(-15, len(xrsa)-15)
        
        fig, ax = plt.subplots(1,1,figsize=(8,4))
        ax.plot(x_axis[15:-15], xrsb[15:-15], c='r')
        ax.plot(x_axis[15:-15], xrsa[15:-15], c='b')
        ax.set_yscale('log')
        ax.set_ylabel('GOES Flux (W/m^2)')
        ax.set_xlabel('Duration (Minutes)')
        ax.set_title(f'{self.data["UTC peak time"][i][:16]}, Class={self.data["class"][i]}')
       # plt.savefig('example_lc.png', bbox_inches='tight', dpi=200)
        #plt.show()
        return plt
        
    def peak_flux_histogram(self):
        peak_flux = self.data['peak flux']
        above_c5 = len(np.where(self.data['above c5']==True)[0])
        print(above_c5)
        logbins=np.logspace(np.log10(1e-8),np.log10(1e-2), 50)
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.hist(self.data['peak flux'], bins=logbins, range=(1e-8, 1e-2))
        ax.axvline(5e-6, c='r', lw=2, label='C5')
        ax.set_xscale('log')
        ax.set_xlabel('GOES Flux W/m$^2$')
        ax.set_ylabel('# of Flares')
        ax.set_title(f'Peak Flux \n Abolve C5={above_c5}, Below={len(peak_flux) - above_c5}')
        plt.savefig('peak_flux_histogram.png', bbox_inches='tight', dpi=200)
        ax.legend()
        plt.show()
        
    def c5_10min_histogram(self):
        above_c5 = len(np.where(self.data['above c5']==True)[0])
        above_c5_10min = len(np.where(self.data['above c5 10min']==True)[0])
        above_c5_10min_data = self.data[self.data['above c5 10min']==True]
        logbins=np.logspace(np.log10(1e-8),np.log10(1e-2), 50)
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.hist(above_c5_10min_data['peak flux'], bins=logbins, range=(1e-8, 1e-2))
        ax.axvline(5e-6, c='r', lw=2, label='C5')
        ax.axvline(1e-5, c='k', lw=2, label='M1')
        ax.set_xscale('log')
        ax.set_xlabel('GOES Flux W/m$^2$')
        ax.set_ylabel('# of Flares')
        ax.set_title(f'Peak Flux of Flares above C5 for 10 Minutes')
        ax.legend()
        plt.savefig('c5_10min_peak_flux_histogram.png', bbox_inches='tight', dpi=200)
        plt.show()
        
    def time_to_peak_histogram(self):
        ttp = self.data['start to peak time']
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.hist(ttp, bins=20, range=(0, 40))
        ax.set_xlabel('Start to Peak Time (minutes)')
        ax.set_ylabel('# of Flares')
        ax.set_title(f'Start to Peak \n Mean={np.mean(ttp):.0f} minutes, Mode={st.mode(ttp)[0]} minutes')
        plt.savefig('stp_all.png', bbox_inches='tight', dpi=200)
        plt.show()
        
    def ttp_histogram_c5(self):
        c5 = np.where(self.data['above c5']==True)[0]
        ttp = self.data['start to peak time'][c5]
        print(np.min(ttp), np.max(ttp))
        print(np.mean(ttp))
        print(st.mode(ttp))
        print(np.median(ttp))
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.hist(ttp, bins=30, range=(0, 60))
        ax.axvline(np.mean(ttp), c='r', lw=2, label='Mean')
        ax.axvline(np.median(ttp), c='k', lw=2, label='Median')
        ax.set_xlabel('Start to Peak Time (minutes)')
        ax.set_ylabel('# of Flares')
        ax.set_title(f'Start to Peak for >C5 flares \n Mean={np.mean(ttp):.0f} min, Mode={st.mode(ttp)[0]} min, Median={np.median(ttp):.0f} min')
        ax.legend()
        plt.savefig('stp_c5_10min.png', bbox_inches='tight', dpi=200)
        plt.show()
        
    def year_hists(self):
        self.year_list = ['2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
        self.flare_separations = []
        for year in self.year_list:
            flares = np.flatnonzero(np.core.defchararray.find(self.data['UTC peak time'], year)!=-1)
            self.flare_separations.append(flares)
        self.abovec5list = []
        self.belowc5list = []
        for i, year in enumerate(self.flare_separations):
            peak_flux = self.data['peak flux'][year]
            above_c5 = len(np.where(self.data['above c5'][year]==True)[0])
            self.abovec5list.append(above_c5)
            self.belowc5list.append(len(peak_flux) - above_c5)

            #logbins = [1e-8, 2.5e-8, 5e-8, 7.5e-8, 1e-7, 2.5e-7, 5e-7, 7.5e-7, 1e-6, 2.5e-6, 5e-6, 7.5e-6, 1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3]
            logbins = np.logspace(-8, -2)
            print(logbins)
            fig, ax = plt.subplots(1,1,figsize=(8,6))
            ax.hist(peak_flux, bins=logbins, range=(1e-8, 1e-2))
            ax.axvline(5e-6, c='r', lw=2)
            ax.set_xscale('log')
            ax.set_xlabel('GOES Flux W/m$^2$')
            ax.set_ylabel('# of Flares')
            ax.set_title(f'GOES Classes {self.year_list[i]} \n Abolve C5={above_c5}, Below={len(peak_flux) - above_c5}')
            plt.savefig(f'peakflux_{self.year_list[i]}.png', bbox_inches='tight', dpi=200)
            
    def year_barplot(self):
        year_counts = {
            'Below c5': self.belowc5list,
            'Above c5': self.abovec5list
        }
        width = 0.6
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        bottom = np.zeros(8)
        for classs, classcount in year_counts.items():
            p = ax.bar(self.year_list, classcount, width, label=classs, bottom=bottom)
            bottom += classcount
            ax.bar_label(p, label_type='center')
            
        ax.set_title('GOES Class')
        ax.legend()
        plt.savefig('GOESbyyear_barplot.png', bbox_inches='tight', dpi=200)
        plt.show()
        
    def duration_histogram(self):
        abovemask = np.where(self.data['above c5']==True)[0]
        above_c5 = self.data['xrsa'][abovemask]
        belowmask = np.where(self.data['above c5']==False)[0]
        below_c5 = self.data['xrsa'][belowmask]
     
        above_c5_lengths = [len(above_c5[i])-30 for i, data in enumerate(above_c5)]
        both_length = len(np.where(np.array(above_c5_lengths) > 60)[0])
        below_c5_lengths = [len(below_c5[i])-30 for i, data in enumerate(below_c5)]
        total_lengths = [len(self.data['xrsa'][i])-30 for i, data in enumerate(self.data['xrsa'])]
        duration_above_60 = len(np.where(np.array(total_lengths) > 60)[0])
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        ax.hist([below_c5_lengths, above_c5_lengths], bins=25, range=(0, 250), stacked=True, color=['b', 'r'], label=['Below C5', 'Above C5'])
        #ax.hist(above_c5_lengths, range=(0, 250), color='r', stacked=True)
        ax.axvline(60, c='k', lw=2)
        ax.set_title(f'Flare Duration \n Over 60 Min. = {duration_above_60}, Above C5 = {len(above_c5)}, Both = {both_length}')
        ax.set_ylabel('Number of Flares')
        ax.set_xlabel('Minutes')
        ax.legend()
        plt.savefig('duration_histogram.png', dpi=250)
        plt.show()
        
        self.long_ones = np.where(np.array(total_lengths) > 200)[0]
        #plt.show()
        
    def long_plots(self):
        os.makedirs('Long_Duration_Plots', exist_ok=True)
        for i in self.long_ones:
            plt = self.plot_one(i)
            plt.savefig(f'Long_Duration_Plots/plot_{i}.png')
            
        
        
        
        
if __name__ == '__main__':
    test = FITS_plots(flare_fits)
    #test.plot_one(2)
    # test.peak_flux_histogram()
    # test.time_to_peak_histogram()
    # test.ttp_histogram_c5()
    test.year_hists()
    test.year_barplot()
    #test.c5_10min_histogram()
    #test.duration_histogram()