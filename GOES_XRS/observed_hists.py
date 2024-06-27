#MAKES THE OBSERVATION HISTOGRAM PLOT
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ObservationHists:
    
    C5_LAUNCH_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/xrsb_xrsa_3minem/Launches/5e-06_4.5e-07_1e+47_results.csv'
    
    C8_LAUNCH_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C8/xrsb_xrsa_3minem/Launches/8e-06_6e-07_1e+47_results.csv'
    
    M1_LAUNCH_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/M1/xrsb_xrsa_3minem/Launches/1e-05_1.25e-06_7e+47_results.csv'
    
    def __init__(self):
        self.c5_df = pd.read_csv(self.C5_LAUNCH_FILE)
        self.c8_df = pd.read_csv(self.C8_LAUNCH_FILE)
        self.m1_df = pd.read_csv(self.M1_LAUNCH_FILE)
        
    def plot_single_hist_foxsi(self, ax, df, success_flux, letter):
        #fig, ax = plt.subplots(1,1,figsize=(8,6))
        logbins=np.logspace(np.log10(5e-7),np.log10(5e-3), 40)
        ax.axvline(5e-6, c='k', lw=2, label='C5')
        ax.axvline(9.67e-6, c='r', lw=2, label='Max FOXSI Flux Observed in Campaign')
        ax.legend()
        ax.set_xscale('log')
        ax.set_xlabel('GOES Flux [W/m$^2$]', fontsize=14)
        ax.set_ylabel('Number of Flares', fontsize=14)
        ax.hist(df['Max_FOXSI'], bins=logbins, range=(1e-8, 1e-2))
        ax.set_title(f'Maximum Flux Observed by FOXSI \n Success Flux = {success_flux}', fontsize=14)
        ax.text(0.05, 0.9, letter, fontsize=24, transform=ax.transAxes, va='center')
        #plt.show()
        
    def plot_single_hist_hic(self, ax, df, success_flux, letter):
        #fig, ax = plt.subplots(1,1,figsize=(8,6))
        logbins=np.logspace(np.log10(5e-7),np.log10(5e-3), 40)
        ax.axvline(5e-6, c='k', lw=2, label='C5')
        ax.axvline(8.23e-6, c='r', lw=2, label='Mean HiC Flux Observed in Campaign')
        ax.legend()
        ax.set_xscale('log')
        ax.set_xlabel('GOES Flux [W/m$^2$]', fontsize=14)
        ax.set_ylabel('Number of Flares', fontsize=14)
        ax.hist(df['Mean_HiC'], bins=logbins, range=(1e-8, 1e-2))
        ax.set_title(f'Mean Flux Observed by HiC \n Success Flux = {success_flux}', fontsize=14)
        ax.text(0.05, 0.9, letter, fontsize=24, transform=ax.transAxes, va='center')
        
    def do_all_plots(self):
        fig, axs = plt.subplots(3, 2, figsize=(26, 26))
        c5_foxsi = self.plot_single_hist_foxsi(axs[0, 0], self.c5_df, 'C5', 'A.')
        c5_hic = self.plot_single_hist_foxsi(axs[0, 1], self.c5_df, 'C5', 'B.')
        c8_foxsi = self.plot_single_hist_foxsi(axs[1, 0], self.c8_df, 'C8', 'C.')
        c8_hic = self.plot_single_hist_foxsi(axs[1, 1], self.c8_df, 'C8', 'D.')
        m1_foxsi = self.plot_single_hist_foxsi(axs[2, 0], self.m1_df, 'M1', 'E.')
        m1_hic = self.plot_single_hist_foxsi(axs[2, 1], self.m1_df, 'M1', 'F.')
        plt.savefig('obs_hists.png', dpi=250, bbox_inches='tight')
        
    
if __name__ == '__main__':
    tester = ObservationHists()
    tester.do_all_plots()

        
