#DO MAX FLARES ALL SHOWN HERE
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

class MaxFlares:
    
    C5_LAUNCH_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/xrsb_xrsa_3minem/Launches/5e-06_4.5e-07_1e+47_results.csv'
    
    C8_LAUNCH_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C8/xrsb_xrsa_3minem/Launches/8e-06_6e-07_1e+47_results.csv'
    
    M1_LAUNCH_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/M1/xrsb_xrsa_3minem/Launches/1e-05_1.25e-06_7e+47_results.csv'
    
    FLARE_FITS = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/GOES_XRS_historical.fits'
    def __init__(self):
        fitsfile = fits.open(self.FLARE_FITS)
        self.all_flare_data = fitsfile[1].data
        self.c5_df = pd.read_csv(self.C5_LAUNCH_FILE)
        self.c8_df = pd.read_csv(self.C8_LAUNCH_FILE)
        self.m1_df = pd.read_csv(self.M1_LAUNCH_FILE)
        
    def make_dfs(self, launch_combo_df):
        launch_df = launch_combo_df[launch_combo_df['Cancelled?']==False].reset_index()
        cancelled_df = launch_combo_df[launch_combo_df['Cancelled?']==True].reset_index()
        return launch_df, cancelled_df
        
    def save_launch_canc_dfs(self):
        self.c5_launch, self.c5_cancel = self.make_dfs(self.c5_df)
        self.c8_launch, self.c8_cancel = self.make_dfs(self.c8_df)
        self.m1_launch, self.m1_cancel = self.make_dfs(self.m1_df)
        
    def plot_hist(self, ax, launch_df, cancelled_df, letter):
        ''' Here is a histogram of all launches/cancellations, with all the potential flares plotted in the background.
        '''
        logbins=np.logspace(np.log10(1e-8),np.log10(5e-3), 50)
        #fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.hist(self.all_flare_data['peak flux'], bins=logbins, range=(1e-8, 1e-2), color='r', alpha=0.2, label='All Flares')
        ax.hist(cancelled_df['Flare_Max_Flux'], bins=logbins, range=(1e-8, 1e-2), color='b', label='Cancelled Flares', alpha=0.5)
        ax.hist(launch_df['Flare_Max_Flux'], bins=logbins, range=(1e-8, 1e-2), color='k', label='Launched Flares', alpha=0.7)
        ax.axvline(5e-6, c='k', lw=2, ls='--', label='C5')
        ax.axvline(1.6e-5, c='g', lw=2, ls='--', label='Flare Class from Campaign')
        ax.set_xscale('log')
        ax.set_xlabel('GOES Flux [W/m$^2$]')
        ax.set_ylabel('# of Flares')
        ax.legend()
        ax.set_title(f'Maximum Flare Flux')
        ax.text(0.05, 0.9, letter, fontsize=24, transform=ax.transAxes, va='center')
        #plt.savefig(os.path.join(self.out_dir, self.plot_folder, self.savestring, 'maxflux_allflaresshown_histogram.png'), bbox_inches='tight', dpi=250)
        #plt.show()
        
    def make_all_plots(self):
        fig, axs = plt.subplots(3, 1, figsize=(8, 20))
        c5 = self.plot_hist(axs[0], self.c5_launch, self.c5_cancel, 'A.')
        c8 = self.plot_hist(axs[1], self.c8_launch, self.c8_cancel, 'B.')
        m1 = self.plot_hist(axs[2], self.m1_launch, self.m1_cancel, 'C.')
        plt.savefig('max_flares.png', dpi=250, bbox_inches='tight')
        
if __name__ == '__main__':
    tester = MaxFlares()
    tester.save_launch_canc_dfs()
    tester.make_all_plots()