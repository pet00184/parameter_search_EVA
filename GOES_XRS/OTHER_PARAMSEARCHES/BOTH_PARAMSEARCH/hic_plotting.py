import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
import os
from astropy.io import fits
from datetime import datetime


class PlottingResults:
    ''' Plots that utilize and compare all of the prameter scores. These will mainly be used to get an overview of which
    parameters are worth looking into more in depth.
    '''
    
    def __init__(self, keys_list, nice_keys_list, success_flux_key, out_dir, score_csv, plot_folder):
        self.score_df = pd.read_csv(os.path.join(out_dir, score_csv))
        self.out_dir = out_dir
        self.keys_list = keys_list
        self.nice_keys_list = nice_keys_list
        self.success_flux_key = success_flux_key
        self.plot_folder = plot_folder
        
        os.makedirs(os.path.join(self.out_dir, self.plot_folder), exist_ok=True)
        

    def plot_specific_cf(self, param_dict, savestring, big_paramset=False):
        ''' Using a dictionary of parameters and keys (must be what was used for the chosen run), a CF matrix is plotted
        for that combination. 
        '''
        cf_df = self.score_df
        for key, val in param_dict.items():
            cf_df = cf_df[cf_df[key]==val].reset_index(drop=True)
        if cf_df.shape[0] > 1: raise ValueError('More than one combination left! Double check you have all parameters.')
        confusion_matrix = np.array([[cf_df.loc[0, 'TN'], cf_df.loc[0, 'TN_canc'], 
                        cf_df.loc[0, f'FP_no{self.success_flux_key}_short'], cf_df.loc[0, f'FP_no{self.success_flux_key}_long']],
                        [cf_df.loc[0, 'FN'], cf_df.loc[0, 'FN_canc'], 
                        cf_df.loc[0, f'FP_{self.success_flux_key}_short'], cf_df.loc[0, 'TP']]])
        fig, ax = plt.subplots()
        ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.7)
        for k in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                ax.text(x=j, y=k,s=confusion_matrix[k, j], va='center', ha='center', size='xx-large')
        ylabels = ['', 'False', 'True']
        ax.set_ylabel(f'Long Duration Flare above {self.success_flux_key}?', fontsize=14)
        ax.set_yticklabels(ylabels, rotation=90, fontsize=12)
        ax.xaxis.set_ticks_position("bottom")
        xlabels = ['', 'No Trigger', 'Cancelled Launch','Launch \n Failed Obs', 'Launch \n Successful Obs']
        ax.set_xlabel('Launch Results', fontsize=14)
        ax.set_xticklabels(xlabels, fontsize=12)
        plt.title(f"Precision = {cf_df.loc[0, 'Precision']:.2f}, Recall = {cf_df.loc[0, 'Recall']:.2f}, Launch/Trigger Ratio = {cf_df.loc[0, 'LaunchTriggerRatio']:.2f}, \n Gordon = {cf_df.loc[0, 'Gordon']:.2f}, HiC Obs post-peak = {cf_df.loc[0, 'ObsAfterPeak']:.2f}")
        #lastly, need to make the stuff for showing the values!
        if not big_paramset:
            param_units = [cf_df.loc[0, f'{key}_units'] for key in param_dict.keys()]
            param_list = [f'{nice_key}={val} {unit}' for (key, val), unit, nice_key in zip(param_dict.items(), param_units, self.nice_keys_list)]
            plt.text(1.01, 0.8, ("Parameters \n" + "\n".join(param_list)), fontsize=12, transform=ax.transAxes)
        savestring = [f'{key}{val}' for key, val in param_dict.items()]
        savestring = "_".join(savestring)
        plt.savefig(os.path.join(self.out_dir, self.plot_folder, savestring, 'cf.png'), bbox_inches='tight', dpi=250)
        plt.close()
    
class LaunchPlotting:
    ''' Class for plotting specific launch/cancellation histograms! This will take in a dictionary, similar to how the cf
    is plotted. One challenge is opening the correct file! maybe doing the savestring naming convention...? It probably wouldn't
    work if the file name gets too long....
    '''

    def __init__(self, combo_dict, nice_keys_list, flare_fits, out_dir, savestring, plot_folder):
        self.combo_dict = combo_dict
        self.nice_keys_list = nice_keys_list
        self.launch_combo_list = os.listdir(os.path.join(out_dir, 'Launches'))
        fitsfile = fits.open(flare_fits)
        self.all_flare_data = fitsfile[1].data
        self.savestring = savestring
        self.plot_folder = plot_folder
        self.out_dir = out_dir

    def find_correct_launch_file(self):
        param_combo_string = "_".join([str(val) for val in self.combo_dict.values()])
        launch_csv_str = "_".join([param_combo_string, 'results.csv'])
        self.launch_combo_df = pd.read_csv(os.path.join(self.out_dir, 'Launches', launch_csv_str))
        
    def save_launch_cancellation_dfs(self):
        self.launch_df = self.launch_combo_df[self.launch_combo_df['Cancelled?']==False].reset_index()
        self.cancelled_df = self.launch_combo_df[self.launch_combo_df['Cancelled?']==True].reset_index()
        
    def plot_observation_histograms(self, hic=False, cancellation=False):
        ''' Plots a histogram of the mean flux observed by HiC
        '''
        if cancellation:
            df = self.cancelled_df
        else:
            df = self.launch_df
        param_units = [df.loc[0, f'{key}_units'] for key in self.combo_dict.keys()]
        param_list = [f'{nice_key}={val} {unit}' for (key, val), unit, nice_key in zip(self.combo_dict.items(), param_units, self.nice_keys_list)]
        logbins=np.logspace(np.log10(5e-7),np.log10(5e-3), 40)
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.axvline(5e-6, c='k', lw=2, label='C5')
        ax.legend()
        ax.set_xscale('log')
        ax.set_xlabel('GOES Flux W/m$^2$', fontsize=14)
        ax.set_ylabel('Number of Flares', fontsize=14)
        #ax.set_xlim(1e-6, 2e-3)
        plt.text(1.01, 0.8, ("Parameters: \n" + "\n".join(param_list)), fontsize=12, transform=ax.transAxes)
        ax.hist(df['Mean_HiC'], bins=logbins, range=(1e-8, 1e-2))
        if cancellation:
            ax.set_title(f'Mean Flux Missed by HiC due to Cancellation')
            plt.savefig(os.path.join(self.out_dir, self.plot_folder, self.savestring, 'meanhic_cancellation_histogram.png'), bbox_inches='tight', dpi=250)
        else:
            ax.set_title(f'Mean Observed Flux for HiC', fontsize=16)
            plt.savefig(os.path.join(self.out_dir, self.plot_folder, self.savestring, 'meanhic_launch_histogram.png'), bbox_inches='tight', dpi=250)

def make_combination_plots(cf_dict, keys_list, nice_keys_list, flux_key, flare_fits, out_dir, score_csv, plot_folder):
    savestring = [f'{key}{val}' for key, val in cf_dict.items()]
    savestring = '_'.join(savestring)
    os.makedirs(os.path.join(out_dir, plot_folder, savestring), exist_ok=True)
    plotter =  PlottingResults(keys_list, nice_keys_list, flux_key, out_dir, score_csv, plot_folder)
    plotter.plot_specific_cf(cf_dict, savestring)
    launch_plotter = LaunchPlotting(cf_dict, nice_keys_list, flare_fits, out_dir, savestring, plot_folder)
    launch_plotter.find_correct_launch_file()
    launch_plotter.save_launch_cancellation_dfs()
    #making histogram plots:
    launch_plotter.plot_observation_histograms()
    launch_plotter.plot_observation_histograms(cancellation=True)
    
    
    
        
if __name__ == '__main__':
    keys_list = ['xrsa', '5minem']
    nice_keys_list = ['XRSA', 'dEmission Measure (5 min)']
    flux_key = 'C5'
    out_dir = os.path.join('RESULTS', flux_key, 'xrsa_5minem')
    score_csv = 'AllParameterScores.csv'
    flare_fits = '../GOES_XRS_historical.fits'
    
    cf_dict = {
        #'5minem': 5e47,
        #'5mintemp': 10.0 ,#if we want a 0, we might need to check out how that saves in launch files...
        'xrsa': 5.5e-7,
        '5minem': 5e47
        #'xrsb': 4e-6,
    } #needs to be in the same order as everything!!!

    #make_combination_plots(cf_dict, nice_keys_list, flare_fits, out_dir, score_csv)
    savestring = [f'{key}{val}' for key, val in cf_dict.items()]
    savestring = '_'.join(savestring)
    os.makedirs(os.path.join(out_dir, self.plot_folder, savestring), exist_ok=True)
    launch_plotter = LaunchPlotting(cf_dict, nice_keys_list, flare_fits, out_dir, savestring)
    launch_plotter.find_correct_launch_file()
    launch_plotter.save_launch_cancellation_dfs()
    launch_plotter.plot_longdurationobs_hist()    
        