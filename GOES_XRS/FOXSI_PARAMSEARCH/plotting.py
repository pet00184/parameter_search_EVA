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
    
 def make_full_pr_plot(self, main_key):
     ''' Precision-Recall plot of all combinations tested during the run. 
     color_key = key used to color code the data, so that it is more easily readable. (In most cases, this will be xrsb)
     '''
     nice_main_key = self.nice_keys_list[np.where(np.array(self.keys_list) == main_key)[0][0]]
     main_key_values = self.score_df[main_key].unique()
     unit = self.score_df.loc[0, f'{main_key}_units']
     colors = cm.rainbow(np.linspace(0, 1, len(main_key_values)))
     fig, ax = plt.subplots()
     ax.set_xlabel(r"Recall", fontsize=14)
     ax.set_ylabel(r'Precision', fontsize=12)
     ax.set_title(f'Precision-Recall Curve \n Parameters: {", ".join(self.nice_keys_list)}', fontsize=13) 
     ax.set_xlim(0, 1)
     ax.set_ylim(0, 1)
     for i, value in enumerate(main_key_values):
         main_key_df = self.score_df[self.score_df[main_key]==value].reset_index()
         ax.scatter(main_key_df['Recall'], main_key_df['Precision'], marker='o', c=[colors[i]]*main_key_df.shape[0], label = f'{nice_main_key}={value:.1e} {unit}')
     plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=7, title=f'Sorted by: {nice_main_key}')
     plt.savefig(os.path.join(self.out_dir, self.plot_folder, f'fullprplot_{main_key}coded.png'), bbox_inches='tight', dpi=250)
    
 def make_optimal_pr_plot(self):
     ''' Only plots the highest precision scores for every 0.25 increase in recall.
     '''
     optimal_df = pd.DataFrame()
     recall_bins = np.arange(0, 1.025, .025)
     for i in range(len(recall_bins)-1):
         recall_binned_df = self.score_df[(self.score_df['Recall'] > recall_bins[i]) & (self.score_df['Recall'] <= recall_bins[i+1])].reset_index(drop=True)
         recall_binned_df = recall_binned_df.sort_values(by='Precision', ascending=False).reset_index()
         if recall_binned_df.shape[0] > 0:
             optimal_df = optimal_df._append(recall_binned_df.iloc[0], ignore_index=True)
     fig, ax = plt.subplots()
     ax.set_xlabel('Recall')
     ax.set_ylabel('Precision')
     ax.set_title(f'Optimal Precision-Recall Curve \n Parameters: {", ".join(self.nice_keys_list)}')
     ax.set_xlim(0, 1)
     ax.set_ylim(0, 1)
     for i, recall in enumerate(optimal_df['Recall']):
         label_list = []
         for j, key in enumerate(self.keys_list):
             key_units = f'{key}_units'
             label_list.append(f'{self.nice_keys_list[j]}={optimal_df.loc[i, key]:.1e} {optimal_df.loc[i, key_units]}')
         ax.plot(recall, optimal_df.loc[i, 'Precision'], marker='o', markersize=5, label=", ".join(label_list))
     plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=7, title=f'Optimal Param Combos:')
     plt.savefig(os.path.join(self.out_dir, self.plot_folder, 'fullprplot_optimal.png'), bbox_inches='tight', dpi=250)
     plt.close()
    
 def make_singlevarying_pr_plot(self, combo_list, main_key):
     ''' Makes a PR plot for a definited set of combination values, and all the options for a single value. For example,
     this could be used to see how the EM varies the results with fixed XRSA, XRSB and temp.
     imput: 
     combo_array = list (in same order as keys_list) of specific parameter combinations, aside from the one that 
     will be changing.
     main_key = main parameter that will vary. 
     '''
     nice_main_key = self.nice_keys_list[np.where(np.array(self.keys_list) == main_key)[0][0]]
     frozen_keys = self.keys_list
     nice_frozen_keys = self.nice_keys_list
     nice_frozen_keys.pop(np.where(np.array(frozen_keys) == main_key)[0][0])
     frozen_keys.remove(main_key)
     print(frozen_keys)
     print(nice_frozen_keys)
     frozen_key_units = [self.score_df.loc[0, f'{fkey}_units'] for fkey in frozen_keys]
     frozen_keyval_combos = [list(a) for a in zip(frozen_keys, nice_frozen_keys, combo_list, frozen_key_units)]
     frozen_df = self.score_df
     for frozen_combo in frozen_keyval_combos:
         frozen_df = frozen_df[frozen_df[frozen_combo[0]]==frozen_combo[2]].reset_index(drop=True)
     title_list = [f'{frozen_key[1]}={frozen_key[2]:.1e} {frozen_key[3]}' for frozen_key in frozen_keyval_combos]
     fig, ax = plt.subplots()
     ax.set_xlabel('Recall')
     ax.set_ylabel('Precision')
     ax.set_title(f'Precision-Recall Curve Varying {nice_main_key}')
     ax.set_xlim(0, 1)
     ax.set_ylim(0, 1)
     for i, recall in enumerate(frozen_df['Recall']):
         main_units = frozen_df.loc[i, f'{main_key}_units']
         ax.plot(recall, frozen_df.loc[i, 'Precision'], marker='o', markersize=5, label=f'{nice_main_key}={frozen_df.loc[i, main_key]:.1e} {main_units}')
     plt.legend(loc='lower right', fontsize=8, title=f'{nice_main_key} values:')
     plt.text(1.01, 0.8, ("Fixed Parameters: \n" + "\n".join(title_list)), fontsize=10)
     fixed_str = [frozen[0] + str(frozen[2]) for frozen in frozen_keyval_combos]
     fixed_str = "_".join(fixed_str)
     plt.savefig(os.path.join(self.out_dir, self.plot_folder, f'prplot_varying{main_key}_{fixed_str}.png'), bbox_inches='tight', dpi=250)
     plt.close()
    
 def plot_specific_cf(self, param_dict, savestring, big_paramset=False):
     ''' Using a dictionary of parameters and keys (must be what was used for the chosen run), a CF matrix is plotted
     for that combination. 
     '''
     cf_df = self.score_df
     for key, val in param_dict.items():
         cf_df = cf_df[cf_df[key]==val].reset_index(drop=True)
     if cf_df.shape[0] > 1: raise ValueError('More than one combination left! Double check you have all parameters.')
     confusion_matrix = np.array([[cf_df.loc[0, 'TN'], cf_df.loc[0, 'TN_canc'], 
                     cf_df.loc[0, f'FP_no{self.success_flux_key}'], cf_df.loc[0, f'TP_no{self.success_flux_key}']],
                     [cf_df.loc[0, 'FN'], cf_df.loc[0, 'FN_canc'], 
                     cf_df.loc[0, f'FP_{self.success_flux_key}'], cf_df.loc[0, 'TP']]])
     fig, ax = plt.subplots()
     ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.7)
     for k in range(confusion_matrix.shape[0]):
         for j in range(confusion_matrix.shape[1]):
             ax.text(x=j, y=k,s=confusion_matrix[k, j], va='center', ha='center', size='xx-large')
     ylabels = ['', 'False', 'True']
     ax.set_ylabel(f'Flare above {self.success_flux_key}?', fontsize=14)
     ax.set_yticklabels(ylabels, rotation=90, fontsize=12)
     ax.xaxis.set_ticks_position("bottom")
     xlabels = ['', 'No Trigger', 'Cancelled Launch',f'Launch \n No {self.success_flux_key} Obs', f'Launch \n {self.success_flux_key} Obs']
     ax.set_xlabel('Launch Results', fontsize=14)
     ax.set_xticklabels(xlabels, fontsize=12)
     plt.title(f"Precision = {cf_df.loc[0, 'Precision']:.2f}, Recall = {cf_df.loc[0, 'Recall']:.2f}, Launch/Trigger Ratio = {cf_df.loc[0, 'LaunchTriggerRatio']:.2f}, \n Gordon Score = {cf_df.loc[0, 'Gordon']:.2f}, Peak Ratio = {cf_df.loc[0, 'PeakRatio']:.2f}")
     #lastly, need to make the stuff for showing the values!
     if not big_paramset:
         param_units = [cf_df.loc[0, f'{key}_units'] for key in param_dict.keys()]
         param_list = [f'{nice_key}={val} {unit}' for (key, val), unit, nice_key in zip(param_dict.items(), param_units, self.nice_keys_list)]
         plt.text(1.01, 0.8, ("Parameters \n" + "\n".join(param_list)), fontsize=12, transform=ax.transAxes)
     savestring = [f'{key}{val}' for key, val in param_dict.items()]
     savestring = "_".join(savestring)
     plt.savefig(os.path.join(self.out_dir, self.plot_folder, savestring, 'cf.png'), bbox_inches='tight', dpi=250)
     plt.close()
         
def make_summary_plots(keys_list, flux_key, nice_keys_list, score_csv, out_dir, plot_folder):
    plotter = PlottingResults(keys_list, nice_keys_list, flux_key, out_dir, score_csv, plot_folder)
    for key in keys_list:
        plotter.make_full_pr_plot(key)
        plotter.make_optimal_pr_plot()
        
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
        ax.hist(df['Max_FOXSI'], bins=logbins, range=(1e-8, 1e-2))
        if cancellation:
            ax.set_title(f'Max Flux Missed by FOXSI due to Cancellation')
            plt.savefig(os.path.join(self.out_dir, self.plot_folder, self.savestring, 'maxfoxsi_cancellation_histogram.png'), bbox_inches='tight', dpi=250)
        else:
            ax.set_title(f'Max Observed Flux for FOXSI', fontsize=16)
            plt.savefig(os.path.join(self.out_dir, self.plot_folder, self.savestring, 'maxfoxsi_launch_histogram.png'), bbox_inches='tight', dpi=250)
            
    def plot_flare_histogram_includingallflares(self):
        ''' Here is a histogram of all launches/cancellations, with all the potential flares plotted in the background.
        '''
        param_units = [self.launch_df.loc[0, f'{key}_units'] for key in self.combo_dict.keys()]
        param_list = [f'{nice_key}={val} {unit}' for (key, val), unit, nice_key in zip(self.combo_dict.items(), param_units, self.nice_keys_list)]
        logbins=np.logspace(np.log10(1e-8),np.log10(5e-3), 50)
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.hist(self.all_flare_data['peak flux'], bins=logbins, range=(1e-8, 1e-2), color='r', alpha=0.2, label='All Flares')
        ax.hist(self.cancelled_df['Flare_Max_Flux'], bins=logbins, range=(1e-8, 1e-2), color='b', label='Cancelled Flares', alpha=0.5)
        ax.hist(self.launch_df['Flare_Max_Flux'], bins=logbins, range=(1e-8, 1e-2), color='k', label='Launched Flares', alpha=0.7)
        ax.axvline(5e-6, c='k', lw=1, ls='--', label='C5')
        ax.set_xscale('log')
        ax.set_xlabel('GOES Flux W/m$^2$')
        ax.set_ylabel('# of Flares')
        ax.legend()
        plt.text(1.01, 0.8, ("Parameters: \n" + "\n".join(param_list)), fontsize=12, transform=ax.transAxes)
        ax.set_title(f'Maximum Flare Flux')
        plt.savefig(os.path.join(self.out_dir, self.plot_folder, self.savestring, 'maxflux_allflaresshown_histogram.png'), bbox_inches='tight', dpi=250)
            
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
    launch_plotter.plot_flare_histogram_includingallflares()
        
        