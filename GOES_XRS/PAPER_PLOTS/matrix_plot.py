#FLARE CATEGORIZATION MATRIX PLOT
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

class FlareCategorizationMatrix:
    ''' Plot of the flare categorization matrix summary, as well as the three fcm's used for the flare campaign.
    '''
    
    C5_FILE_PATH = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/xrsb_xrsa_3minem/AllParameterScores_BOTH.csv'
    C8_FILE_PATH = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C8/xrsb_xrsa_3minem/AllParameterScores_BOTH.csv'
    M1_FILE_PATH = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/M1/xrsb_xrsa_3minem/AllParameterScores_BOTH.csv'
    
    C5_PARAM_DICT = {'xrsa': 4.5e-7,
                    'xrsb': 5.0e-6,
                    '3minem': 1.0e47}
                    
    C8_PARAM_DICT = {'xrsa': 6.0e-7,
                    'xrsb': 8.0e-6,
                    '3minem': 1e47}
    
    M1_PARAM_DICT = {'xrsa': 1.25e-6,
                    'xrsb': 1.0e-5,
                    '3minem': 7e47}
                    
    def __init__(self):
        ''' Imports csv files for each parameter search, and keeps only the parameter combination utilized in the flare campaign for each success threshold.
        '''
        c5_df = pd.read_csv(self.C5_FILE_PATH)
        c8_df = pd.read_csv(self.C8_FILE_PATH)
        m1_df = pd.read_csv(self.M1_FILE_PATH)
        
        self.c5_df = self.find_correct_combos(c5_df, self.C5_PARAM_DICT)
        self.c8_df = self.find_correct_combos(c8_df, self.C8_PARAM_DICT)
        self.m1_df = self.find_correct_combos(m1_df, self.M1_PARAM_DICT)
        
    def find_correct_combos(self, df, param_dict):
        ''' Finds the specific parameter scores for the combinations used in the campaign.
        '''
        for key, val in param_dict.items():
            df = df[df[key]==val].reset_index(drop=True)
        if df.shape[0] > 1: raise ValueError('More than one combination left! Check your param dict.')
        return df
        
    def plot_fcm(self, df, success_flux_key, ax, title, label):
        #fig, ax = plt.subplots()
        fcm = confusion_matrix = np.array([[df.loc[0, 'TN'], df.loc[0, 'TN_canc'], 
                        df.loc[0, f'FP_no{success_flux_key}_short'], df.loc[0,  
                        f'FP_no{success_flux_key}_long']],
                        [df.loc[0, 'FN'], df.loc[0, 'FN_canc'], 
                        df.loc[0, f'FP_{success_flux_key}_short'], df.loc[0, 'TP']]])
        ax.matshow(fcm, cmap=plt.cm.Blues, alpha=0.7)
        for k in range(fcm.shape[0]):
            for j in range(fcm.shape[1]):
                ax.text(x=j, y=k,s=fcm[k, j], va='center', ha='center', size='xx-large')
        ax.vlines(x=[-0.5, 0.5, 1.5, 2.5, 3.5], ymin=-0.5, ymax=1.5, color='k')
        ax.hlines(y=[-0.5, 0.5, 1.5], xmin=-0.5, xmax=3.5, color='k')
        ylabels = ['', 'False', 'True']
        ax.set_ylabel(f'Long Duration Flare \n above {success_flux_key}?', fontsize=14)
        ax.set_yticklabels(ylabels, rotation=90, fontsize=14)
        ax.xaxis.set_ticks_position("bottom")
        xlabels = ['', 'No Trigger', 'Cancelled \nLaunch','Failed \nLaunch', 'Successful \nLaunch']
        ax.set_xlabel('Results', fontsize=14)
        ax.set_xticklabels(xlabels, fontsize=14)   
        ax.set_title(title, fontsize=14)
        ax.text(-0.23, 0.5, label, fontsize=24, transform=ax.transAxes, va='center')
        #plt.savefig(f'fcm_{success_flux_key}.png', dpi=250, bbox_inches='tight') 
        return ax           
        
    def make_all_fcm_plots(self):
        fig, axs = plt.subplots(4, 1, figsize=(8, 22))
        exp_matrix = self.plot_explanation_fcm(axs[0], 'A.')
        c5_matrix = self.plot_fcm(self.c5_df, 'C5', axs[1], 'XRSA = 4.5 x $10^{-7}$ W/m$^2$, XRSB = 5 x $10^{-6}$ W/m$^2$, EM = 1 x $10^{47}$ cm$^{-3}$', 'B.')
        c8_matrix = self.plot_fcm(self.c8_df, 'C8', axs[2], 'XRSA = 6 x $10^{-7}$ W/m$^2$, XRSB = 8 x $10^{-6}$ W/m$^2$, EM = 1 x $10^{47}$ cm$^{-3}$', 'C.')
        m1_matrix = self.plot_fcm(self.m1_df, 'M1', axs[3], 'XRSA = 1.25 x $10^{-6}$ W/m$^2$, XRSB = 1 x $10^{-5}$ W/m$^2$, EM = 7 x $10^{47}$ cm$^{-3}$', 'D.')
        plt.savefig('all_mats_v1.png', bbox_inches='tight', dpi=250)
        
    def plot_explanation_fcm(self, ax, label):
        #fig, ax = plt.subplots()
        num_fcm = np.array([[2, 2, 0, 0], [1, 1, 0, 2]])
        cmap = ListedColormap(['r', 'pink', 'g'])
        exp_fcm = np.array([['TN', 'TN', 'FP', 'FP'], ['FN', 'FN', 'FP', 'TP']])
        ax.matshow(num_fcm, cmap=cmap)
        #FOR VERSION 1
        for k in range(exp_fcm.shape[0]):
            for j in range(exp_fcm.shape[1]):
                ax.text(x=j, y=k,s=exp_fcm[k, j], va='center', ha='center', size='xx-large')
        ax.vlines(x=[-0.5, 0.5, 1.5, 2.5, 3.5], ymin=-0.5, ymax=1.5, color='k')
        ax.hlines(y=[-0.5, 0.5, 1.5], xmin=-0.5, xmax=3.5, color='k')
        ylabels = ['', 'False', 'True']
        ax.set_ylabel(f'Long Duration Flare \n above success flux?', fontsize=14)
        ax.set_yticklabels(ylabels, rotation=90, fontsize=14)
        ax.xaxis.set_ticks_position("bottom")
        xlabels = ['', 'No Trigger', 'Cancelled \nLaunch','Failed \nLaunch', 'Successful \nLaunch']
        ax.set_xlabel('Results', fontsize=14)
        ax.set_xticklabels(xlabels, fontsize=14)
        ax.set_title('Scoring of Flare Categorization Matrix', fontsize=14)
        ax.text(-0.23, 0.5, label, fontsize=24, transform=ax.transAxes, va='center')
        #plt.savefig('fcm_exp_version1.png', dpi=250, bbox_inches='tight')
        #plt.show()
        return ax
        
        
    
if __name__ == '__main__':
    tester = FlareCategorizationMatrix()
    tester.make_all_fcm_plots()
    #tester.plot_explanation_fcm()
        