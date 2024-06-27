#BIG PR PLOT
import pandas as pd
import matplotlib.pyplot as plt

class BigPRPlots:
    ''' Makes the array of 6 PR summary plots to show for the single results.
    '''
    
    XRSA_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/xrsa/AllParameterScores_BOTH.csv'
    XRSB_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/xrsb/AllParameterScores_BOTH.csv'
    EM3MIN_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/3minem/AllParameterScores_BOTH.csv'
    TEMP3MIN_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/3mintemp/AllParameterScores_BOTH.csv'
    XRSA3MIN_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/3minxrsa/AllParameterScores_BOTH.csv'
    XRSB3MIN_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/3minxrsb/AllParameterScores_BOTH.csv'
    
    def __init__(self):
        self.xrsa_df = pd.read_csv(self.XRSA_FILE)
        self.xrsb_df = pd.read_csv(self.XRSB_FILE)
        self.em3min_df = pd.read_csv(self.EM3MIN_FILE)
        self.temp3min_df = pd.read_csv(self.TEMP3MIN_FILE)
        self.xrsa3min_df = pd.read_csv(self.XRSA3MIN_FILE)
        self.xrsb3min_df = pd.read_csv(self.XRSB3MIN_FILE)
        
    def make_pr_plot(self, ax, df, title, key, label):
        #fig, ax = plt.subplots(1, 1)
        cmap = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']*5
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for i, recall in enumerate(df['Recall']):
            ax.scatter(recall, df['Precision'].iloc[i], marker='o', s=55, label=df[key].iloc[i], color=cmap[i])
        ax.legend(loc='upper right', ncols=1, fontsize=9, frameon=False, reverse=True, markerscale=0.8, title='Threshold Values [W/m$^2$]', title_fontsize=9, alignment='right')
        ax.text(0.1, 0.9, label, fontsize=24, transform=ax.transAxes, va='center')
        #plt.show()
        return ax
        
    def make_plot_grid(self):
        fig, axs = plt.subplots(3, 2, figsize=(15, 20))
        xrsa_plot = self.make_pr_plot(axs[0, 0], self.xrsa_df, 'XRSA', 'xrsa', 'A.')
        xrsb_plot = self.make_pr_plot(axs[0, 1], self.xrsb_df, 'XRSB', 'xrsb', 'B.')
        em3min_plot = self.make_pr_plot(axs[2, 0], self.em3min_df, 'EM (from 3-minute dXRSA and dXRSB)', '3minem', 'E.')
        temp3min_plot = self.make_pr_plot(axs[2, 1], self.temp3min_df, 'Temperature (from 3-minute dXRSA and dXRSB)', '3mintemp', 'F.')
        xrsa3min_plot = self.make_pr_plot(axs[1, 0], self.xrsa3min_df, '3-minute dXRSA', '3minxrsa', 'C.')
        xrsb3min_plot = self.make_pr_plot(axs[1, 1], self.xrsb3min_df, '3-minute dXRSB', '3minxrsb', 'D.')
        plt.savefig('big_pr_plot.png', bbox_inches='tight', dpi=250)
        
            
        
        
if __name__ == '__main__':
    tester = BigPRPlots()
    tester.make_plot_grid()