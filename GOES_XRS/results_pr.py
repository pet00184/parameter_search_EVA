#PR CURVES FOR THE THREE RESULTS UTILIZED IN THE LAUNCH CAMPAIGN
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class ResultsPRPlots:
    ''' Makes the three PR plots, with optimal values overlaid for all three combiantions utilized in the launch campaign.
    '''
    
    C5_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/xrsb_xrsa_3minem/AllParameterScores_BOTH.csv'
    C8_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C8/xrsb_xrsa_3minem/AllParameterScores_BOTH.csv'
    M1_FILE = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/M1/xrsb_xrsa_3minem/AllParameterScores_BOTH.csv'
    
    def __init__(self):
        self.c5_df = pd.read_csv(self.C5_FILE)
        self.c8_df = pd.read_csv(self.C8_FILE)
        self.m1_df = pd.read_csv(self.M1_FILE)
        
    def make_pr_plot(self, ax, df, title, label):
        #fig, ax = plt.subplots(1, 1)
        self.make_optimal_df(df)
        cmap = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']*5
        print(ax)
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title(title, fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        for i, recall in enumerate(df['Recall']):
            ax.scatter(recall, df['Precision'].iloc[i], marker='o', s=55, label=None, color='gray', alpha=.5)
        for i, recall in enumerate(self.optimal_df['Recall']):
            ax.scatter(recall, self.optimal_df['Precision'].iloc[i], marker='o', s=55, color=cmap[i])
        ax.text(0.05, 0.9, label, fontsize=24, transform=ax.transAxes, va='center')
        
        self.make_table(ax, df)
        #plt.show()
        
    def make_optimal_df(self, df):
        #doing optimal recall on top
        if hasattr(self, 'optimal_df'): del self.optimal_df
        self.optimal_df = pd.DataFrame()
        recall_bins = np.arange(0, 1.025, .025)
        for i in range(len(recall_bins)-1):
            recall_binned_df = df[(df['Recall'] > recall_bins[i]) & (df['Recall'] <= recall_bins[i+1])].reset_index(drop=True)
            recall_binned_df = recall_binned_df.sort_values(by='Precision', ascending=False).reset_index()
            if recall_binned_df.shape[0] > 0:
                self.optimal_df = self.optimal_df._append(recall_binned_df.iloc[0], ignore_index=True)
        
    def make_big_plot(self):
        fig, axs = plt.subplots(3, 1, figsize=(8, 20))
        self.make_pr_plot(axs[0], self.c5_df, 'C5 Success', 'A.')
        self.make_pr_plot(axs[1], self.c8_df, 'C8 Success', 'B.')
        self.make_pr_plot(axs[2], self.m1_df, 'M1 Success', 'C.')
        plt.savefig('results_plot.png', bbox_inches='tight', dpi=250)
        plt.show()
        
    def make_table(self, ax, df):
        ''' Try running this after doing the 
        '''
        #self.make_optimal_df(df)
        xrsa = np.array(self.optimal_df['xrsa'])
        xrsb = np.array(self.optimal_df['xrsb'])
        em = np.array(self.optimal_df['3minem'])
        attempt = np.vstack((xrsa, xrsb, em)).T
        cmap = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']*5
        cmap = cmap[0:len(xrsa)]
        cell_text = [[1,2,3], [1,2,3], [1,2,3]]
        #fig, ax = plt.subplots(1,1)
        ax.table(attempt, cellLoc='center', rowColours=cmap, rowLabels=['   ']*len(cmap),  colLabels=['XRSA [W/m$^2$]', 'XRSB [W/m$^2$]', 'EM (from 3-min dXRS) [cm$^{-3}$]'], loc='right', bbox=[1.1,0,1.0,1])
        ax.set_xlim(0, 1)
        #plt.savefig('testertable.png', bbox_inches='tight')
        return ax
        
        
        
        
if __name__ == '__main__':
    tester = ResultsPRPlots()
    tester.make_big_plot()
    #tester.make_table(tester.c5_df)
