#COMPARISON OF THE FOXIS, HIC AND BOTH SCORING PR CURVES
import pandas as pd
import matplotlib.pyplot as plt

class ComparisonPRPlots:
    ''' Makes the plot that compares scoring with FOXSI, HIC and BOTH criteria. 
    '''
    
    XRSB_BOTH = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/xrsb/AllParameterScores_BOTH.csv'
    XRSB_FOXSI = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/xrsb/AllParameterScores_FOXSI.csv'
    XRSB_HIC = '/Users/pet00184/Flare_Prediction/HistoricalFlare_HiCUpdated/GOES_XRS/BOTH_Paramsearch/RESULTS/C5/xrsb/AllParameterScores_HIC.csv'
    
    def __init__(self):
        self.xrsb_both_df = pd.read_csv(self.XRSB_BOTH)
        self.xrsb_foxsi_df = pd.read_csv(self.XRSB_FOXSI)
        self.xrsb_hic_df = pd.read_csv(self.XRSB_HIC)
        
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
        fig, axs = plt.subplots(3, 1, figsize=(8, 20))
        both_plot = self.make_pr_plot(axs[0], self.xrsb_both_df, 'XRSB Utilizing Both Success Criteria', 'xrsb', 'A.')
        foxsi_plot = self.make_pr_plot(axs[1], self.xrsb_foxsi_df, 'XRSB Utilizing FOXSI Success Criteria', 'xrsb', 'B.')
        hic_plot = self.make_pr_plot(axs[2], self.xrsb_hic_df, 'XRSB Utilizing HiC Success Criteria', 'xrsb', 'C.')
        plt.savefig('comparison_pr_plot.png', bbox_inches='tight', dpi=250)
        #plt.show()
        
if __name__ == '__main__':
    tester = ComparisonPRPlots()
    tester.make_plot_grid()