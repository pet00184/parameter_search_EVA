import plotting as fp
import os

''' This script is for when you find a specifc parameter combination that looks interesting, and you want more information
on what it looks like. This will give you a flare categorization matrix, and well as some of the other histograms shown
in the SPHERE presentation.
'''

#these are the same definitions as what is in run_paramsearch.py!
keys_list = ['xrsb'] 
key = ['xrsb'] 
nice_keys_list = ['XRSB'] 
flux_key = 'C5'
os.makedirs('RESULTS', exist_ok=True)
savestring = "_".join(keys_list)
out_dir = os.path.join('RESULTS', flux_key, savestring)
flare_fits = '../../../GOES_XRS_computed_params.fits'

''' This is the dictionary of the specific combination that you want to make special plots for. 
    ** Note: the order of the parameters in the dictionary must be the same order that you list in keys_list
    ** Note: if you are getting errors of not finding a file, it may be because the specific value is input incorrectly.
     When not doing something in scientific notaion, adding a . at the end of the number (like 0.) usually fixes the issue.
'''
cf_dict = {
    'xrsb': 5e-6
} 


if __name__=='__main__':
    fp.make_combination_plots(cf_dict, key, nice_keys_list, flux_key, flare_fits, out_dir, 'AllParameterScores.csv', 'FOXSI_Plots')