import both_plotting as bp
import hic_plotting as hp
import foxsi_plotting as fp
import os

keys_list = ['xrsb', 'xrsa', '3minem'] #what ur naming the run
key = ['xrsb', 'xrsa', '3minem'] #what the keys actually are with the param dict
nice_keys_list = ['XRSB', 'XRSA', 'dEmission Measure (3 min)'] #to be used for plotting
flux_key = 'C5'
os.makedirs('RESULTS', exist_ok=True)
savestring = "_".join(keys_list)
out_dir = os.path.join('RESULTS', flux_key, savestring)
flare_fits = '../GOES_XRS_historical.fits'

cf_dict = {
    'xrsb': 5e-6,
    'xrsa': 4.5e-7,
    '3minem': 1e47
} #needs to be in the same order as everything!!!

if __name__=='__main__':
    bp.make_combination_plots(cf_dict, key, nice_keys_list, flux_key, flare_fits, out_dir, 'AllParameterScores_BOTH.csv', 'Plots_BOTH')
    hp.make_combination_plots(cf_dict, key, nice_keys_list, flux_key, flare_fits, out_dir, 'AllParameterScores_HiC.csv', 'Plots_HiC')
    fp.make_combination_plots(cf_dict, key, nice_keys_list, flux_key, flare_fits, out_dir, 'AllParameterScores_FOXSI.csv', 'Plots_FOXSI')