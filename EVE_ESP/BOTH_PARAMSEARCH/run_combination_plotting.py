import both_plotting as bp
import hic_plotting as hp
import foxsi_plotting as fp
import os

keys_list = ['xrsb_proxy'] #what ur naming the run
key = ['xrsb_proxy'] #what the keys actually are with the param dict
nice_keys_list = ['XRSB PROXY'] #to be used for plotting
flux_key = 'C5'
os.makedirs('RESULTS', exist_ok=True)
savestring = "_".join(keys_list)
out_dir = os.path.join('RESULTS', flux_key, savestring)
flare_fits = '../GOES_XRS_historical_anyneg1_gone.fits'

cf_dict = {
    'xrsb_proxy': 7e-6,
} #needs to be in the same order as everything!!!

if __name__=='__main__':
    bp.make_combination_plots(cf_dict, key, nice_keys_list, flux_key, flare_fits, out_dir, 'AllParameterScores_BOTH.csv', 'Plots_BOTH')
    hp.make_combination_plots(cf_dict, key, nice_keys_list, flux_key, flare_fits, out_dir, 'AllParameterScores_HiC.csv', 'Plots_HiC')
    fp.make_combination_plots(cf_dict, key, nice_keys_list, flux_key, flare_fits, out_dir, 'AllParameterScores_FOXSI.csv', 'Plots_FOXSI')