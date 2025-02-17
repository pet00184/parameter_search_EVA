import paramsearch as ps
import save_scores_both as ss_both
import save_scores_hic as ss_hic
import save_scores_foxsi as ss_fox
import both_plotting as pl
import os
from astropy.io import fits
import numpy as np
import pandas as pd
import functools
import warnings
import sys
import multiprocessing as mp
warnings.filterwarnings("ignore")

eve_params = '../EVE_ESP_historical_anyneg1_gone.fits' #change depending on which GOES/EVE files you want!!
goes_file = '../GOES_XRS_historical_anyneg1_gone.fits'

eve_hdu = fits.open(eve_params)
eveparam = eve_hdu[1].data

fitsfile = fits.open(goes_file)
goes_flare_data = fitsfile[1].data


################# Dictionary of all Parameters ################################################################        
params = {
    '07nm_esp': [[0, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-5], eveparam['0.1-7_ESPquad'], 'W/m^2'],
    '17nm_esp': [[0, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3], eveparam['17.1_ESP'], 'W/m^2'],
    '25nm_esp': [[0, 5e-4, 6e-4, 7e-4, 8e-4], eveparam['25.7_ESP'], 'W/m^2'],
    '30nm_esp': [[0, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3], eveparam['30.4_ESP'], 'W/m^2'],
    'xrsb_proxy': [[0, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6], eveparam['XRS-B_proxy'], 'W/m^2'],
    'xrsa_proxy': [[0, 3e-7, 4e-7, 5e-7, 6e-7, 7e-7], eveparam['XRS-A_proxy'], 'W/m^2'],
    
    }
    
def make_param_info(keys_list):
    param_combinations = np.array(np.meshgrid(*[params[key][0] for key in keys_list])).T.reshape(-1, len(keys_list))
    param_arrays = list(zip(*[params[key][1] for key in keys_list]))
    param_units = [params[key][2] for key in keys_list]
    return keys_list, param_combinations, param_arrays, param_units  

############### Dictionary of what flux level observed by FOXSI counts as success #############################
''' These flux levels can be changed so that parameter searches may be done for if the Sun is quiet and we want to 
count C4, C3 flux observations as success. For the main parameter search, we will use C5, in line with what 
HiC has defined as the minimum flare class they want. THIS IS NOT FOR OBSERVATION, BUT FOR WHAT THE MAX FLUX IS'''

success_flux_vals = {
    'C1': 1e-6,
    'C2': 2e-6,
    'C3': 3e-6, 
    'C4': 4e-6, 
    'C5': 5e-6, 
    'C8': 8e-6,
    'M1': 1e-5
    }

################################################################################################################

def run_paramsearch(param_directory, flux_key, param_names, param_units, param_array_list, goes_file, eve_file, param_combo_list):
    flux_val = success_flux_vals[flux_key]
    param_search = ps.ParameterSearch(flux_key, flux_val, param_names, param_units, param_array_list, param_combo_list, param_directory, goes_file, eve_file)
    param_search.loop_through_parameters()

def run_multiprocessing_paramsearch(keys_list, out_dir, flux_key, goes_file, eve_file):
    os.makedirs(out_dir, exist_ok=True)
    param_names, param_combinations, param_arrays, param_units = make_param_info(keys_list)
    #getting the number of cores for the slurm job
    try:
        num_cores = int(sys.argv[1])
    except IndexError:
        num_cores = os.cpu_count()
    print('num cores used:', num_cores)
    print('Total Params:', len(param_combinations))
    #splitting the array so all available cores are used
    splitup = np.array_split(param_combinations, num_cores)
    #doing the multiple runs!
    call_me = functools.partial(run_paramsearch, out_dir, flux_key, param_names, param_units, param_arrays, goes_file, eve_file)
    with mp.Pool(num_cores) as p:
        p.map(call_me, splitup)

##################################################################################################################

def run_savescores(out_dir, flux_key, param_names, param_units, which_savescore, launches_and_tag):
    tag, launch_df_list = launches_and_tag
    flux_val = success_flux_vals[flux_key]
    save_scores = which_savescore(out_dir, flux_key, flux_val, launch_df_list, tag, param_names, param_units)
    save_scores.loop_through_param_combos()

def run_multiprocessing_savescores(keys_list, out_dir, flux_key, score_name, which_savescore):
    param_names, _, _, param_units = make_param_info(keys_list)
    launches_list = np.array(os.listdir(os.path.join(out_dir, 'Launches')))
    print(launches_list)
    try:
        num_cores = int(sys.argv[1])
    except IndexError:
        num_cores = os.cpu_count()
    print('num cores used:', num_cores)
    print('Number of combinations:', len(launches_list))
    splitup = np.array_split(launches_list, num_cores)
    splitup = [[i, s] for (i, s) in enumerate(splitup)]
    call_me = functools.partial(run_savescores, out_dir, flux_key, param_names, param_units, which_savescore)
    with mp.Pool(num_cores) as p:
        p.map(call_me, splitup)
    make_large_df(keys_list, out_dir, score_name)

def make_large_df(keys_list, out_dir, score_name):
    if os.path.exists(os.path.join(out_dir, score_name)):
        os.remove(os.path.join(out_dir, score_name))
    score_files = [f for f in os.listdir(out_dir) if f.endswith('temp.csv')]
    total_score_df = pd.concat([pd.read_csv(os.path.join(out_dir, file), index_col=0) for file in score_files], ignore_index=True)
    total_score_df = total_score_df.sort_values(by=keys_list)
    total_score_df = total_score_df.reset_index(drop=True)
    total_score_df.to_csv(os.path.join(out_dir, score_name))
    print('All parameter scores saved.')
    for score_file in score_files:
        os.remove(os.path.join(out_dir, score_file))

##################################################################################################################

if __name__ == '__main__':
    keys_list = ['xrsb_proxy'] #what ur naming the run
    key = ['xrsb_proxy'] #what the keys actually are with the param dict
    nice_keys_list = ['XRSB PROXY'] #to be used for plotting
    flux_key = 'C5'
    os.makedirs('RESULTS', exist_ok=True)
    savestring = "_".join(keys_list)
    out_dir = os.path.join('RESULTS', flux_key, savestring)
    run_multiprocessing_paramsearch(key, out_dir, flux_key, goes_file, eve_params)
    #running save scores
    run_multiprocessing_savescores(key, out_dir, flux_key, 'AllParameterScores_HiC.csv', ss_hic.SaveScores)
    run_multiprocessing_savescores(key, out_dir, flux_key, 'AllParameterScores_FOXSI.csv', ss_fox.SaveScores)
    run_multiprocessing_savescores(key, out_dir, flux_key, 'AllParameterScores_BOTH.csv', ss_both.SaveScores)
    # make summary plots for foxsi, hic and both 
    pl.make_summary_plots(key, flux_key, nice_keys_list, 'AllParameterScores_HiC.csv', out_dir, 'Plots_HiC')
    pl.make_summary_plots(key, flux_key, nice_keys_list, 'AllParameterScores_FOXSI.csv', out_dir, 'Plots_FOXSI')
    pl.make_summary_plots(key, flux_key, nice_keys_list, 'AllParameterScores_BOTH.csv', out_dir, 'Plots_BOTH')
