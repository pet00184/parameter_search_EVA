import paramsearch as ps
import save_scores as ss
import plotting as pl
import os
from astropy.io import fits
import numpy as np
import pandas as pd
import functools
import warnings
import sys
import multiprocessing as mp
warnings.filterwarnings("ignore")

'''The FITS file with all the flares and flare data is defined below. For each array we are intersted in, 
we do flare_data['keyword']
'''
flare_fits = '../../../GOES_XRS_computed_params.fits'
fitsfile = fits.open(flare_fits)
flare_data = fitsfile[1].data


################# Dictionary of all Parameters ################################################################    
'''Below is a dictionary of all the parameters I have made so far for the runs. The setup for each entry is as follows:
    -- keyword : [[array of threshold values to try], corresponding array of flares, units]
'''    
params = {
    #long and short wavelength flux (this is the base values we get from GOES XRS)
    'xrsb': [[0, 1e-6, 2e-6, 3e-6, 4e-6, 5e-6, 6e-6, 7e-6], flare_data['xrsb'], 'W/m^2'],
    'xrsa': [[0, 2.5e-7, 3e-7, 3.5e-7, 4e-7, 4.5e-7, 5e-7], flare_data['xrsa'], 'W/m^2'],
    #1, 3 and 5 minute XRSB differences
    '1minxrsb': [[0, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6], flare_data['XRSB 1-min Differences'], 'W/m^2'],
    '3minxrsb': [[0, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6], flare_data['XRSB 3-min Differences'], 'W/m^2'],
    '5minxrsb': [[0, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6], flare_data['XRSB 5-min Differences'], 'W/m^2'],
    #1, 3, and 5 minute XRSA differences
    '1minxrsa': [[0, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6], flare_data['XRSA 1-min Differences'], 'W/m^2'],
    '3minxrsa': [[0, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6], flare_data['XRSA 3-min Differences'], 'W/m^2'],
    '5minxrsa': [[0, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6], flare_data['XRSA 5-min Differences'], 'W/m^2'],
    #Temperature calculated from 1, 3, and 5 minute flux differences (like a background subtracted temp)
    'temp_from1minxrs': [[0, 6, 8, 10, 12], flare_data['Temp (XRS 1-min Differences)'], 'MK'],
    'temp_from3minxrs': [[0, 6, 8, 10, 12], flare_data['Temp (XRS 3-min Differences)'], 'MK'],
    'temp_from5minxrs': [[0, 6, 8, 10, 12], flare_data['Temp (XRS 5-min Differences)'], 'MK'],
    #Emission Measure calculated from 1, 3, and 5 minute flux differences (like a background subtracted EM)
    'em_from1minxrs': [[5e47, 7e47, 1e48, 3e48], flare_data['EM (XRS 1-min Differences)'], 'cm^-3'], 
    'em_from3minxrs': [[5e47, 7e47, 1e48, 3e48], flare_data['EM (XRS 3-min Differences)'], 'cm^-3'], 
    'em_from5minxrs': [[1e47, 5e47, 7e47, 1e48, 3e48], flare_data['EM (XRS 5-min Differences)'], 'cm^-3'], 
    #Temperature
    'temperature': [[0, 6, 8, 10, 12], flare_data['Temperature'], 'MK'],
    #Emission Measure
    'emissionmeasure': [[5e47, 7e47, 1e48, 3e48], flare_data['Emission Measure'], 'cm^-3'],
    #1, 3, and 5 minute Temperature differences
    '1mintemp': [[0, 6, 8, 10, 12], flare_data['1-minute Temperature Difference'], 'MK'],
    '3mintemp': [[0, 6, 8, 10, 12], flare_data['3-minute Temperature Difference'], 'MK'],
    '5mintemp': [[0, 6, 8, 10, 12], flare_data['5-minute Temperature Difference'], 'MK'],
    #1, 3, and 5 minute Emission Measure differences
    '1minem': [[5e47, 7e47, 1e48, 3e48], flare_data['1-minute Emission Measure Difference'], 'cm^-3'], 
    '3minem': [[5e47, 7e47, 1e48, 3e48], flare_data['3-minute Emission Measure Difference'], 'cm^-3'], 
    '5minem': [[1e47, 5e47, 7e47, 1e48, 3e48], flare_data['5-minute Emission Measure Difference'], 'cm^-3'],
    
    }
    
def make_param_info(keys_list):
    ''' 
    Input: list of parameter dictionary keywords for all parameters you wish to test.
    Returns:
    ------------------------
        keys_list = list of the parameter keys given
        param_combinations = every possible combination of the parameter thresholds to test
        param_arrays = the flare arrays needed for each parameter
        param_units = units of each parameter
    '''
    param_combinations = np.array(np.meshgrid(*[params[key][0] for key in keys_list])).T.reshape(-1, len(keys_list))
    param_arrays = list(zip(*[params[key][1] for key in keys_list]))
    param_units = [params[key][2] for key in keys_list]
    return keys_list, param_combinations, param_arrays, param_units  

############### Dictionary of what flux level observed by FOXSI counts as success #############################
''' These flux levels can be changed so that parameter searches may be done for if the Sun is quiet and we want to 
count C4, C3 flux observations as success. For the main parameter search, we will use C5, in line with what 
FOXSI has defined as its nominal success criteria. 
        ** start with C5 for now! (which is what I have it set to already)
'''

success_flux_vals = {
    'C1': 1e-6,
    'C2': 2e-6,
    'C3': 3e-6, 
    'C4': 4e-6, 
    'C5': 5e-6
    }

################################################################################################################

def run_paramsearch(param_directory, flux_key, param_names, param_units, param_array_list, param_combo_list):
    ''' Runs the parameter search code on the parameters and success flux level being tested. This is run within the
    run_multiprocessing_paramsearch() function.
    '''
    flux_val = success_flux_vals[flux_key]
    param_search = ps.ParameterSearch(flux_key, flux_val, param_names, param_units, param_array_list, param_combo_list, param_directory)
    param_search.loop_through_parameters()

def run_multiprocessing_paramsearch(keys_list, out_dir, flux_key):
    ''' Separates the run_paramsearch() function so that it can be run on all available cores. This speeds up the process
    by a lot!
    '''
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
    call_me = functools.partial(run_paramsearch, out_dir, flux_key, param_names, param_units, param_arrays)
    with mp.Pool(num_cores) as p:
        p.map(call_me, splitup)

def run_savescores(out_dir, flux_key, param_names, param_units, launches_and_tag):
    ''' Once the parameter search is run, this function runs the save_scores code. During this step, scores such as 
    the precision, recall, accuracy, etc. are saved for each parameter combination.
    '''
    tag, launch_df_list = launches_and_tag
    flux_val = success_flux_vals[flux_key]
    save_scores = ss.SaveScores(out_dir, flux_key, flux_val, launch_df_list, tag, param_names, param_units)
    save_scores.loop_through_param_combos()

def run_multiprocessing_savescores(keys_list, out_dir, flux_key):
    ''' This is the same idea as above, separating the run_savescores() function among all available cores.
    '''
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
    call_me = functools.partial(run_savescores, out_dir, flux_key, param_names, param_units)
    with mp.Pool(num_cores) as p:
        p.map(call_me, splitup)
    make_large_df(keys_list, out_dir)

def make_large_df(keys_list, out_dir):
    ''' Once all cores are finished running, this amasses all the parameter scores into one large .csv file, so that 
    it is easier to look at all combinations.
    '''
    if os.path.exists(os.path.join(out_dir, 'AllParameterScores.csv')):
        os.remove(os.path.join(out_dir, 'AllParameterScores.csv'))
    score_files = [f for f in os.listdir(out_dir) if f.endswith('.csv')]
    total_score_df = pd.concat([pd.read_csv(os.path.join(out_dir, file), index_col=0) for file in score_files], ignore_index=True)
    total_score_df = total_score_df.sort_values(by=keys_list)
    total_score_df = total_score_df.reset_index(drop=True)
    total_score_df.to_csv(os.path.join(out_dir, 'AllParameterScores.csv'))
    print('All parameter scores saved.')
    for score_file in score_files:
        os.remove(os.path.join(out_dir, score_file))

##################################################################################################################

if __name__ == '__main__':
    ''' Below is what actaully happens when you run this file. Right now, I have it set up to show a simple parameter 
    search that only looks at XRSB values. 
    '''
    keys_list = ['xrsb'] #what you want to name the run (this can be the same as  key)
    key = ['xrsb'] #what the keys actually are with the param dict- they need to match what is written in param_dict
    nice_keys_list = ['XRSB'] #these are "prettier" names for each parameter you want to test, which are used when plotting so its easier to understand what we are looking at.
    flux_key = 'C5' #this is the success flux key. As stated above, no need to change this when we are starting out! This basically says that FOXSI is happy with observing anyting above C5.
    ### below just sets up where the results are going to be saved
    os.makedirs('RESULTS', exist_ok=True)
    savestring = "_".join(keys_list)
    out_dir = os.path.join('RESULTS', flux_key, savestring)
    score_csv = 'AllParameterScores.csv'
    #### actually running the parameter search
    run_multiprocessing_paramsearch(key, out_dir, flux_key)
    ### saving the scores
    run_multiprocessing_savescores(key, out_dir, flux_key)
    ### basic plotting, which will give us a PR curve to compare all the parameter combinations
    pl.make_summary_plots(key, flux_key, nice_keys_list, score_csv, out_dir, 'FOXSI_Plots')
