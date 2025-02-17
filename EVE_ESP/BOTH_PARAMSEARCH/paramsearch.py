import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
from scipy import stats as st
import math
import os


class ParameterSearch:
    ''' Parameter search utilizing GOES data as the scoring mechanism, but utilizing eve as the parameters for searching.'''

    eve_latency = 1
    goes_latency = 3
    
    def __init__(self, success_flux_key, success_flux_value, parameter_names, parameter_units, parameter_arrays, parameter_combinations, directory, goes_file, eve_file):
        '''Saves .fits file data to Astropy Table structure (works similarly to regular .fits, but also lets you
        parse the data by rows.)
        '''
        goes_fitsfile = fits.open(goes_file)
        self.goes_data = Table(goes_fitsfile[1].data)[:]
        self.goes_header = goes_fitsfile[1].header

        eve_fitsfile = fits.open(eve_file)
        self.eve_data = Table(eve_fitsfile[1].data)[:]
        self.eve_header = eve_fitsfile[1].header

        self.success_flux_key = success_flux_key
        self.success_flux_value = success_flux_value
        self.param_grid = np.array(parameter_combinations)
        self.param_arrays = parameter_arrays
        self.param_names = parameter_names
        self.param_units = parameter_units
        self.directory = directory
        self.calculated_flarelist = [] #has format of [flare #, flare ID, max hic, mean hic] for each tuple
        self.launches_df = pd.DataFrame(columns=('Flare_Number', 'Flare_ID', 'Trigger_Time', 'GOES_Trigger_Time', 'Cancelled?', 'FOXSI_Peak_Observed?', 'HiC_Post_Peak_Obs?', 'Max_FOXSI', 'Mean_FOXSI', 
                         'Max_HiC', 'Mean_HiC', 'HiC_Mean_LongDuration', 'HiC_Max_LongDuration', 'Max_FOXSI_{self.success_flux_key}',
                        f'Flare_{self.success_flux_key}', f'Flare_{self.success_flux_key}_LongDuration', 'LongDuration', 
                        'Flare_Class', 'Flare_Max_Flux', 'Peak_Time', 'Start_to_Peak_Time', 'Trigger_to_Peak_Time', 'Duration', 'Background_Flux'))
                        
        os.makedirs(f'{self.directory}/Launches', exist_ok=True)
        
    def loop_through_parameters(self):
        ''' Loops through each parameter, and performes launch analysis on each flare.
        '''
        for j, parameter in enumerate(self.param_grid):
            print(f'starting parameter search for {parameter} with success set to {self.success_flux_key}')
            parameter_savestring = "_".join([str(param) for param in parameter])
            self.loop_through_flares(parameter)
            if len(self.calculated_flarelist)>0:
                self.perform_postloop_functions(parameter, j)
                self.save_param_combo_info(parameter)
                self.save_launch_DataFrame(parameter_savestring)
                self.calculated_flarelist = []
                self.launches_df = self.launches_df.iloc[0:0]
            

################ Flare Loop Functions ############################################################################   
   
    def save_param_combo_info(self, parameter):
        ''' Saving the parameter names, units, and specific combination in a more easily accessible way for the launch df.
        '''
        for i, param_name in enumerate(self.param_names):
            self.launches_df[param_name] = parameter[i]
            self.launches_df[f'{param_name}_units'] = self.param_units[i]
        
    def loop_through_flares(self, parameter):
       ''' Loops through each flare in the calculated array that is being checked. For simplest example, array is just
       self.data['xrsb']. 
   
       Input: 
       arrays_to_check: array of flares to loop through, when checking of parameter was met. (For the simple xrsb example
           this is self.data['xrsb])
       parameter: the parameter currently being used (in simple example, this is xrsb flux level)
       '''
       for i, flare in enumerate(self.param_arrays):
           self.flareloop_check_if_value_surpassed(flare, parameter, i)
           if self.triggered_bool: 
               self.calculate_observed_xrsb_and_cancellation(i)         

    def flareloop_check_if_value_surpassed(self, arrays, parameters, i):
        ''' Process to check if a specific flare surpasses the parameter trigger levels set for this run.
        
        ADDING CANCELLATION: I am still saving what we "would have" observed if we didn't cancel, and just doing 
        a bool for cancelled. This way, we can still get some information on what we are cancelling on. We are doing a 
        simple cancellation of only cancelling if the xrsa flux is decreasing during the pre-launch window.
    
        Input: 
        array = list of arrays (flare) to be checked (xrsa, xrsb or a computed temp/derivative etc.)
        parametesr = list of values that if surpassed triggers a launch.
    
        Returns: 
        triggered_bool = True if this flare triggers a launch, otherwise is False.
        indeces of the trigger, hic obs start/end to be used for computing observed flux
        CANCELLATION bool, so that we know if we would have cancelled the launch or not.

        **EVE EDITION: both datasets have a 1 minute cadence, and I utilized the same times for each. Therefore, the trigger
        index should be the same for both. We don't need to put a later latency on the GOES time/trigger, since we aren't triggering
        on it. For realtime, we just won't know what it is until later... maybe add what we would have observed during the time of trigger?
        '''
        self.triggered_bool = False
        df = pd.DataFrame()
        if isinstance(parameters, np.float64):
            triggered_check = np.where(arrays > parameters)[0]
        elif isinstance(parameters, np.int64):
            triggered_check = np.where(arrays > parameters)[0]
        else:
            for arr, p in zip(arrays, parameters):
                df[f'param {p}'] = np.array(arr) >= p
            truth_df = df.all(1)
            triggered_check = np.where(truth_df == True)[0]
        if not len(triggered_check)==0:
            self.triggered_bool = True
            self.trigger_index = triggered_check[0]
            # launch time- we are using 30 secons, but saving the outer and inner bounds of the observation! (start is 30 seconds later and end is thirty seconds earlier than index. This only matters for mean calculations.)
            self.foxsi_obs_start = self.trigger_index + self.eve_latency + 3 + 2#+ latency + launch prep + launch time
            self.hic_obs_start = self.foxsi_obs_start + 1
            self.foxsi_obs_end = self.foxsi_obs_start + 7 #to account for 30 second discussion delay
            self.hic_obs_end = self.hic_obs_start + 7 #to account for 30 second discussion delay       

    def calculate_observed_xrsb_and_cancellation(self, i):
        ''' Slices out the HiC observation windows of the current flare # (i), and calculates the max and mean 
        observed fluxes. Also calculates if the launch would be cancelled, and if the peak would have been observed.
    
        Appends [i, flare ID, hic max, hic mean] to the flarelist so that the tuples can be zipped and 
        moved to a pandas DF after all the flares are looped through.
        '''
        foxsi_obs_xrsb = np.array(self.goes_data['xrsb'][i][self.foxsi_obs_start:self.foxsi_obs_end])
        hic_obs_xrsb = np.array(self.goes_data['xrsb'][i][self.hic_obs_start:self.hic_obs_end])
        max_mean_list = []
        for obs, start in zip([foxsi_obs_xrsb, hic_obs_xrsb], [self.foxsi_obs_start, self.hic_obs_start]):
            if len(obs) < 6: #dealing with the observation range being outside the flare (probably the next flare)
                max_observed = math.nan
                mean_observed = math.nan
                #peak_bool = math.nan
            else:
                max_observed = np.max(obs)
                #need to do a weighted mean now that we are doing 30 second discussion time:
                mean_observed = (obs[0] + np.sum(obs[1:-1])*2 + obs[-1])/12
                #peak_bool = (self.data['time'][i][14] < self.data['time'][i][start]) &  (self.data['time'][i][start] < self.data['peak time'][i])
            max_mean_list.append([max_observed, mean_observed])
        if len(foxsi_obs_xrsb) < 6:
            foxsi_peak_bool = math.nan
        else:
            foxsi_peak_bool = (self.goes_data['time'][i][14] < self.goes_data['time'][i][self.foxsi_obs_start]) &  (self.goes_data['time'][i][self.foxsi_obs_start] < self.goes_data['peak time'][i])
        if len(hic_obs_xrsb) < 6:
            hic_post_peak_bool = math.nan
        else:
            hic_post_peak_bool = self.goes_data['time'][i][self.hic_obs_start] > self.goes_data['peak time'][i]
        foxsi_max_observed, foxsi_mean_observed = max_mean_list[0][0], max_mean_list[0][1]
        hic_max_observed, hic_mean_observed = max_mean_list[1][0], max_mean_list[1][1]
        flare_ID = self.goes_data['flare ID'][i]
        trigger_time = self.eve_data['HHMM'][i][self.trigger_index] #trying to change this to EVE because the time is nicer looking
        goes_trigger_time = self.goes_data['time'][i][self.trigger_index - 2] #the time of the GOES data when we get the trigger
        #calculating if cancellation will happen- SWITCH THIS TO EVE 0-7 NM DECREASING???
        if (self.trigger_index + 3) < len(self.eve_data['XRS-B_proxy'][i]):
            cancellation_bool = (self.eve_data['XRS-A_proxy'][i][self.trigger_index + 3] - self.eve_data['XRS-A_proxy'][i][self.trigger_index]) < 0
        else:
            cancellation_bool = math.nan  
        self.calculated_flarelist.append([i, flare_ID, cancellation_bool, trigger_time, goes_trigger_time, foxsi_peak_bool, hic_post_peak_bool, foxsi_max_observed, foxsi_mean_observed, hic_max_observed, hic_mean_observed])

################ Post-Loop Functions ############################################################################      
    def perform_postloop_functions(self, parameter, j):
        ''' Once completed, a finished DataFrame should have info saved for all launches.
        '''
        self.save_flarelist_to_df()
        self.save_fitsinfo_to_df()
        self.calculate_c5_bool()
        self.calculate_Successlevel_andLongDuration()
        self.calculate_HiCobs_success()
        self.drop_na()
    
    def save_flarelist_to_df(self):
        ''' This is done outside of the loop (for all iterations). Saves calculated values to DataFrame for each flare.
        '''
        self.launches_df[['Flare_Number', 'Flare_ID', 'Cancelled?', 'Trigger_Time', 'GOES _Trigger_Time', 'FOXSI_Peak_Observed?', 'HiC_Post_Peak_Obs?', 'Max_FOXSI', 'Mean_FOXSI', 'Max_HiC', 'Mean_HiC']] = self.calculated_flarelist  
    
    def save_fitsinfo_to_df(self):
        ''' Saves the flare class, peak flux, start to peak time, and if flare is above C5 bool info from the FITS file
        using the flare number. 
        '''
        for f, flare_id in enumerate(self.launches_df['Flare_ID']):
            launched_flare = np.where(flare_id == self.goes_data['flare ID'])[0][0]
            self.launches_df.loc[f, 'Flare_Class'] = self.goes_data['class'][launched_flare]
            self.launches_df.loc[f, 'Flare_Max_Flux'] = self.goes_data['peak flux'][launched_flare]
            self.launches_df.loc[f, 'Start_to_Peak_Time'] = self.goes_data['start to peak time'][launched_flare]
            self.launches_df.loc[f, f'Flare_{self.success_flux_key}'] = self.goes_data['peak flux'][launched_flare] > self.success_flux_value #self.data['above C5'][launched_flare]
            self.launches_df.loc[f, 'Background_Flux'] = self.goes_data['background flux'][launched_flare]
            self.launches_df.loc[f, 'Peak_Time'] = self.goes_data['peak time'][launched_flare] #changed this out of UTC to get the right timestamp for post analysis
            self.launches_df.loc[f, 'Duration'] = len(self.goes_data['xrsa'][launched_flare])-30 
            self.launches_df.loc[f, 'Trigger_to_Peak_Time'] = (self.launches_df.loc[f, 'Peak_Time'] - self.launches_df.loc[f, 'Trigger_Time'])/60.0
            if self.foxsi_obs_start < len(self.goes_data['time'][launched_flare]):
                self.launches_df.loc[f, 'Start_to_FOXSI_Obs'] = self.goes_data['time'][launched_flare][self.foxsi_obs_start] - self.goes_data['time'][launched_flare][14]
            # Saves True/False booleans for if the flare is over 20 percent of the peak value for at least 20 minutes.
            min_flux_level = self.launches_df.loc[f, 'Flare_Max_Flux']*0.2
            peak_time_indx = np.where(self.goes_data['time'][launched_flare]==self.launches_df.loc[f, 'Peak_Time'])[0]
            if len(peak_time_indx)>0:
                if peak_time_indx[0] + 20 >= len(self.goes_data['time'][launched_flare]):
                    final_flux = 0
                else:
                    final_flux = self.goes_data['time'][launched_flare][peak_time_indx[0]+20] #20 minutes later
                self.launches_df.loc[f, 'LongDuration'] = final_flux > min_flux_level
            else:
                self.launches_df.loc[f, 'LongDuration'] = math.nan
    
    def calculate_c5_bool(self):
        ''' Saves True/False boolean results, for if the Flare and the observed flux is above C5. (Done for the flare 
        itself, max/mean FOXSI and max/mean HiC)
        '''    
        self.launches_df[f'Max_FOXSI_{self.success_flux_key}'] = self.launches_df['Max_FOXSI'] > self.success_flux_value
        self.launches_df[f'Mean_FOXSI_{self.success_flux_key}'] = self.launches_df['Mean_FOXSI'] > self.success_flux_value 
    
    def calculate_Successlevel_andLongDuration(self):
        '''' Saves True/False booleans for if the flare is over 20 percent of the peak value for at least 20 minutes and
        the flux level for the flare was met
        '''
        self.launches_df[f'Flare_{self.success_flux_key}_LongDuration'] = (self.launches_df[f'Flare_{self.success_flux_key}']==True) & (self.launches_df['LongDuration']==True)
        
    def calculate_HiCobs_success(self):
        ''' Saves True/False booleans for if the actual observation occured when the flux was at least 20 percent of the peak.
        '''  
        self.launches_df['HiC_Max_LongDuration'] = self.launches_df['Max_HiC'] > self.launches_df['Flare_Max_Flux']*0.2
        self.launches_df['HiC_Mean_LongDuration'] = self.launches_df['Mean_HiC'] > self.launches_df['Flare_Max_Flux']*0.2
        
    def drop_na(self):
        ''' Drops rows with Nan for observation times. This helps get rid of double counting, since sometimes the 
        next flare is triggered on the previous flare ID.
        '''
        print('before drop NA')
        print(len(self.launches_df['Flare_ID']))
        self.launches_df = self.launches_df.dropna(subset=['Max_HiC', 'LongDuration'])
        print('after drop NA')
        print(len(self.launches_df['Flare_ID']))
        
        
    def save_launch_DataFrame(self, parameter_savestring):
        self.launches_df.to_csv(f'{self.directory}/Launches/{parameter_savestring}_results.csv')
        print('launch dataframe saved!')
