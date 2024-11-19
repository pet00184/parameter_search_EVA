import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table
from matplotlib import pyplot as plt
from scipy import stats as st
import math
import os


class ParameterSearch:
    
    flare_fits = '../../../GOES_XRS_computed_params.fits'
    
    def __init__(self, success_flux_key, success_flux_value, parameter_names, parameter_units, parameter_arrays, parameter_combinations, directory):
        '''Saves .fits file data to Astropy Table structure (works similarly to regular .fits, but also lets you
        parse the data by rows.)
        '''
        fitsfile = fits.open(self.flare_fits)
        self.data = Table(fitsfile[1].data)[:]
        self.header = fitsfile[1].header
        self.success_flux_key = success_flux_key
        self.success_flux_value = success_flux_value
        self.param_grid = np.array(parameter_combinations)
        self.param_arrays = parameter_arrays
        self.param_names = parameter_names
        self.param_units = parameter_units
        self.directory = directory
        self.calculated_flarelist = [] #has format of [flare #, flare ID, max foxsi, mean foxsi] for each tuple
        self.launches_df = pd.DataFrame(columns=('Flare_Number', 'Flare_ID', 'Trigger_Time', 'Cancelled?', 'Peak_Observed?', 'Max_FOXSI', f'Max_FOXSI_{self.success_flux_key}',
                        f'Mean_FOXSI_{self.success_flux_key}', f'Flare_{self.success_flux_key}', 'Flare_Class', 'Flare_Max_Flux', 'Peak_Time', 'Start_to_Peak_Time', 'Trigger_to_Peak_Time', 'Duration', 'Background_Flux'))
                        
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
        indeces of the trigger, foxsi obs start/end to be used for computing observed flux
        CANCELLATION bool, so that we know if we would have cancelled the launch or not.
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
            self.foxsi_obs_start = self.trigger_index + 3 + 4 + 2 #+ latency + launch prep + launch time
            self.foxsi_obs_end = self.foxsi_obs_start + 6       

    def calculate_observed_xrsb_and_cancellation(self, i):
        ''' Slices out the FOXSI observation windows of the current flare # (i), and calculates the max and mean 
        observed fluxes. Also calculates if the launch would be cancelled, and if the peak would have been observed.
    
        Appends [i, flare ID, foxsi max, foxsi mean] to the flarelist so that the tuples can be zipped and 
        moved to a pandas DF after all the flares are looped through.
        '''
        foxsi_obs_xrsb = self.data['xrsb'][i][self.foxsi_obs_start:self.foxsi_obs_end]
        if len(foxsi_obs_xrsb)==0: #dealing with the observation range being outside the flare (probably the next flare)
            foxsi_max_observed = math.nan
            foxsi_mean_observed = math.nan
            peak_bool = math.nan
        else:
            foxsi_max_observed = np.max(foxsi_obs_xrsb)
            foxsi_mean_observed = np.mean(foxsi_obs_xrsb)
            peak_bool = self.data['time'][i][self.foxsi_obs_start] < self.data['peak time'][i]
        flare_ID = self.data['flare ID'][i]
        trigger_time = self.data['time'][i][self.trigger_index] #will need to figure out how to get this to the correct UTC
        #calculating if cancellation will happen
        if (self.trigger_index + 3) < len(self.data['xrsa'][i]):
            cancellation_bool = (self.data['xrsa'][i][self.trigger_index + 3] - self.data['xrsa'][i][self.trigger_index]) < 0
        else:
            cancellation_bool = math.nan  
        self.calculated_flarelist.append([i, flare_ID, cancellation_bool, trigger_time, peak_bool, foxsi_max_observed, foxsi_mean_observed])

################ Post-Loop Functions ############################################################################      
    def perform_postloop_functions(self, parameter, j):
        ''' Once completed, a finished DataFrame should have info saved for all launches.
        '''
        self.save_flarelist_to_df()
        self.save_fitsinfo_to_df()
        self.calculate_c5_bool()
        self.drop_na()
    
    def save_flarelist_to_df(self):
        ''' This is done outside of the loop (for all iterations). Saves calculated values to DataFrame for each flare.
        '''
        self.launches_df[['Flare_Number', 'Flare_ID', 'Cancelled?', 'Trigger_Time', 'Peak_Observed?', 'Max_FOXSI', 'Mean_FOXSI']] = self.calculated_flarelist  
    
    def save_fitsinfo_to_df(self):
        ''' Saves the flare class, peak flux, start to peak time, and if flare is above C5 bool info from the FITS file
        using the flare number. 
        '''
        for f, flare_id in enumerate(self.launches_df['Flare_ID']):
            launched_flare = np.where(flare_id == self.data['flare ID'])[0][0]
            self.launches_df.loc[f, 'Flare_Class'] = self.data['class'][launched_flare]
            self.launches_df.loc[f, 'Flare_Max_Flux'] = self.data['peak flux'][launched_flare]
            self.launches_df.loc[f, 'Start_to_Peak_Time'] = self.data['start to peak time'][launched_flare]
            self.launches_df.loc[f, f'Flare_{self.success_flux_key}'] = self.data['peak flux'][launched_flare] > self.success_flux_value #self.data['above C5'][launched_flare]
            self.launches_df.loc[f, 'Background_Flux'] = self.data['background flux'][launched_flare]
            self.launches_df.loc[f, 'Peak_Time'] = self.data['peak time'][launched_flare] #changed this out of UTC to get the right timestamp for post analysis
            self.launches_df.loc[f, 'Duration'] = len(self.data['xrsa'][launched_flare])-30 
            self.launches_df.loc[f, 'Trigger_to_Peak_Time'] = (self.launches_df.loc[f, 'Peak_Time'] - self.launches_df.loc[f, 'Trigger_Time'])/60.0
    
    def calculate_c5_bool(self):
        ''' Saves True/False boolean results, for if the Flare and the observed flux is above C5. (Done for the flare 
        itself, max/mean FOXSI and max/mean HiC)
        '''    
        self.launches_df[f'Max_FOXSI_{self.success_flux_key}'] = self.launches_df['Max_FOXSI'] > self.success_flux_value #5e-6
        self.launches_df[f'Mean_FOXSI_{self.success_flux_key}'] = self.launches_df['Mean_FOXSI'] > self.success_flux_value #5e-6
        
    def drop_na(self):
        ''' Drops rows with Nan for observation times. This helps get rid of double counting, since sometimes the 
        next flare is triggered on the previous flare ID.
        '''
        print('before drop NA')
        print(len(self.launches_df['Flare_ID']))
        self.launches_df = self.launches_df.dropna(subset=['Max_FOXSI'])
        print('after drop NA')
        print(len(self.launches_df['Flare_ID']))
        
        
    def save_launch_DataFrame(self, parameter_savestring):
        self.launches_df.to_csv(f'{self.directory}/Launches/{parameter_savestring}_results.csv')
        print('launch dataframe saved!')
