import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.table import Table
import os

class SaveScores:
    ''' Class for a multiprocess approach to saving the scores of every Launch file for every parameter combination.
    
    Input:
    ----------------------------------------------------------------
    Launch_dir = the directory that points to the listdir of the launch files.
    Launch_df_list = list of launch csv files to be looped over. The scores from each will be appended to a scores dataframe, 
    which will then be saved.
    Tag = number to tag the score csv file when naming it. The number of tags is the number of cores running the script.
    
    Output:
    -----------------------------------------------------------------
    Score_df = .csv file with this thread's launch scores saved. After the multiprocess is run, a function will combine all of 
    these separate scores into one large .csv file.
    '''
    
    flare_fits = '../GOES_XRS_historical_anyneg1_gone.fits'
    
    def __init__(self, out_dir, success_flux_key, success_flux_value, launch_df_list, tag, param_names, param_units):
        fitsfile = fits.open(self.flare_fits)
        self.data = Table(fitsfile[1].data)[:]
        self.header = fitsfile[1].header
        self.out_dir = out_dir
        self.launch_dir = os.path.join(out_dir, 'Launches')
        self.launch_df_list = launch_df_list
        self.tag = tag
        self.param_names = param_names
        self.param_units = param_units
        self.success_flux_key = success_flux_key
        self.success_flux_value = success_flux_value
        
    def loop_through_param_combos(self):
        self.score_df = pd.DataFrame(columns=('Precision', 'Recall', 'Gordon', 'LaunchTriggerRatio', 'Accuracy', 'FOXSI_PeakRatio', 'HiC_ObsAfterPeak', 'Fbeta', 'TN', 'TN_canc', 'FN', 'FN_canc', 
                        f'FP_{self.success_flux_key}_short', f'FP_no{self.success_flux_key}_short', f'FP_no{self.success_flux_key}_long', 'TP'))
        self.append_param_names_and_units()
        for i, param_combo in enumerate(self.launch_df_list):
            param_df = pd.read_csv(os.path.join(self.launch_dir, param_combo))
            self.make_all_above_successflux_long_bool()
            self.save_triggers_launches_obs_cancellations(param_df)
            self.save_cf_input(i)
            self.save_scores(i)
            self.save_param_combo_values(param_df, i)
        self.save_score_df()
            
    def append_param_names_and_units(self):
        ''' Using the parameter names and parameter units, make new columns with those values. The parameter name column
        will be empty, and later will be populated with the specific parameter combination.
        '''
        for i, param in enumerate(self.param_names):
            self.score_df[param] = ''
            self.score_df[f'{param}_units'] = [self.param_units[i]]*len(self.launch_df_list)
            
    def make_all_above_successflux_long_bool(self):
        ''' Will make a true/False boolean that defines which of all the flares meet the success flux crieteria, and are 
        at least 20 percent of the peak 20 minutes after the peak time.
        '''
        self.data['min_flux_forlongduration'] = self.data['peak flux']*0.2
        self.data['long duration'] = None
        for i, flare in enumerate(self.data['time']):
            peak_time_indx = np.where(flare == self.data['peak time'][i])[0]
            if len(peak_time_indx)>0:
                if peak_time_indx[0] + 20 >= len(flare):
                    final_flux = 0
                else:
                    final_flux = flare[peak_time_indx[0]+20] #20 minutes later
                self.data['long duration'][i] = final_flux > self.data['min_flux_forlongduration'][i]
            else:
                self.data['long duration'][i] = False
        
        self.all_above_successflux_long = (self.data['peak flux'] >= self.success_flux_value) & (self.data['long duration']==True)
            
    def save_triggers_launches_obs_cancellations(self, param_df):
        '''This saves arrays of true/false for the baseline truth (was the flare above C5), the trigger prediction
        (there was a trigger), the launch prediction (was there a launch) and the launch/obs prediction 
        (launch and obs) and cancellation. From this, we can save the scores needed
        for the confusion matrices. 
        '''
        #make an array that determines whether there was a trigger
        all_flareID = self.data['flare ID']
        flareID_triggers = np.array(param_df['Flare_ID'])
        trigger_indx = []
        for flare in flareID_triggers:
            indx = np.where(flare==all_flareID)[0][0]
            trigger_indx.append(indx)
        self.trigger_array = np.array([False]*len(all_flareID))
        np.put(self.trigger_array, trigger_indx, True)   
        
        #make an array that determines whether there was a launch
        flareID_launches = np.array(param_df['Flare_ID'][param_df['Cancelled?']==False])
        launch_indx = []
        for flare in flareID_launches:
            indx = np.where(flare==all_flareID)[0][0]
            launch_indx.append(indx)
        self.launch_array = np.array([False]*len(all_flareID))
        np.put(self.launch_array, launch_indx, True)
        
        #make an array that defines if both FOXSI and HiC Success criteria were met.
        flareID_success_observations = np.array(param_df['Flare_ID'][(param_df['Cancelled?']==False) & (param_df[f'HiC_Mean_LongDuration']==True) & (param_df[f'Max_FOXSI_{self.success_flux_key}']==True)])
        obs_indx = []
        for flare in flareID_success_observations:
            indx = np.where(flare==all_flareID)[0][0]
            obs_indx.append(indx)
        self.obs_array = np.array([False]*len(all_flareID))
        np.put(self.obs_array, obs_indx, True)
        
        #make an array that defines where the launch was cancelled
        flareID_cancellations = np.array(param_df['Flare_ID'][param_df['Cancelled?']==True])
        canc_indx = []
        for flare in flareID_cancellations:
            indx = np.where(flare==all_flareID)[0][0]
            canc_indx.append(indx)
        self.canc_array = np.array([False]*len(all_flareID))
        np.put(self.canc_array, canc_indx, True)
        
        #make an array that defines where the peak was observed by FOXSI
        flareID_foxsipeak_observations = np.array(param_df['Flare_ID'][(param_df['Cancelled?']==False) & (param_df['FOXSI_Peak_Observed?']==True)])
        foxsipeak_obs_indx = []
        for flare in flareID_foxsipeak_observations:
            indx = np.where(flare==all_flareID)[0][0]
            foxsipeak_obs_indx.append(indx)
        self.foxsipeak_obs_array = np.array([False]*len(all_flareID))
        np.put(self.foxsipeak_obs_array, foxsipeak_obs_indx, True) 
        
        #make an array that defines where the peak was before HiC Obs
        flareID_hicpostpeak_observations = np.array(param_df['Flare_ID'][(param_df['Cancelled?']==False) & (param_df['HiC_Post_Peak_Obs?']==True)])
        hicpostpeak_obs_indx = []
        for flare in flareID_hicpostpeak_observations:
            indx = np.where(flare==all_flareID)[0][0]
            hicpostpeak_obs_indx.append(indx)
        self.hicpostpeak_obs_array = np.array([False]*len(all_flareID))
        np.put(self.hicpostpeak_obs_array, hicpostpeak_obs_indx, True)
        
    def save_cf_input(self, i):
        ''' This is where we save the values for each box of the 4x2 confusion matrix. I'll now save all of this as
        columns in the launch dataframe. More work on the front end, but plotting will be easier!
        '''
        self.score_df.loc[i, 'TN'] = len(np.where((self.all_above_successflux_long==False) & (self.trigger_array==False))[0])
        self.score_df.loc[i, 'TN_canc'] = len(np.where((self.all_above_successflux_long==False) & (self.canc_array==True))[0])
        self.score_df.loc[i, 'TP'] = len(np.where((self.all_above_successflux_long==True) & (self.launch_array==True) & (self.obs_array==True))[0])
        self.score_df.loc[i, f'FP_no{self.success_flux_key}_long'] = len(np.where((self.all_above_successflux_long==False) & (self.launch_array==True) & (self.obs_array==True))[0])
        self.score_df.loc[i, 'FN'] = len(np.where((self.all_above_successflux_long==True) & (self.trigger_array==False))[0])
        self.score_df.loc[i, 'FN_canc'] = len(np.where((self.all_above_successflux_long==True) & (self.canc_array==True))[0])
        self.score_df.loc[i, f'FP_{self.success_flux_key}_short'] = len(np.where((self.all_above_successflux_long==True) & (self.launch_array==True) & (self.obs_array==False))[0])
        self.score_df.loc[i, f'FP_no{self.success_flux_key}_short'] = len(np.where((self.all_above_successflux_long==False) & (self.launch_array==True) & (self.obs_array==False))[0])
        self.score_df.loc[i, 'FOXSI_PeakRatio'] = len(np.where(self.foxsipeak_obs_array==True)[0])/len(np.where(self.launch_array==True)[0])
        self.score_df.loc[i, 'HiC_ObsAfterPeak'] = len(np.where(self.hicpostpeak_obs_array==True)[0])/len(np.where(self.launch_array==True)[0])
        
    def save_scores(self, i):
        ''' Saves the precision, recal and "trigger-to-launch" and gordon scores from the cf input.
        '''
        precision = (self.score_df.loc[i, 'TP'])/(self.score_df.loc[i, 'TP'] + self.score_df.loc[i, f'FP_no{self.success_flux_key}_long'] +  self.score_df.loc[i, f'FP_{self.success_flux_key}_short'] + self.score_df.loc[i, f'FP_no{self.success_flux_key}_short'])
        recall = (self.score_df.loc[i, 'TP'])/(self.score_df.loc[i, 'TP']  + self.score_df.loc[i, 'FN'] + self.score_df.loc[i, 'FN_canc'] + self.score_df.loc[i, f'FP_{self.success_flux_key}_short'])
        ttl_score = (self.score_df.loc[i, 'TP'] + self.score_df.loc[i, f'FP_no{self.success_flux_key}_long'] + 
                        self.score_df.loc[i, f'FP_{self.success_flux_key}_short'] + self.score_df.loc[i, f'FP_no{self.success_flux_key}_short'])/(
                        self.score_df.loc[i, 'TP'] + self.score_df.loc[i, f'FP_no{self.success_flux_key}_long'] +
                        self.score_df.loc[i, f'FP_{self.success_flux_key}_short'] + self.score_df.loc[i, f'FP_no{self.success_flux_key}_short'] +
                        self.score_df.loc[i, 'TN_canc'] + self.score_df.loc[i, 'FN_canc'])
        gordon_score = (self.score_df.loc[i, f'FP_{self.success_flux_key}_short'] + self.score_df.loc[i, f'FP_no{self.success_flux_key}_short'] + self.score_df.loc[i, f'FP_no{self.success_flux_key}_long']) / (self.score_df.loc[i, 'FN_canc']
                         + self.score_df.loc[i, 'FN'])
        fbeta = (1.25*precision*recall)/((0.25*precision)+recall) 
        accuracy = (self.score_df.loc[i, 'TP'] + self.score_df.loc[i, 'TN'] + self.score_df.loc[i, 'TN_canc'])/len(self.data['flare ID'])
        self.score_df.loc[i, 'Fbeta'] = fbeta
        self.score_df.loc[i, 'Precision'] = precision
        self.score_df.loc[i, 'Recall'] = recall
        self.score_df.loc[i, 'LaunchTriggerRatio'] = ttl_score
        self.score_df.loc[i, 'Gordon'] = gordon_score
        self.score_df.loc[i, 'Accuracy'] = accuracy
        
    def save_param_combo_values(self, param_df, i):
        ''' Saves the parameter value for each combination under the correct column name.
        '''
        for param_name in self.param_names:
            param_value = param_df.loc[0, param_name]
            self.score_df.loc[i, param_name] = param_value
            
    def save_score_df(self):
        ''' Saves the score_df as a .csv file, with the tag defining which core is being used.
        '''
        self.score_df.to_csv(os.path.join(self.out_dir, f'parameter_scores{self.tag}_temp.csv'))
        print(f'saved dataframe for core {self.tag}')
        
        
            