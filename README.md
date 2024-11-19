# Parameter Search of Historical GOES XRS Data
---
The following code runs a statistical study of GOES XRS parameters to be utilized as triggers for a sounding rocket launch. The specific timing of the FOXSI/HiC launches are hard coded into the search, as well as the instruments success criteria. For all combinations of parameters, results as to what FOXSI/HI-C would launch on and observe are simulated for every trigger combination, tested on approx. 13,500 flares. 


##Setup: 
---
### Step 1: Download the Repository
The way I like to do this is through running `git clone https://github.com/pet00184/parameter_search_EVA.git` in my terminal. I know you said you have another way to do this, so however is best for you to get it on your local machine works!

### Step 2: Download Python
If you haven't already done this, you can either download python from the internet, or in your terminal run `brew install python`. 

I generally like to use VSCode when working on this stuff, which might be a good application to download if you don't already use it.

### Step 3: Download dependent packages 
I made a requirements.txt file that should download all the python packages you need at once. To do this, run `pip install -r requirements.txt` in your terminal.

If you start trying to run things and get a "module not found" error, I probably missed a package. Do `pip install packagename` to fix that.

### Step 4: Download the GOES XRS .FITS file from Google Drive
Making the FITS file takes a while and can be annoying to do. It also is too big to put on Github. Instead, I suggest downloading it from Google Drive, under the [Eva Parameter Search Work](https://drive.google.com/drive/u/0/folders/1DVOGmY8eDlSSePoL9E8wmtNEeaWN2LjG) file I shared with you. The name of the file is `GOES_XRS_computed_params.fits`. Put this file in the same folder as the github `parameter_search_EVA` folder.

## File Organization
---
Below is a short overview of what is in all the folders. All of the parameter Search is contained in the `GOES_XRS` folder.

- `FOXSI_PARAMSEARCH`: This is where all the parameter search code that we will be using is stored! For your work, we are only going to assume FOXSI is launching. I will go through this folder in more detail in the next section.
- `OTHER_PARAMSEARCHES`: For the April 2024 flare campaign, another sounding rocket Hi-C launched as well. In this folder are parameter searches that only look at Hi-C's success criteria, and look at both FOXSI and Hi-C simultaneously.
- `MAKING_FITS`: This is where the code to assemble the FITS file is located. This may be useful to look at once we look more into getting EVE data ready for a parmeter search, but I would generally ignore it for now.
- `DATA_SUMMARY_PLOTS`: These are some overview plots of the GOES XRS data itself. Could be useful to look at to get a better idea of what the data you are working with looks like!

## Running the Parameter Search
---
To run the parameter search, go into the `FOXSI_PARAMSEARCH` file. 

The first script to run is `run_paramsearch.py`. Within this file, you will find a dictionary with all the potential parameters to test. 
1. To run different parameter combinations, you will need to edit which parameters you are calling within this file. (We can go over this more in person!) 
2. **For now, there is a basic parameter search with only XRSB data setup, so you can run that first**.
3. You should have a lot of things printing out in your terminal when you run this. If you get a "leaked semaphore object" error at the end, you can just ctrl + c. 
	
Once the parameter search has been run, you will end up with a new folder with the following setup: `RESULTS/C5/xrsb`. Within that folder, you should have the following: 
1. `AllParameterCombinations.csv`: This is a .csv file with all the scores for each parameter combination you test. This file should be small for this XRSB example, but will get very large when doing multiple parameter runs!
2. `Launches`: This folder has the specific results of which flares were launched on for every parameter combination. I would generally ignore this.
3. `FOXSI_Plots`: This is where the plots will show up. After running `run_paramsearch.py` you will get Precision-Recall plots, which can be used to see which specific parameter combinations look the best.
	 
If you see a specific parameter combination that you want to get more information on, you can now run the `run_combination_plotting.py` script. 
1. Just like `run_paramsearch.py`, you will need to go into the file itself and make sure you are listing the same parameters.
	
Once combination plotting has been run, a new folder with additional plots for that specific parameter combination will be found under `RESULTS/C5/xrsb`. 

*Once the XRSB example works and you have had a look through the code, feel free to try different parameter combinations. You will end up with a separate folder for every different combination you try.*


	 


