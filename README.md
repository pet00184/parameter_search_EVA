# Parameter Search of Historical GOES XRS Data

Statistical study of parameters that may be utilized as triggers for a sounding rocket launch. The parameter search utilizes the specific timing of the FOXSI/HI-C rocket launch timing, and utilizes the FOXSI/HI-C science goals as success criteria. For all combinations of parameters, results as to what FOXSI/HI-C would launch on and observe are simulated for every trigger combination, tested on approx. 10,500 flares. 

To run a parameter search assuming FOXSI launches first, with HI-C launching 1 minute later: perform tests in the `BOTH_PARAMSEARCH` folder.

To run a parameter search assuming only FOXSI or Hi-C are launching: choose either the `FOXSI_PARAMSEARCH` or  `HIC_PARAMSEARCH` folder. 

yay!!

update this down here: 

outline: 
- overview of the repository
- instructions on how to run a parameter search
- overview of how the parameter search works
- what you should end up with


basically tell her to just use the FOXSI repo, and make sure it works!!

notes: 
- get rid of the paper plots folder here
- make a new folder to store the both and hic parameter searches in
- make a different folder for making the GOES historical fits
- download the main science files that you want and remake the parameter stuff on your own, and just put those on the google drive so eva doens't have to do it themselves.
- maybe see if you can just do one big FITS file....???? 

## Overview of the Paramter Search Code
---
The following code runs a statistical study of GOES XRS parameters to be utilized as triggers for a sounding rocket launch. The specific timing of the FOXSI/HiC launches are hard coded into the search, as well as the instruments success criteria. For all combinations of parameters, results as to what FOXSI/HI-C would launch on and observe are simulated for every trigger combination, tested on approx. 10,500 flares. 

Parameter search runs may be performed assuming both rockets are launching and need their success criteria met, and also on independent rockets (assuming they are launching by themselves). **For this work, we should assume that FOXSI-5 will be launching independently**. The parameter searches can be found in the following folders: 

- `BOTH_PARAMSEARCH`: folder that assumes both FOXSI and HiC are launching. This parameter search was utilized for the FOXSI-4 and HiC joint launch in April 2024. 
- `FOXSI_PARAMSEARCH`: folder that assumes only FOXSI is launching. **This is the folder we should use for this work!!**
- `HIC_PARAMSEARCH`: for completion, we also have a folder that assumes only HiC is launching. 

Working with only one rocket will make things a bit easier this time around, since we don't need to be looking at results for two separate rockets with separate goals. This time around, we will only need to focus on what parameters optimize the FOXSI goals of observign a large solar flare as early in the flare as possible.

## How to run the parameter Search
---
### Step #1: Download the GOES Dataset
GOES has a flare list and historical data that is constantly being updated. Every few months, I redownload the data and parse out all the flares since 2017, so that we get more flares to work with. **To speed up the process, the following necessary FITS files have been updated and are shared on Google Drive for you to download**:

- `GOES_XRS_historical.fits`: This is the main 
