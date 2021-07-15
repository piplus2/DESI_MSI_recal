# DESI-MSI mass recalibration

Companion code for the publication:

Inglese, P., Huang, H., Wu, V., Lewis, M.R. and Takats, Z., 2021. 
Mass recalibration for desorption electrospray ionization mass spectrometry 
imaging using endogenous reference ions. _bioRxiv_.

## How to run:

```
python desi_recalibration.py input output roi [params]

input:  input imzML file
output: input imzML file
roi:    path of sample ROI mask CSV file. If set equal to 'full', the entire  
        image is analyzed.
        
---- params ----

-h, --help:                 show help message
--analyzer {tof, orbitrap}: MS analyzer. It can be either 'tof' or 'orbitrap'
--ion-mode {pos, neg}:      DESI polarization mode. It can be either 'neg' or   
                            'pos'
--search-tol <value>:       search tolerance in ppm. If set to 'auto', default  
                            values are used (TOF=100 ppm, Orbitrap=20 ppm)
--min-coverage <value>:     minimum coverage percentage to retain a candidate  
                            reference mass (default=75)                       
```

The code saves the images of the candidate reference ions in the subfolder of
the output folder, called `runID + '_recal_imgs'`, where `runID` is the name 
of the input imzML file.