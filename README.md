# DESI-MSI mass recalibration

Companion code for the publication:

Inglese, P., Huang, H., Wu, V., Lewis, M.R. and Takats, Z., 2021. 
Mass recalibration for desorption electrospray ionization mass spectrometry 
imaging using endogenous reference ions. _bioRxiv_.

## How to run:

### ROI selection

```
python select_roi.py
```

![gui](./tools/resources/Screenshot%202021-07-15%20110104.png)

The imzML file is loaded using the `File -> Open raw peaks ...` button
(shortcut = `CTRL + O`).
Once the reference image is displayed, the user can start annotating with the
selected label (top right checkbox), by drawing a closed contour using the mouse.  
In case of mistakes during the drawing, the user can delete the current curve by
pressing the `Delete selection` button (B).  
Once the region is drawn, the user
can confirm by pressing the `Add selection` button (A). The image is updated showing
the currently annotated pixels.  
When enough pixels are annotated with both labels, the image can be segmented
pressing the `Process...` button (E).  
The current segmentation mask is saved in a CSV file in the same folder of the `imzML` file by
pressing the `Save...` button (F). All connected regions smaller than the selected
value in `Smallest ROI size:` (D) are assigned to the background.  
The current annotations or segmentation can be reset by pressing the `Reset` button (C).  
The RGB colors of the image can be controlled through the sliders below the image.

### Mass recalibration

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