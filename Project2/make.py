# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 12:09:50 2022

@author: matti
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import gc
import astropy
import astropy.units as u
from astropy.coordinates import SkyCoord

import gammapy
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.irf import load_cta_irfs
from gammapy.datasets import MapDataset
from gammapy.makers import MapDatasetMaker, SafeMaskMaker
from gammapy.data import Observation
from gammapy.modeling.models import PowerLawNormSpectralModel, Models,  FoVBackgroundModel,GaussianSpatialModel, SkyModel, TemplateSpatialModel
from gammapy.astro.darkmatter import (
    DarkMatterAnnihilationSpectralModel)
from gammapy.modeling.models import Models


sky_models = Models.read("gc_models.yaml")

irfs = load_cta_irfs("$CALDB/South_z60_50h/irf_file.fits")

livetime = 180.0 * u.hr
pointing = SkyCoord(0, 0, unit="deg", frame="galactic")

#Now creating the observation using pre-defined pointing direction, livetime and IRFs.
                     
obs = Observation.create(pointing = pointing, livetime=livetime, irfs=irfs)

en_ax = MapAxis.from_bounds(10,                   #Lower energy bound
                            300000,               #Higher energy bound
                            nbin=100,
                            unit="GeV", 
                            name="energy",   #Name of the axis. It seems as though this has to be set to a specific value (such as "energy_true") for some applications, i.e. some functions seem to need to be able to identify the meaning of some axes.
                            interp="lin"          #Set to "lin" for linear scale
                           )


#Now creating the actual geom-object. Per default, this defines a 2D image. If you want more axes, you need to pre-define these, and then pass the to the axes-argument, like I have done with the energy axis here. 

geom = WcsGeom.create(skydir = (0,0),             #The coordinates to which the central position of the skymap corresponds
                      binsz=0.05, 
                      width=(2,2),              #Width (in degrees) of the skymap
                      frame ="galactic", 
                      axes=[en_ax])

empty = MapDataset.create(geom, name="simu")  


maker = MapDatasetMaker(selection=["exposure", "background", "psf", "edisp"])   
#maker = MapDatasetMaker(selection=["exposure", "psf", "edisp"])
#maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg)
n = 200
for i in range(4,10):
    for j in range(n):
        dataset = maker.run(empty, obs)
        dm_model = sky_models[7]
        dm_model.spectral_model.norm.value= 120 + 10*(i)
        dataset.models = dm_model
        mod1 = sky_models[0]
        mod1.spectral_model.norm.value=10
        dataset.models += mod1
        dataset.models +=sky_models[1]
        dataset.fake(j+i*n)
        fig = plt.figure(i, figsize=(2,2))
        dataset.counts.plot_interactive(add_cbar=True, stretch="linear")
        
        plt.close(fig)
        fig.savefig("DL_pics/small_dm/dmpic"+str(j+200*i)+'.png')
        del dataset
        del dm_model, fig
        gc.collect()
        
        
        dataset = maker.run(empty, obs)
        dm_model = sky_models[7]
        dm_model.spectral_model.norm.value=10*(i+2)
        dataset.models = dm_model
        dataset.models += sky_models[0]
        dataset.fake(2000+j+i*n)
        
        fig = plt.figure(i, figsize=(2,2))
        dataset.counts.plot_interactive(add_cbar=True, stretch="linear")
        plt.close(fig)
        fig.savefig("DL_pics/small_dm/dm1pic"+str(2000+j+n*i)+'.png')
        del dataset
        del dm_model, fig
        gc.collect()
        
        dataset = maker.run(empty, obs)
        dm_model = sky_models[7]
        dm_model.spectral_model.norm.value=25*(i+4)
        #print(sky_models)
        dataset.models = dm_model
        dataset.models += sky_models[1]
    
        #dataset = maker_safe_mask.run(dataset, obs)
        dataset.fake(4000+j+i*n)
        
        #maker_safe_mask = SafeMaskMaker(methods=["offset-max"], offset_max=4.0 * u.deg) 
        fig = plt.figure(i+10, figsize =(2,2))
        dataset.counts.plot_interactive(add_cbar=True, stretch="linear")
        plt.close(fig)
        fig.savefig("DL_pics/small_dm/dm1pic"+str(4000+j+n*i)+'.png')
        del dataset
        del dm_model, fig
        gc.collect()
    
    
    

