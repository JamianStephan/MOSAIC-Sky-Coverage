import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia

def GAIA_query(ra,dec,mag_limit):
    job = Gaia.launch_job_async("SELECT *     FROM gaiadr3.gaia_source     WHERE CONTAINS(POINT(gaiadr3.gaia_source.ra,gaiadr3.gaia_source.dec),CIRCLE("+str(ra)+","+str(dec)+","+str(10/60)+"))=1     AND phot_g_mean_mag IS NOT NULL     AND parallax IS NOT NULL     ORDER BY phot_g_mean_mag ASC;", dump_to_file=False)

    GAIA_data = job.get_results()
    
    ra=[]
    dec=[]
    mag=[]
    
    for i in range(0,len(GAIA_data)):
        if GAIA_data['phot_g_mean_mag'][i]<=mag_limit:
            mag.append(GAIA_data['phot_g_mean_mag'][i])
            ra.append(GAIA_data['ra'][i])
            dec.append(GAIA_data['dec'][i])

    scales=[] #Need scale values for the marker size of each star to illustrate magnitude on a plot

    for i in mag: #Scale is linear with magnitude from {16}, i.e mag {12} star has marker size {2} times larger than a mag {14} star.
        I=(mag_limit-i)
        scales.append(I*10)

    stars_data={}
    stars_data['ra']=ra
    stars_data['dec']=dec
    stars_data['mag']=mag
    stars_data['scales']=scales
    stars_data['mag_limit']=mag_limit
    stars_data['source']="query"

    return stars_data

