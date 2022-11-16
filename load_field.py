
from astropy.io.votable import parse_single_table

def load_field(field_name, mag_limit): #Loads the stars near GOODS South within 100 arcmin; downloaded using ASQL from GAIA DR3
    table=parse_single_table(field_name)
    CANDLES=table.array

    ra=[] #Right Ascension of stars
    dec=[] #Declination
    mag=[] #Magnitude
    for i in CANDLES[:]: #Converts all mag, ra, and dec into floats (from strings)
        if i[2]!='': #Some observations do not have a magnitude; so these are ignored
            if float(i[69])<=mag_limit: #Faint stars of mag {16} and above are ignored
                mag.append(float(i[69])) 
                ra.append(float(i[5])) 
                dec.append(float(i[7]))
                
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
    stars_data['source']="field"

    return stars_data       