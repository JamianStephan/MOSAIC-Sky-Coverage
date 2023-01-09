import numpy as np
import csv
import matplotlib.pyplot as plt
import time

#=====================================================================================================================
class FOV_stars: #Class for the FOV simulation
    def __init__(self,stars_data):
        self.stars_data=stars_data #Dictionary of the stars' data. Stores ra, dec, mag, and "scale" (for graphical plotting)

        #Physical constraints: all are initial/rough values, final TBD later
        self.innerannulus_diameter=7.4 #Diameter of the GLAO NGS pickoff region, in arcmin
        self.innercircle_diameter=5 #Inner diameter of the annulus where the PFS arms can pickoff stars, in arcmin
        self.outerannulus_diameter=10 #Outer diameter of the annulus/FOV, where the PFS arms can pickoff stars, in arcmin

        self.LGS_distance=7.4/2 #Distance of LGS footprints from FOV centre, in arcmin
        self.LGS_radius=1.1 #Radius of LGS footprints, in arcmin

        self.PFSarm_width=0.5 #Distance from PFS star to edge of PFS arm widthways, in arcmin
        self.PFSarm_length=0.5 #Distance from PFS star to end of PFS arm, in arcmin
        self.PFSarm_minangle=45 #Minimum angle between each PFS arms
        self.PFSstars_minmag=15.5 #Minimum magnitude for PFS stars

        self.NGSstars_mindistance=1 #Minimum distance between NGS asterism stars, in arcmin
        self.NGSstars_minmag=17 #Minimum magnitude of stars to use in the NGS asterism
        

            
    #=====================================================================================================================
    #Following function provides co-ord conversions for a sphere's projection onto a flat surface, using an orthographic projection

    #INPUTS:
    #ra and dec: input co-ords to be converted, in radians
    #ra0 and dec0: co-ords of the azimuthal projection's axis in radians
    #OUTPUTS:
    #x and y: corresponding converted co-ords in the projection plane

    def orthographic(self,ra,dec,ra0,dec0): #Orthographic projection plane around ra0 and dec0
        x=np.cos(dec)*np.sin(ra-ra0)
        y=np.cos(dec0)*np.sin(dec)-np.sin(dec0)*np.cos(dec)*np.cos(ra-ra0)
        return x,y
    
#=====================================================================================================================
#Some useful short functions for geometry calculations 
    def distance(self,x,y): #Distance between (0,0) and (x,y):
        d=np.sqrt(x**2+y**2)
        return d  
    
    def line(self,A,B): #Returns gradient (m) and y intercept (c) of a line joining points A and B, form y=m*x+c
        m=(A[1]-B[1])/(A[0]-B[0])
        c=A[1]-m*A[0] 
        return m,c

    def perp_line(self,m,c,C): #Returns gradient and y intercept of a line perpendicular to y=m*x+c going through C
        m_perp=-1/m
        c_perp=C[1]-m_perp*C[0]
        return m_perp,c_perp   

    #=====================================================================================================================
    #Function converts all stars' ra and decs to x and y values on an orthographic flat plane projection

    #INPUTS:
    #ra0_rad and dec0_rad: pointing of the telescope/centre of FOV in radians
    #OUTPUTS:
    #None returned
    #Generates x and y co-ords of stars on the projection plane, stored to self.stars_data ['x'] and ['y']
   
    def convstars(self,ra0_rad,dec0_rad):
        x=[] #x-coords of stars on projection plane
        y=[] #y-coords of stars on projection plane

        for i in range(0,len(self.stars_data['ra'])): #Loop converts every stars' (ra,dec) to (x,y) in the NGSchosen_index projection plane
            xval,yval=self.orthographic(np.radians(self.stars_data['ra'][i]),np.radians(self.stars_data['dec'][i]),ra0_rad,dec0_rad) #Angles need to be in radians
            x.append(xval)
            y.append(yval)

        void,self.onearcmin_xy=self.orthographic(ra0_rad,dec0_rad+np.radians(1/60),ra0_rad,dec0_rad) #Length of 1 arcmin in the x,y projection units: used as a scale for conversion
        self.outerannulus_xyradius=self.outerannulus_diameter/2*self.onearcmin_xy #Using the 1 arcmin length in x,y, converts the outer annulus diameter to x,y units
        self.innerannulus_xyradius=self.innerannulus_diameter/2*self.onearcmin_xy #Using the 1 arcmin length in x,y, converts the inner annulus diameter to x,y units
        self.innercircle_xyradius=self.innercircle_diameter/2*self.onearcmin_xy #Using the 1 arcmin length in x,y, converts the inner circle diameter to x,y units
       
        self.stars_data['x']=x  #x and y positions are stored to the stars_data dictionary
        self.stars_data['y']=y
        return

    #=====================================================================================================================
    #Function identifies all the stars in the FOV and specifically in which region of the focal plane they lie, i.e. in the inner circle or inner/outer annulus

    #INPUTS:
    #x_offset and y_offset: the FOV pointing displacement from the azimuthal projection axis in x,y; default 0 and 0
    #To be used for optimisation where we shift the FOV centre without having to recalculate all x,y values for a new projection axis - saves alot of time as cuts out convstars
    #OUTPUTS:
    #innercircle_index,innerannulus_index, and outerannulus_index: index of the stars within each of these regions
   
    def findstars(self,x_offset=0,y_offset=0):
        innercircle_index= [] #Index of stars inside inner circle
        innerannulus_index = [] #Index of stars inside inner annulus
        outerannulus_index = [] #Index of stars inside outer annulus

        for i in range(0,len(self.stars_data['x'])): #Loop goes through each star and indentifies if they are within the regions
            dist = self.distance(self.stars_data['x'][i]-x_offset,self.stars_data['y'][i]-y_offset) #Distance between star and FOV centre
            if  dist < self.outerannulus_xyradius: #If star is within outer annulus diameter
                if dist < self.innerannulus_xyradius: #If star is within inner annulus diameter
                    if dist < self.innercircle_xyradius: #If star is within inner circle diameter
                        innercircle_index.append(i) #True, true, true gives inner circle stars
                    else: 
                        innerannulus_index.append(i) #True, true, false gives inner annulus stars
                else:
                    outerannulus_index.append(i) #True, false, false gives outer annulus stars
             
        return innercircle_index,innerannulus_index,outerannulus_index

    #=====================================================================================================================    
    #Function picks off the three brightest stars in the outer annulus below mag {15.5} (with angular constraints of {45} degrees)
    #Radial PFS arm co-ords/dimensions are calculated for these PFS stars
    #The other FOV stars are then checked if they are vignetted

    #INPUTS: 
    #None - uses class variables
    #OUTPUTS:
    #PFSstars_index: indexes of stars used as the PFS guide stars, maximum 3 - these are removed from the region indexes
    #PFSvignettedstars_index: indexes of stars vignetted by the PFS arms - these are removed from the region indexes
    #PFSstars_angle: angle of the PFS stars from the +y axis anticlockwise
    #also generates self.PFSarm_corners: list of co-ordinates for the 3 PFS arms' corners, form [[A,B,C,D],[A,B,C,D],[A,B,C,D]]

    #NOTE: This function will break if the PFS star is at exactly 0, 90, 180, or 270 degrees from +y (north), i.e. the ra or dec is exactly the same as the pointing
    #This shouldn't ever happen due to the precision of GAIA -  can be fixed if this does happen. 

    def PFSarms(self):
        PFSarm_xywidth=self.PFSarm_width*self.onearcmin_xy #Distance from PFS star to edge of PFS arm widthways in x,y units
        PFSarm_xylength=self.PFSarm_length*self.onearcmin_xy #Distance from PFS star to end of PFS arm in x,y units

                            #PFS1,2,3 - order A,B,C,D
        self.PFSarm_corners=[[],[],[]] #List of co-ordinates for the PFS arms' corners (see diagram below)
                          #PFS1,2,3 - inside, have AB, AD, and BC [m,c]
        self.PFSarm_lines=[[],[],[]] #List of m and c for PFS arms' boundary lines

        #D-----------A                      B-----------C
        #            |                      |
        #         x  | ---> FOV Centre <--- |  x
        #            |                      |
        #C-----------B                      A-----------D

        PFSoption_index=self.outerannulus_index+self.innerannulus_index #Indexes of all stars in the region the PFS can pick from, aka. the two annuli
        PFSregion_mag=[] #Magnitude of PFS star options

        for i in PFSoption_index.copy(): #Loop retrieves the magnitudes and indexes of all annuli stars with a magnitude below {15.5}
            if self.stars_data['mag'][i]<self.PFSstars_minmag: #Check if the PFS star options are below the set limit
                PFSregion_mag.append(self.stars_data['mag'][i]) #If they are, record their magnitude
            else:
                PFSoption_index.remove(i) #Remove stars that are below the set magnitude from the PFS star options, so only viable stars remain in the list

        if len(PFSregion_mag)==0: #If there are no stars suitable for PFS arms, return blank lists
            return [],[],[]
        
        PFSregion_mag,PFSoption_index = zip(*sorted(zip(PFSregion_mag,PFSoption_index))) #Sorts the indexes and magnitudes in order of magnitudes, brightest first
        PFSoption_index=list(PFSoption_index) #Last sort returns a tuple, but we need it as a list

        PFSstars_angle=[np.radians(-self.PFSarm_minangle-1)] #This list is the angles of PFS stars, anticlockwise from +y (north)
        #The first dummy angle inside is needed for the way I've coded it, allows the check that each PFS star is not within {45} degrees of another
        PFSstars_index=[] #Indexes of stars that are used as PFS guide stars
        
        for i in PFSoption_index: #Ordered by magnitude, so brightest stars are checked first
            #These ifs determine what quadrant the star is in, and then calculates the appropriate angle from +y (north)
            if self.stars_data['x'][i]>0 and self.stars_data['y'][i]>0:
                theta=2*np.pi-np.arctan(self.stars_data['x'][i]/self.stars_data['y'][i])
            elif self.stars_data['x'][i]<0 and self.stars_data['y'][i]>0:
                theta=np.arctan(-self.stars_data['x'][i]/self.stars_data['y'][i])
            elif self.stars_data['x'][i]<0 and self.stars_data['y'][i]<0:
                theta=np.pi-np.arctan(+self.stars_data['x'][i]/self.stars_data['y'][i])
            else: 
                theta=np.pi+np.arctan(-self.stars_data['x'][i]/self.stars_data['y'][i])
            
            #Following if statement determines if the star is not within {45} degrees of a previous PFS star
            if all([abs(theta-previous_angles)>np.radians(self.PFSarm_minangle)
                                        for previous_angles in PFSstars_angle]):

                PFScount=len(PFSstars_index) #Index needed to save the PFS star's details to

                PFSstars_index.append(i) #Adds the star's index to the PFS stars list
                PFSstars_angle.append(theta) #Adds the star's angle to the PFS stars list

                #We want the PFS stars to only appear on the PFS star list so that they are not used for NGS asterisms
                if i in self.outerannulus_index:
                    self.outerannulus_index.remove(i) #Removes the star's index from the outer annulus list
                else:
                    self.innerannulus_index.remove(i) #Removes the star's index from the inner annulus list

                #A and B are the corners of the PFS arm towards the FOV centre (diagram earlier)
                A=(self.stars_data['x'][i]+PFSarm_xylength*np.sin(theta)+PFSarm_xywidth*np.cos(theta),
                   self.stars_data['y'][i]-PFSarm_xylength*np.cos(theta)+PFSarm_xywidth*np.sin(theta))
                B=(self.stars_data['x'][i]+PFSarm_xylength*np.sin(theta)-PFSarm_xywidth*np.cos(theta),
                   self.stars_data['y'][i]-PFSarm_xylength*np.cos(theta)-PFSarm_xywidth*np.sin(theta))

                #Following calculates the gradients and y intercepts of the PFS arms' boundary lines, and then are stored in a list
                AB_m,AB_c=self.line(A,B)
                AD_m,AD_c=self.perp_line(AB_m,AB_c,A)
                BC_m,BC_c=self.perp_line(AB_m,AB_c,B)
                self.PFSarm_lines[PFScount].append([AB_m,AB_c])
                self.PFSarm_lines[PFScount].append([AD_m,AD_c])
                self.PFSarm_lines[PFScount].append([BC_m,BC_c])

                #Following calculates C and D positions, the origins of the PFS arms: they "originate" at the top/bottom of the FOV boundary
                if theta<np.pi/2 and theta>-np.pi/2 or theta>3*np.pi/2 or theta<-3*np.pi/2: #Ifs PFS star in top half, origins are at y=+FOV boundary
                    C=((self.outerannulus_xyradius-BC_c)/BC_m,self.outerannulus_xyradius)
                    D=((self.outerannulus_xyradius-AD_c)/AD_m,self.outerannulus_xyradius) 
                else: #If PFS star in bottom half, origins are at y=-FOV boundary
                    C=((-self.outerannulus_xyradius-BC_c)/BC_m,-self.outerannulus_xyradius)
                    D=((-self.outerannulus_xyradius-AD_c)/AD_m,-self.outerannulus_xyradius)

                #Save all 4 PFS corners
                self.PFSarm_corners[PFScount].append(A)
                self.PFSarm_corners[PFScount].append(B)
                self.PFSarm_corners[PFScount].append(C)
                self.PFSarm_corners[PFScount].append(D)

            if len(PFSstars_index)==3: #If 3 PFS stars are found, stop finding more
                break
                
        PFSvignettedstars_index=[] #Indexes of stars that are vignetted by PFS arms

        #Python "freaks out" when removing an element from a list in a loop when that list is used for loop values
        #So following reproduces the list to prevent the freak out
        innerannulus_index=self.innerannulus_index.copy() 
        innercircle_index=self.innercircle_index.copy() 
        outerannulus_index=self.outerannulus_index.copy() 
        
        #Identifies which stars are vignetted by PFS arms. These stars are removed the circle and annuli lists, and are added to a vignetted list
        for count,i in enumerate(innerannulus_index+innercircle_index+outerannulus_index): #Check each star in the tech and science FOV
            starx=self.stars_data['x'][i]
            stary=self.stars_data['y'][i]
            for o in range(0,PFScount): #Checks each star against each PFS arm
                theta=PFSstars_angle[o+1] #First PFS angle was a dummy value so skip 1 index ahead
                AB=self.PFSarm_lines[o][0]
                AD=self.PFSarm_lines[o][1]
                BC=self.PFSarm_lines[o][2]

                #The orientation of the arms sides depends on the star's angle/quadrant, and this alters the inequality conditions for vignetting
                if np.pi/2>theta>0: 
                    if (stary>AB[0]*starx+AB[1] and stary<AD[0]*starx+AD[1] 
                                                    and stary>BC[0]*starx+BC[1]):
                        PFSvignettedstars_index.append(i) #PFS vignetted stars are added to the vignetted stars list, and removed from the appropriate region list
                        if i in outerannulus_index:
                            self.outerannulus_index.remove(i)
                        elif i in innerannulus_index: 
                            self.innerannulus_index.remove(i)
                        else:
                            self.innercircle_index.remove(i)

                elif np.pi>theta>np.pi/2:
                    if (stary<AB[0]*starx+AB[1] and stary<AD[0]*starx+AD[1]
                                                    and stary>BC[0]*starx+BC[1]):
                        PFSvignettedstars_index.append(i)
                        if i in outerannulus_index:
                            self.outerannulus_index.remove(i)
                        elif i in innerannulus_index: 
                            self.innerannulus_index.remove(i)
                        else:
                            self.innercircle_index.remove(i)

                elif 3*np.pi/2>theta>np.pi:
                    if (stary<AB[0]*starx+AB[1] and stary>AD[0]*starx+AD[1] 
                                                    and stary<BC[0]*starx+BC[1]):
                        PFSvignettedstars_index.append(i)
                        if i in outerannulus_index:
                            self.outerannulus_index.remove(i)
                        elif i in innerannulus_index: 
                            self.innerannulus_index.remove(i)
                        else:
                            self.innercircle_index.remove(i)

                else:
                    if (stary>AB[0]*starx+AB[1] and stary>AD[0]*starx+AD[1]
                                                    and stary<BC[0]*starx+BC[1]):
                        PFSvignettedstars_index.append(i)
                        if i in outerannulus_index:
                            self.outerannulus_index.remove(i)
                        elif i in innerannulus_index: 
                            self.innerannulus_index.remove(i)
                        else:
                            self.innercircle_index.remove(i)
        
        return PFSstars_index,PFSvignettedstars_index,PFSstars_angle   

    #=====================================================================================================================
    #Function identifies which stars in the FOV are obscured by the footprints of the LGS, and notes which PFS stars are inside such footprints

    #INPUTS:
    #LGS_angle: angle of the LGS footprints template, rotating anti-clockwise in radians
    #OUTPUTS:
    #footprintstars_index: indexes of the stars inside the LGS footprints (apart from those also vignetted by PFS arms) - these are removed from the region indexes
    #footprintstars_LGS: which footprint the respective obscured stars lie in
    #footprintPFSstars_index: indexes of the PFS stars that are also inside LGS footprints 
    #LGS_pos: the x and y co-ords of the centre of the LGS footprints, stored in [[LGS1_x,LGS1_y],[..]...] form

    def LGS_footprint(self,LGS_angle):      
        LGS_xyd=self.LGS_distance*self.onearcmin_xy #Distance of LGS footprints from FOV centre in x,y units
        LGS_xyradius=self.LGS_radius*self.onearcmin_xy #Radius of LGS footprint in x,y units
        
        #Following calculates the x and y positions of the 4 LGS footprints using the LGS angle and LGS distance from FOV centre
        LGS_pos=[[LGS_xyd*np.sin(LGS_angle),LGS_xyd*np.cos(LGS_angle)],[LGS_xyd*np.sin(LGS_angle+np.pi/2),LGS_xyd*np.cos(LGS_angle+np.pi/2)],
                [LGS_xyd*np.sin(LGS_angle+np.pi),LGS_xyd*np.cos(LGS_angle+np.pi)],[LGS_xyd*np.sin(LGS_angle+3*np.pi/2),LGS_xyd*np.cos(LGS_angle+3*np.pi/2)]]

        footprintstars_index=[] #Indexes of stars inside the LGS footprints
        footprintstars_LGS=[] #LGS footprint number that each obscured star lies in
        
        #Python "freaks out" when removing an element from a list in a loop when that list is used for loop values
        #So following reproduces the list to prevent the freak out
        innercircle_index=self.innercircle_index.copy()
        innerannulus_index=self.innerannulus_index.copy()
        outerannulus_index=self.outerannulus_index.copy()

        footprintPFSstars_index=[] #Indexes of PFS stars inside footprints. This is a seperate reference list, and those stars are NOT removed from the PFS stars index list

        #Following takes x,y of each FOV star not vignetted by PFS arms and identifies if they are within the equation of a circle for the LGS footprints
        #Stars that are in LGS footprints are removed from the regions lists and are added to the footprints list
        for i in innercircle_index:
            for o in range(0,4):
                if (self.stars_data['x'][i]-LGS_pos[o][0])**2+(self.stars_data['y'][i]-LGS_pos[o][1])**2<LGS_xyradius**2:
                    footprintstars_index.append(i)     
                    footprintstars_LGS.append(o+1) 
                    self.innercircle_index.remove(i)      
        for i in innerannulus_index:
            for o in range(0,4):
                if (self.stars_data['x'][i]-LGS_pos[o][0])**2+(self.stars_data['y'][i]-LGS_pos[o][1])**2<LGS_xyradius**2:
                    footprintstars_index.append(i)
                    footprintstars_LGS.append(o+1)  
                    self.innerannulus_index.remove(i)
        for i in outerannulus_index:
            for o in range(0,4):
                if (self.stars_data['x'][i]-LGS_pos[o][0])**2+(self.stars_data['y'][i]-LGS_pos[o][1])**2<LGS_xyradius**2:
                    footprintstars_index.append(i)
                    footprintstars_LGS.append(o+1)  
                    self.outerannulus_index.remove(i)

        #Following identifies which PFS stars are inside the LGS footprints
        for i in self.PFSstars_index:
            for o in range(0,4):
                if (self.stars_data['x'][i]-LGS_pos[o][0])**2+(self.stars_data['y'][i]-LGS_pos[o][1])**2<LGS_xyradius**2:
                    footprintPFSstars_index.append(i)
                    
        return footprintstars_index, footprintstars_LGS, footprintPFSstars_index, LGS_pos

    #=====================================================================================================================
    #Function identifies the 3 most viable stars for the NGS asterism after LGS footprints, PFS stars, and PFS vignetting are taken into account
    #Calculates the area and barycentre of the corresponding asterism if possible (i.e. 3 are actually usable)
    #Goes for the 3 brightest that are not within {1} arcmin of eachother
    #Will need a new function for all different permutations, i.e. 1 2 4, 1 2 5, 1 2 6,....., 1 3 4,....., 2 3 4, 2 3 5,....
        
    #INPUTS:
    #Override: default = True, if an asterism can't be found with 3 stars due to the distance constraint, is the constraint ignored
    #OUTPUTS: 
    #NGSasterism_area: area of the asterism in arcmin^2, returns a value of 0 if there are less than 3 PFS stars
    #NGSasterism_barycentre: distance of the asterism barycentre to the FOV in arcmin
    #NGSchosen_index: indexes of stars that are chosen for the NGS asterism within constraints
    #NGSfaint_index: stars that are below the NGS limiting magnitude 
    #NGSnonchosen_index: stars that are not chosen for the NGS asterism, but could be used/are viable
 
    def NGSasterism(self,override=True):
        NGSstars_mag=[] #Magnitude of the stars that can be used as NGS
        NGSstars_index=self.innercircle_index+self.innerannulus_index #Indexes of stars in the correct regions/not vignetted that can be used as NGS
        NGSchosen_index=[] #Stars that are chosen for the NGS asterism - i.e. brightest 3 within the distance and magnitude constraints
        NGSfaint_index=[] #Stars that are below the NGS limiting magnitude 
        NGSnonchosen_index=[] #Stars that are not chosen for the NGS asterism, but could be used

        for i in NGSstars_index.copy(): #Retrieves the magnitudes of stars in the correct regions and of the right magnitudes
            if self.stars_data['mag'][i]<self.NGSstars_minmag: #Only retrieve magnitudes that obey the magnitude limit
                NGSstars_mag.append(self.stars_data['mag'][i])
            else:
                NGSstars_index.remove(i) #Stars that are not within the magnitude limit are removed
                NGSfaint_index.append(i) #These are then added to the faint index list

        if len(NGSstars_index)==0: #If there are no suitable NGS stars, return no area and barycentre
            NGSstars_index=[]
            return 0,0,NGSchosen_index,NGSfaint_index,NGSnonchosen_index
        
        NGSstars_mag, NGSstars_index = zip(*sorted(zip(NGSstars_mag,NGSstars_index))) #Sorts the lists by magnitude, so brightest stars are first        
        NGSstars_mag=list(NGSstars_mag) #Need the stars magnitudes in a list, last line returns a tuple
        NGSstars_index=list(NGSstars_index) #Need the the stars indexes in a list, last line returns a tuple
        
        if len(NGSstars_index)>=3: #If there are 3 or more available NGS stars, see if they obey the min distance constraint
            NGSchosen_index.append(NGSstars_index[0]) #Brightest star is of course avaiable for asterism
            for i in range(1,len(NGSstars_index)): #This loop finds the next brightest star that obeys the distance constraint
                if abs(self.distance(self.stars_data['x'][NGSchosen_index[0]]-self.stars_data['x'][NGSstars_index[i]],self.stars_data['y'][NGSchosen_index[0]]-self.stars_data['y'][NGSstars_index[i]]))>self.NGSstars_mindistance*self.onearcmin_xy:
                    NGSchosen_index.append(NGSstars_index[i]) #If found, add to the chosen NGS star list for the asterism
                    break

            for o in range(i,len(NGSstars_index)): #This loops finds the next brightest star that obeys the distance constraints for both the previous asterism stars
                if (abs(self.distance(self.stars_data['x'][NGSchosen_index[0]]-self.stars_data['x'][NGSstars_index[o]],self.stars_data['y'][NGSchosen_index[0]]-self.stars_data['y'][NGSstars_index[o]]))>self.NGSstars_mindistance*self.onearcmin_xy 
                and abs(self.distance(self.stars_data['x'][NGSchosen_index[1]]-self.stars_data['x'][NGSstars_index[o]],self.stars_data['y'][NGSchosen_index[1]]-self.stars_data['y'][NGSstars_index[o]]))>self.NGSstars_mindistance*self.onearcmin_xy):
                    NGSchosen_index.append(NGSstars_index[o]) #If found, add to the chosen NGS star list for the asterism
                    break
            #The result of the last loops is that either 1, 2, or 3 stars have been found which obey the distance constraint

            if len(NGSchosen_index)==3: #If three have been found, these are used for the asterism
                a,b,c=NGSchosen_index[0],NGSchosen_index[1],NGSchosen_index[2] #Three asterism stars are labelled a,b,c
                NGSstars_index.remove(a),NGSstars_index.remove(b),NGSstars_index.remove(c) #These 3 stars are removed from the NGSstars index list
                NGSnonchosen_index=NGSstars_index #This NGSstars index list then contains the non chosen viable stars
                #Area is found using the Shoelace formula
                NGSasterism_area=((self.stars_data['x'][a]*self.stars_data['y'][b]+self.stars_data['x'][b]*self.stars_data['y'][c]+self.stars_data['x'][c]*self.stars_data['y'][a]
                                -self.stars_data['y'][a]*self.stars_data['x'][b]-self.stars_data['y'][b]*self.stars_data['x'][c]-self.stars_data['y'][c]*self.stars_data['x'][a])/(2*self.onearcmin_xy**2))
                #Barycentre x/y is average of x/y co-ords of the 3 points
                NGSasterism_barycentre=[(self.stars_data['x'][a]+self.stars_data['x'][b]+self.stars_data['x'][c])/3,(self.stars_data['y'][a]+self.stars_data['y'][b]+self.stars_data['y'][c])/3]  
                return abs(NGSasterism_area),NGSasterism_barycentre,NGSchosen_index,NGSfaint_index,NGSnonchosen_index
            
            else: #If three haven't been found within the constraint, then either we can override the constraint or accept it and have only a two or one star asterism
                if override==True:
                    print("Minimum NGS distance overridden")
                    a,b,c=NGSstars_index[0], NGSstars_index[1],NGSstars_index[2] #If constraint is overridden, we pick the 3 brightest stars
                    NGSstars_index.remove(a),NGSstars_index.remove(b),NGSstars_index.remove(c)
                    NGSnonchosen_index=NGSstars_index
                    NGSchosen_index=[a,b,c]
                    #Area is found using the Shoelace formula
                    NGSasterism_area=((self.stars_data['x'][a]*self.stars_data['y'][b]+self.stars_data['x'][b]*self.stars_data['y'][c]+self.stars_data['x'][c]*self.stars_data['y'][a]
                                    -self.stars_data['y'][a]*self.stars_data['x'][b]-self.stars_data['y'][b]*self.stars_data['x'][c]-self.stars_data['y'][c]*self.stars_data['x'][a])/(2*self.onearcmin_xy**2))
                    #Barycentre x/y is average of x/y co-ords of the 3 points
                    NGSasterism_barycentre=[(self.stars_data['x'][a]+self.stars_data['x'][b]+self.stars_data['x'][c])/3,(self.stars_data['y'][a]+self.stars_data['y'][b]+self.stars_data['y'][c])/3] 
                    return abs(NGSasterism_area),NGSasterism_barycentre,NGSchosen_index,NGSfaint_index,NGSnonchosen_index

                if override==False: #If we do not override, then use the stars that obey the constraint in a two or one star asterism
                    if len(NGSchosen_index)==2: #If there are 2 available NGS stars within constraints, calculate the barycentre
                        a=NGSchosen_index[0]
                        b=NGSchosen_index[1]
                        NGSstars_index.remove(a)
                        NGSstars_index.remove(b)
                        NGSnonchosen_index=NGSstars_index
                        NGSasterism_barycentre=[(self.stars_data['x'][a]+self.stars_data['x'][b])/2,(self.stars_data['y'][a]+self.stars_data['y'][b])/2]  
                        return 0,NGSasterism_barycentre,NGSchosen_index,NGSfaint_index,NGSnonchosen_index

                    elif len(NGSchosen_index)==1: #If there is 1 available NGS star within constraints, return its' co-ordinates
                        a=NGSchosen_index[0]
                        NGSstars_index.remove(a)
                        NGSnonchosen_index=NGSstars_index
                        NGSasterism_barycentre=[self.stars_data['x'][a],self.stars_data['x'][b]]    
                        return 0, NGSasterism_barycentre,NGSchosen_index,NGSfaint_index,NGSnonchosen_index

        elif len(NGSstars_index)==2: #If there are 2 only available NGS stars, see if they obey the constraint. If they do, use a 2 star asterism
            if abs(self.distance(self.stars_data['x'][NGSstars_index[0]]-self.stars_data['x'][NGSstars_index[1]],self.stars_data['y'][NGSstars_index[0]]-self.stars_data['y'][NGSstars_index[1]]))>self.NGSstars_mindistance*self.onearcmin_xy:
                    a=NGSstars_index[0]
                    b=NGSstars_index[1]
                    NGSchosen_index=[a,b]
                    NGSasterism_barycentre=[(self.stars_data['x'][a]+self.stars_data['x'][b])/2,(self.stars_data['y'][a]+self.stars_data['y'][b])/2]  
                    return 0,NGSasterism_barycentre,NGSchosen_index,NGSfaint_index,NGSnonchosen_index
            else: #If they dont obey the constraint, we can override the constraint and use the 2 star asterism, or obey the constraint and use a one star asterism
                if override==True:
                    a=NGSstars_index[0]
                    b=NGSstars_index[1]
                    NGSchosen_index=[a,b]
                    NGSasterism_barycentre=[(self.stars_data['x'][a]+self.stars_data['x'][b])/2,(self.stars_data['y'][a]+self.stars_data['y'][b])/2]  
                    return 0,NGSasterism_barycentre,NGSchosen_index,NGSfaint_index,NGSnonchosen_index
                if override==False:
                    a=NGSstars_index[0]
                    NGSchosen_index=[a]
                    NGSasterism_barycentre=[self.stars_data['x'][a],self.stars_data['x'][b]]    
                    return 0, NGSasterism_barycentre,NGSchosen_index,NGSfaint_index,NGSnonchosen_index
        
        elif len(NGSstars_index)==1: #If there is 1 available NGS star, return its' co-ordinates
            a=NGSstars_index[0]
            NGSchosen_index=[a]
            NGSasterism_barycentre=[self.stars_data['x'][a],self.stars_data['x'][b]]    
            return 0, NGSasterism_barycentre,NGSchosen_index,NGSfaint_index,NGSnonchosen_index
            
    #=====================================================================================================================  
    #Function indentifies intercepts between the PFS arms and the LGS footprints, and then calculates the overlapping area
    #NOT FINISHED DOCUMENTING
        
    #INPUTS:
    #LGS_pos: the x and y co-ords of the centre of the LGS footprints, stored in [[LGS1_x,LGS1_y],[..]...] form 
    #OUTPUTS: 
    #PFSLGS_overlaps: the area and corresponding LGS that the PFS overlaps, in form ['footprint'] = number and ['area'] = area
    #intercepts: x and y positions of intercepts between PFS arm sides and LGS boundary, form [[PFS1 intercepts],[PFS2 intercepts],[PFS3 intercepts]]

    def PFSLGS_overlap(self,LGS_pos,PFSstars_angle):
        r=self.LGS_radius*self.onearcmin_xy #radius of the LGS footprints in x,y units

                   #PFS1,PFS2,PFS3
        intercepts=[[],[],[]] #Co-ords of the intercepts for the LGS footprint and PFS arm boundaries
                              #PFS1,PFS2,PFS3
        intercepts_footprints=[-1,-1,-1] #Footprints the PFS arms intersect with

        PFSLGS_overlaps={} #Dictionary for the PFSLGS_overlaps
        PFSLGS_overlaps['area']=[] #List of overlap areas
        PFSLGS_overlaps['footprint']=[] #List of overlap footprints
        
        #Following calculates the interecepts of the PFS arms and LGS footprint boundaries
        for i in range(0,len(self.PFSstars_index)): #each PFS star
            for o in range(0,4): #each LGS footprint
                p=LGS_pos[o][0] #LGS x co-ord
                q=LGS_pos[o][1] #LGS y co-ord
                for u in range(0,3): #each PFS arm side : AB, AD, BC
                    m=self.PFSarm_lines[i][u][0] #Gradient of the PFS side
                    c=self.PFSarm_lines[i][u][1] #Intercept of the PFS side

                    #For the line y=mx+c and circle r^2=(x-p)^2+(y-q)^2, sub out y and re-arrange to find 0=ax^2+bx+c_
                    #Those coefficients are below:
                    a=1+m**2
                    b=2*(m*c-m*q-p)
                    c_=p**2-r**2+c**2+q**2-2*c*q
                    
                    root=b**2-4*a*c_ #Determinant of the quadratic
                    
                    if root > 0: #If the det > 0, then there are two possible solutions, a + and - version
                        #Only one of these are the actual intercept of the PFS arms, as the PFS lines stop at the corners. Need to find the right one
                        xintercept_plus=(-b+np.sqrt(root))/(2*a)
                        xintercept_minus=(-b-np.sqrt(root))/(2*a)

                        #To determine which solution is right, we check if the x value is between the left and right corners
                        if u == 0: #Which corner is left and right depends on which side of the arm
                            if PFSstars_angle[i+1]<np.pi/2 or PFSstars_angle[i+1]>3*np.pi/2: #Which corner is left and right also depends on which quadrants we are in
                                if xintercept_plus > self.PFSarm_corners[i][1][0] and xintercept_plus < self.PFSarm_corners[i][0][0]:
                                    yintercept=m*xintercept_plus+c
                                    intercepts[i].append([xintercept_plus,yintercept])
                                    intercepts_footprints[i]=o
                                if xintercept_minus > self.PFSarm_corners[i][1][0] and xintercept_minus < self.PFSarm_corners[i][0][0]:
                                    yintercept=m*xintercept_minus+c
                                    intercepts[i].append([xintercept_minus,yintercept]) 
                                    intercepts_footprints[i]=o                    
                            else:
                                if xintercept_plus < self.PFSarm_corners[i][1][0] and xintercept_plus > self.PFSarm_corners[i][0][0]:
                                    yintercept=m*xintercept_plus+c
                                    intercepts[i].append([xintercept_plus,yintercept])
                                    intercepts_footprints[i]=o
                                if xintercept_minus < self.PFSarm_corners[i][1][0] and xintercept_minus > self.PFSarm_corners[i][0][0]:
                                    yintercept=m*xintercept_minus+c
                                    intercepts[i].append([xintercept_minus,yintercept])
                                    intercepts_footprints[i]=o
                        elif u==1:
                            if PFSstars_angle[i+1] < np.pi:
                                if xintercept_plus > self.PFSarm_corners[i][3][0] and xintercept_plus < self.PFSarm_corners[i][0][0]:
                                    yintercept=m*xintercept_plus+c
                                    intercepts[i].append([xintercept_plus,yintercept])
                                    intercepts_footprints[i]=o
                                if xintercept_minus > self.PFSarm_corners[i][3][0] and xintercept_minus < self.PFSarm_corners[i][0][0]:
                                    yintercept=m*xintercept_minus+c
                                    intercepts[i].append([xintercept_minus,yintercept])    
                                    intercepts_footprints[i]=o
                            else:
                                if xintercept_plus < self.PFSarm_corners[i][3][0] and xintercept_plus > self.PFSarm_corners[i][0][0]:
                                    yintercept=m*xintercept_plus+c
                                    intercepts[i].append([xintercept_plus,yintercept])
                                    intercepts_footprints[i]=o
                                if xintercept_minus < self.PFSarm_corners[i][3][0] and xintercept_minus > self.PFSarm_corners[i][0][0]:
                                    yintercept=m*xintercept_minus+c
                                    intercepts[i].append([xintercept_minus,yintercept])   
                                    intercepts_footprints[i]=o                           
                        else:
                            if PFSstars_angle[i+1] < np.pi:
                                if xintercept_plus > self.PFSarm_corners[i][2][0] and xintercept_plus < self.PFSarm_corners[i][1][0]:
                                    yintercept=m*xintercept_plus+c
                                    intercepts[i].append([xintercept_plus,yintercept])
                                    intercepts_footprints[i]=o
                                if xintercept_minus > self.PFSarm_corners[i][2][0] and xintercept_minus < self.PFSarm_corners[i][1][0]:
                                    yintercept=m*xintercept_minus+c
                                    intercepts[i].append([xintercept_minus,yintercept])    
                                    intercepts_footprints[i]=o
                            else:
                                if xintercept_plus < self.PFSarm_corners[i][2][0] and xintercept_plus > self.PFSarm_corners[i][1][0]:
                                    yintercept=m*xintercept_plus+c
                                    intercepts[i].append([xintercept_plus,yintercept])
                                    intercepts_footprints[i]=o
                                if xintercept_minus < self.PFSarm_corners[i][2][0] and xintercept_minus > self.PFSarm_corners[i][1][0]:
                                    yintercept=m*xintercept_minus+c
                                    intercepts[i].append([xintercept_minus,yintercept])   
                                    intercepts_footprints[i]=o


        LGS_xyradius=self.LGS_radius*self.onearcmin_xy #Radius of the LGS footprint in x,y units

        #Following calculates area of the overlap between the PFS arms and the LGS footprints: there are 5 different scenarios depending on the number of intercepts with/corners inside the footprint.
        for count, PFSintercept in enumerate(intercepts):
            inside_count=0
            if PFSintercept != []:
                A=self.PFSarm_corners[count][0]
                B=self.PFSarm_corners[count][1]
                footprint_no=intercepts_footprints[count]
                O_x,O_y = LGS_pos[footprint_no][0],LGS_pos[footprint_no][1] #x and y co-ords of the LGS

                if (A[0]-O_x)**2+(A[1]-O_y)**2<LGS_xyradius**2: #Identifies whether A is inside LGS footprint
                    inside_count=inside_count+1 #Counter for how many PFS corners are inside LGS footprints
                    E_x,E_y=A[0]-O_x,A[1]-O_y #Relabel A as E           

                if (B[0]-O_x)**2+(B[1]-O_y)**2<LGS_xyradius**2: #Identifies whether B is inside LGS footprint
                    inside_count=inside_count+1 #Counter for how many PFS corners are inside LGS footprints
                    E_x,E_y=B[0]-O_x,B[1]-O_y #Relabel B as E
                
                if len(PFSintercept)==4: #order will be AB intercepts, AD intercepts, BC intercepts
                    I0_x,I0_y=PFSintercept[0][0]-O_x,PFSintercept[0][1]-O_y #x and y co-ords of the first intercept, origin shifted to LGS footprint centre
                    I1_x,I1_y=PFSintercept[1][0]-O_x,PFSintercept[1][1]-O_y #x and y co-ords of the second intercept, origin shifted to LGS footprint centre
                    I2_x,I2_y=PFSintercept[2][0]-O_x,PFSintercept[2][1]-O_y #x and y co-ords of the third intercept, origin shifted to LGS footprint centre
                    I3_x,I3_y=PFSintercept[3][0]-O_x,PFSintercept[3][1]-O_y #x and y co-ords of the fourth intercept, origin shifted to LGS footprint centre

                    if inside_count==0:
                        theta02=abs(np.arccos(np.dot([I0_x,I0_y],[I2_x,I2_y])/(np.sqrt(I0_x**2+I0_y**2)*np.sqrt(I2_x**2+I2_y**2))))
                        theta13=abs(np.arccos(np.dot([I1_x,I1_y],[I3_x,I3_y])/(np.sqrt(I1_x**2+I1_y**2)*np.sqrt(I3_x**2+I3_y**2))))
                        x=[I0_x,I1_x,I3_x,I2_x]
                        y=[I0_y,I1_y,I3_y,I2_y]
                        I1I2I3I4=abs(0.5*(x[0]*y[1]+x[1]*y[2]+x[2]*y[3]+x[3]*y[0]-y[0]*x[1]-y[1]*x[2]-y[2]*x[3]-y[3]*x[0])) #Area of the quadrilateral, using shoelace formula
                        OI0I2_sector=(1/2)*LGS_xyradius**2*theta02
                        OI1I3_sector=(1/2)*LGS_xyradius**2*theta13
                        OI0I2_triangle=abs((I0_x*I2_y-I0_y*I2_x)/2)
                        OI1I3_triangle=abs((I3_x*I1_y-I3_y*I1_x)/2)
                        area=(I1I2I3I4+OI0I2_sector+OI1I3_sector-OI0I2_triangle-OI1I3_triangle)/(self.onearcmin_xy**2)
                        PFSLGS_overlaps['footprint'].append(footprint_no) #Record the LGS footprint number the overlap occurs in
                        PFSLGS_overlaps['area'].append(area) #Record the area of the overlap

                    if inside_count==1:
                        E=[E_x,E_y]
                        I0=[I0_x,I0_y]
                        I1=[I1_x,I1_y]
                        I2=[I2_x,I2_y]
                        I3=[I3_x,I3_y]

                        points=[E,I0,I1,I2,I3]

                        avg_x=(E[0]+I1[0]+I2[0]+I3[0]+I0[0])/5
                        avg_y=(E[1]+I1[1]+I2[1]+I3[1]+I0[1])/5

                        phi = []

                        for i  in points:
                            if i[0]>0 and i[1]>0:
                                theta=2*np.pi-np.arctan((i[0]-avg_x)/(i[1]-avg_y))
                            elif i[0]<0 and i[1]>0:
                                theta=np.arctan(-(i[0]-avg_x)/(i[1]-avg_y))
                            elif i[0]<0 and i[1]<0:
                                theta=np.pi-np.arctan(+(i[0]-avg_x)/(i[1]-avg_y))
                            else: 
                                theta=np.pi+np.arctan(-(i[0]-avg_x)/(i[1]-avg_y))
                            phi.append(theta)
                        
                        phi, points = zip(*sorted(zip(phi,points)))
                        phi = list(phi)
                        points=list(points)

                        E_index  = [index for (index, item) in enumerate(points) if item == E][0]

                        U1=points[E_index-1]
                        U2=points[E_index-2]
                        U3=points[E_index-3]
                        U4=points[E_index-4]

                        thetaU12=abs(np.arccos(np.dot([U1[0],U1[1]],[U2[0],U2[1]])/(np.sqrt(U1[0]**2+U1[1]**2)*np.sqrt(U2[0]**2+U2[1]**2))))
                        thetaU34=abs(np.arccos(np.dot([U3[0],U3[1]],[U4[0],U4[1]])/(np.sqrt(U3[0]**2+U3[1]**2)*np.sqrt(U4[0]**2+U4[1]**2))))     
                        points_area = abs((E[0]*U1[1]+U1[0]*U2[1]+U2[0]*U3[1]+U3[0]*U4[1]+U4[0]*E[1]-E[1]*U1[0]-U1[1]*U2[0]-U2[1]*U3[0]-U3[1]*U4[0]-U4[1]*E[0])/2)
                        OU1U2_sector=(1/2)*LGS_xyradius**2*thetaU12
                        OU3U4_sector=(1/2)*LGS_xyradius**2*thetaU34
                        OU1U2_triangle=abs((U1[0]*U2[1]-U1[1]*U2[0])/2)
                        OU3U4_triangle=abs((U3[0]*U4[1]-U3[1]*U4[0])/2)     

                        area=(points_area+OU1U2_sector+OU3U4_sector-OU1U2_triangle-OU3U4_triangle)/(self.onearcmin_xy**2) 

                        PFSLGS_overlaps['footprint'].append(footprint_no) #Record the LGS footprint number the overlap occurs in
                        PFSLGS_overlaps['area'].append(area) #Record the area of the overlap

                elif len(PFSintercept)==2:
                    I0_x,I0_y=PFSintercept[0][0]-O_x,PFSintercept[0][1]-O_y #x and y co-ords of the first intercept, origin shifted to LGS footprint centre
                    I1_x,I1_y=PFSintercept[1][0]-O_x,PFSintercept[1][1]-O_y #x and y co-ords of the second intercept, origin shifted to LGS footprint centre
                    theta=abs(np.arccos(np.dot([I0_x,I0_y],[I1_x,I1_y])/(np.sqrt(I0_x**2+I0_y**2)*np.sqrt(I1_x**2+I1_y**2))))

                    if inside_count==2:
                        OI0I1_sector=(1/2)*LGS_xyradius**2*theta #Area of the circle sector composed of the intercepts and LGS footprint centre
                        OI0I1_triangle=abs((I0_x*I1_y-I0_y*I1_x)/2) #Area of the triangle of the LGS footprint centre and intercepts
                        x=[A[0]-O_x,B[0]-O_x,I1_x,I0_x] #x co-ords of the quadrilateral composed of the intercept and two PFS corners inside the LGS footprint
                        y=[A[1]-O_y,B[1]-O_y,I1_y,I0_y] #y co-ords of the quadrilateral. Note these have to be in correct order
                        ABI0I1=abs(0.5*(x[0]*y[1]+x[1]*y[2]+x[2]*y[3]+x[3]*y[0]-y[0]*x[1]-y[1]*x[2]-y[2]*x[3]-y[3]*x[0])) #Area of the quadrilateral, using shoelace formula
                        area=(OI0I1_sector-OI0I1_triangle+ABI0I1)/(self.onearcmin_xy**2) #Area of overlap = sector - origin/intercept triangle + quadrilateral area
                        PFSLGS_overlaps['footprint'].append(footprint_no) #Record the LGS footprint number the overlap occurs in
                        PFSLGS_overlaps['area'].append(area) #Record the area of the overlap

                    if inside_count==1:
                        OI0I1_sector=(1/2)*LGS_xyradius**2*theta #Area of the circle sector composed of the intercepts and LGS footprint centre
                        OI0I1_triangle=abs((I0_x*I1_y-I0_y*I1_x)/2) #Area of the triangle of the LGS footprint centre and intercepts
                        EI0I1_triangle=abs((I0_x*I1_y+I1_x*E_y+E_x*I0_y-I0_y*I1_x-I1_y*E_x-E_y*I0_x)/2) #Area of the triangle of the PFS corner inside the LGS footprint and the intercepts
                        area=(OI0I1_sector-OI0I1_triangle+EI0I1_triangle)/(self.onearcmin_xy**2) #Area of overlap = sector - origin/intercept triangle + corner/intercept triangle
                        PFSLGS_overlaps['footprint'].append(footprint_no) #Record the LGS footprint number the overlap occurs in
                        PFSLGS_overlaps['area'].append(area) #Record the area of the overlap

                    if inside_count==0:
                        OI0I1_sector=(1/2)*LGS_xyradius**2*theta #Area of the circle sector composed of the intercepts and LGS footprint centre
                        OI0I1_triangle=abs((I0_x*I1_y-I0_y*I1_x)/2) #Area of the triangle of the LGS footprint centre and intercepts
                        area=(OI0I1_sector-OI0I1_triangle)/(self.onearcmin_xy**2)
                        PFSLGS_overlaps['footprint'].append(footprint_no) #Record the LGS footprint number the overlap occurs in
                        PFSLGS_overlaps['area'].append(area) #Record the area of the overlap
            else:
                PFSLGS_overlaps['footprint'].append(-1) #If there is no overlap, record footprint number overlap as -1
                PFSLGS_overlaps['area'].append(0.00) #If there is no overlap, record area as 0: needed for optimisation/consistency
                    
        return PFSLGS_overlaps,intercepts

    #=====================================================================================================================            
    #Following is heavily WIP:
    #1) Converts GAIA stars to x,y co-ords in a projection of the sphere onto a flat surface given a pointing
    #2) Identifies which stars are inside the MOSAIC FOV
    #3) Identifies which stars are to be used to the PFS guide stars and which stars are vignetted by PFS arms
    #4) Identifies which stars are inside the footprints of the LGS 
    #5) Identifies the three (or 2/1/0) brightest remaining stars within constraints to use for the NGS asterism, and finds their area and barycentre
    #6) Calculates the intercepts and areas of the PFS arm and LGS footprint's overlaps
    #Each step also calculates the time taken, allowing us to analyse the most intense processes for optimising later

    #INPUTS:
    #ra0 and dec0: pointing of the telescope/centre of FOV in ra and dec degrees
    #report: does nothing rn
    #proj: projection to use for sphere onto flat surface; ORTH default, or STER, EQUI, GNOM
    #view: wether to display the simulation as a plot; False default, or True
    #LGS_angle: orientation of the LGS footprint, anticlockwise from +y (north) in degrees

    #OUTPUTS: Graphical display of the simulation, and an assortment of metrics

    def run(self,ra0,dec0,report=False,view=False,LGS_angle=0, override=True):
        self.start=time.time() #Initial time of FOV simulation
        
        ra0_rad=np.radians(ra0) #Turns ra0 into radians
        dec0_rad=np.radians(dec0) #Turns dec0 into radians
        
        #1
        self.convstars(ra0_rad,dec0_rad)
        # print("Conv stars:")
        # timea=time.time()
        # print(-self.start+timea)
        #2
        self.innercircle_index,self.innerannulus_index,self.outerannulus_index=self.findstars()
        # print("Find stars:")
        # timeb=time.time()
        # print(timeb-timea)
        #3
        self.PFSstars_index,self.PFSvignettedstars_index,PFSstars_angle=self.PFSarms() #These star indexes are removed from the region indexes and are exclusive
        # print("PFS arm stars:")
        # timec=time.time()
        # print(timec-timeb)
        #4
        self.footprintstars_index,self.footprintstars_LGS,footprintPFSstars_index,LGS_pos=self.LGS_footprint(np.radians(LGS_angle)) #self.footprintstars_index indexes are removed from region indexes
        # print("LGS footprint stars:")                                                                                             
        # timed=time.time()
        # print(timed-timec)
        #5
        NGSasterism_area,NGSasterism_barycentre,NGSchosen_index,NGSfaint_index,NGSnonchosen_index=self.NGSasterism(override=override) #These 3 star indexes are NOT removed from region indexes, but the 3 are mutually exclusive
        # Why?: Because at this point, all the stars in the inner annulus and circle can be used for the NGS asterism as they are not vignetted, and will their permutations will be needed for analysis
        # print("NGS stars:")                                                                                        
        # timee=time.time()
        # print(timee-timed)
        6
        PFSLGS_overlaps,intercepts=self.PFSLGS_overlap(LGS_pos,PFSstars_angle)
        # print("PFS/LGS Overlap:")
        # timef=time.time()
        # print(timef-timee)      
        # 
        #7 will be a function that checks the LGS footprint pupil's overlap: i.e. if we overlay all 4 footprints and their respective PFS intercepts,
        # do the different PFS arms overlap (so the same part of the pupil is overlapped on multiple LGS footprints)? If they do, make likelihood -inf.

        #also note, instead of magnitudes in final case, need to convert to photons!

    #=====================================================================================================================        
        #WIP area

                                
                                                
                                                    
        
    #=====================================================================================================================            
        period=time.time()-self.start #Provides time taken for the total FOV analysis  
    
        #Following visualises the simulation in a plot
        if view==True:
            LGS_xyradius=self.LGS_radius*self.onearcmin_xy
            
            fig, ax = plt.subplots(figsize=(10,10))

            FOV_techfield = plt.Circle((0,0),self.outerannulus_xyradius, color='gray', alpha=0.4) #Technical field FOV
            FOV_sciencefield = plt.Circle((0,0),self.innerannulus_xyradius, color='orange', alpha=0.4) #Science field FOV
            FOV_PFSboundary = plt.Circle((0,0),self.innercircle_xyradius,color='black',fill=False,alpha=0.4)
            ax.add_patch(FOV_techfield)
            ax.add_patch(FOV_sciencefield)
            ax.add_patch(FOV_PFSboundary)
            
            for i in range(0,4):
                LGS=plt.Circle((LGS_pos[i][0],LGS_pos[i][1]),LGS_xyradius, color='red', alpha=0.2)
                ax.add_patch(LGS)

                  

            for i in self.outerannulus_index:    
                plt.scatter(self.stars_data['x'][i],self.stars_data['y'][i],s=self.stars_data['scales'][i]*2,color='brown')
            for i in self.footprintstars_index:    
                plt.scatter(self.stars_data['x'][i],self.stars_data['y'][i],s=self.stars_data['scales'][i]*2,color='red')
            for i in self.PFSstars_index:
                plt.scatter(self.stars_data['x'][i],self.stars_data['y'][i],s=self.stars_data['scales'][i]*2,color='black')
            for i in self.PFSvignettedstars_index:
                plt.scatter(self.stars_data['x'][i],self.stars_data['y'][i],s=self.stars_data['scales'][i]*2,color='red')
                
            for i in range(0,len(self.PFSstars_index)):
                plt.plot([self.PFSarm_corners[i][0][0],self.PFSarm_corners[i][1][0]],[self.PFSarm_corners[i][0][1],self.PFSarm_corners[i][1][1]],color='red')
                plt.plot([self.PFSarm_corners[i][0][0],self.PFSarm_corners[i][3][0]],[self.PFSarm_corners[i][0][1],self.PFSarm_corners[i][3][1]],color='red')
                plt.plot([self.PFSarm_corners[i][1][0],self.PFSarm_corners[i][2][0]],[self.PFSarm_corners[i][1][1],self.PFSarm_corners[i][2][1]],color='red')
            
            for i in NGSnonchosen_index:
                plt.scatter(self.stars_data['x'][i],self.stars_data['y'][i],color='orange')
            for i in NGSfaint_index:
                plt.scatter(self.stars_data['x'][i],self.stars_data['y'][i],color='grey')
            
            if len(NGSchosen_index)==3:        
                a=NGSchosen_index[0]
                b=NGSchosen_index[1]
                c=NGSchosen_index[2]
                plt.plot([self.stars_data['x'][a],self.stars_data['x'][b]],[self.stars_data['y'][a],self.stars_data['y'][b]],color='blue')
                plt.plot([self.stars_data['x'][b],self.stars_data['x'][c]],[self.stars_data['y'][b],self.stars_data['y'][c]],color='blue')
                plt.plot([self.stars_data['x'][c],self.stars_data['x'][a]],[self.stars_data['y'][c],self.stars_data['y'][a]],color='blue')
                plt.scatter(self.stars_data['x'][a],self.stars_data['y'][a],color='blue')
                plt.scatter(self.stars_data['x'][b],self.stars_data['y'][b],color='blue')
                plt.scatter(self.stars_data['x'][c],self.stars_data['y'][c],color='blue')
                plt.scatter(NGSasterism_barycentre[0],NGSasterism_barycentre[1],marker='x',color='blue')
            elif len(NGSchosen_index)==2:
                a=NGSchosen_index[0]
                b=NGSchosen_index[1]
                plt.scatter(self.stars_data['x'][a],self.stars_data['y'][a],color='blue')
                plt.scatter(self.stars_data['x'][b],self.stars_data['y'][b],color='blue')
            elif len(NGSchosen_index)==1:
                a=NGSchosen_index[0]
                plt.scatter(self.stars_data['x'][b],self.stars_data['y'][b],color='blue')
                                                                                                                                                    
            for i in self.outerannulus_index:
                ax.annotate(round(self.stars_data['mag'][i],1),(self.stars_data['x'][i],self.stars_data['y'][i]),xytext=(5, 5),textcoords="offset pixels")
            for i in self.innerannulus_index:
                ax.annotate(round(self.stars_data['mag'][i],1),(self.stars_data['x'][i],self.stars_data['y'][i]),xytext=(5, 5),textcoords="offset pixels")  
            for i in self.innercircle_index:
                ax.annotate(round(self.stars_data['mag'][i],1),(self.stars_data['x'][i],self.stars_data['y'][i]),xytext=(5, 5),textcoords="offset pixels") 
            for i in self.footprintstars_index:
                ax.annotate(round(self.stars_data['mag'][i],1),(self.stars_data['x'][i],self.stars_data['y'][i]),xytext=(5, 5),textcoords="offset pixels")  
            for i in self.PFSstars_index:
                ax.annotate(round(self.stars_data['mag'][i],1),(self.stars_data['x'][i],self.stars_data['y'][i]),xytext=(5, 5),textcoords="offset pixels")  
            for i in self.PFSvignettedstars_index:
                ax.annotate(round(self.stars_data['mag'][i],1),(self.stars_data['x'][i],self.stars_data['y'][i]),xytext=(5, 5),textcoords="offset pixels")  
                
            for i in range(0,4):
                ax.annotate("LGS {}".format(i+1),(LGS_pos[i][0],LGS_pos[i][1]))
        
            plt.xlim(-self.outerannulus_xyradius,self.outerannulus_xyradius)
            plt.ylim(-self.outerannulus_xyradius,self.outerannulus_xyradius)
            
            plt.gca().invert_xaxis() #increasing RA is left
   
            ticks = np.linspace(-self.outerannulus_xyradius,self.outerannulus_xyradius,21)
            ticks_labels = np.linspace(-5,5,21)
            
            for i in intercepts:
                for o in i:
                    plt.scatter(o[0],o[1],color='black',marker='x')

            plt.yticks(ticks)
            plt.xticks(ticks)

            ax.set_xticklabels(ticks_labels)
            ax.set_yticklabels(ticks_labels)

            plt.ylabel("arcmin")
            plt.xlabel("arcmin")

            plt.title("Pointing: ({},{}), Projection Angle: {}, Magnitude Limit: {}".format(ra0,dec0,LGS_angle,self.stars_data['mag_limit'])) 

            plt.scatter(0,0,marker='+',color='black')   


        print("PFS stars available = " +str(len(self.PFSstars_index)))
        print("PFS stars in LGS footprints = " +str(len(footprintPFSstars_index))) 

        if len(NGSchosen_index)==3:
            plt.scatter(self.stars_data['x'][b],self.stars_data['y'][b],color='blue')
            plt.scatter(self.stars_data['x'][c],self.stars_data['y'][c],color='blue')
            print("NGS asterism area = {:.2} arcmin^2".format(NGSasterism_area))
            print("NGS barycentre distance from centre = {:.2} arcmin ".format(self.distance(NGSasterism_barycentre[0],NGSasterism_barycentre[1])/self.onearcmin_xy))
        elif len(NGSchosen_index)==2:
            print("NGS asterism area = {} arcmin^2".format(NGSasterism_area))
            print("NGS barycentre distance from centre = {:.2} arcmin ".format(self.distance(NGSasterism_barycentre[0],NGSasterism_barycentre[1])/self.onearcmin_xy))
        elif len(NGSchosen_index)==1:
            print("NGS asterism area = {} arcmin^2".format(NGSasterism_area))
            print("NGS barycentre distance from centre = {} arcmin ".format(self.distance(NGSasterism_barycentre[0],NGSasterism_barycentre[1])/self.onearcmin_xy))
        else: 
            print("No stars available for NGS")
        for i in range(0,len(PFSLGS_overlaps['footprint'])):
            print("Overlap of PFS arm and footprint {}: {:.2} arcmin^2".format(PFSLGS_overlaps['footprint'][i]+1,PFSLGS_overlaps['area'][i]))
        print("")
        print("Time taken = {:.2}s".format(period))

                    
        

