import numpy as np
import matplotlib.pyplot as plt
from non_snow_retrievals import phase_from_soil_moisture

### Snow water equivalent 

def phase_from_swe(delta_swe, incidence_angle = np.deg2rad(main_inc_angle), alpha = alpha, wavelength = Lambda):
    """
    Equation 18 from Leinss et al. (2015). Digital Object Identifier 10.1109/JSTARS.2015.2432031.

    Returns InSAR phase change between two SAR images given a SWE change (SWE), incidence angle, alpha approximation, and wavelength.
    """
    # https://www.researchgate.net/publication/282550011_Snow_Water_Equivalent_of_Dry_Snow_Measured_by_Differential_Interferometry
    # ki equation is directly below equation 14
    ki = 2 * np.pi / wavelength 
    
    return 2 * ki * alpha / 2 * (1.59 + incidence_angle**(5/2)) * delta_swe

def swe_from_phase(delta_phase, incidence_angle = np.deg2rad(main_inc_angle), alpha = alpha, wavelength = Lambda):
    """
    Equation 18 from Leinss et al. (2015) rearranged to give swe from phase. Digital Object Identifier 10.1109/JSTARS.2015.2432031.

    Returns SWE change (SWE) given InSAR phase change ebtween two SAR images, incidence angle, alpha approximation, and wavelength.
    """
    # https://www.researchgate.net/publication/282550011_Snow_Water_Equivalent_of_Dry_Snow_Measured_by_Differential_Interferometry
    # ki equation is directly below equation 14
    ki = 2 * np.pi / wavelength 
    # simplified the 2 * ki / 2 to just ki
    
    return 2 * delta_phase / ( 2 * ki * alpha * (1.59 + incidence_angle**(5/2)))

def calc_soil_moisture_error(sand, clay, sm_series, timedelta=12):
    soil_phase = []
    for i, sm2 in enumerate(sm_series):
        try:
            sm1 = sm_series.iloc[i-timedelta]
            soil_phase.append(float(phase_from_soil_moisture(sand=sand, clay=clay, sm1=sm1, sm2=sm2)))
        except IndexError:
            soil_phase.append(np.nan)

    return [swe_from_phase(phase) for phase in soil_phase]

def guneriussen_phase_from_depth(delta_d, epsilon, wavelength = Lambda, inc = np.deg2rad(40)):
    """
    Equation 6 from Guneriussen et al. (2001). Publisher Item Identifier S 0196-2892(01)08843-X

    Direct equation for phase change from snow depth change (delta_d, Z_s in paper), incidence angle, and permittivity.
    """
    return 4*np.pi/wavelength * delta_d * (np.cos(inc) - np.sqrt( epsilon - np.sin(inc)**2))
