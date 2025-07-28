import numpy as np
import matplotlib.pyplot as plt
from non_snow_retrievals import *
from constants import *

### Snow water equivalent 

def phase_from_swe(delta_swe, incidence_angle = np.deg2rad(main_inc_angle), alpha = alpha, wavelength = nisar_wavelength):
    """
    Equation 18 from Leinss et al. (2015). Digital Object Identifier 10.1109/JSTARS.2015.2432031.

    Returns InSAR phase change between two SAR images given a SWE change (SWE), incidence angle, alpha approximation, and wavelength.
    """
    # https://www.researchgate.net/publication/282550011_Snow_Water_Equivalent_of_Dry_Snow_Measured_by_Differential_Interferometry
    # ki equation is directly below equation 14
    ki = 2 * np.pi / wavelength 
    
    return 2 * ki * alpha / 2 * (1.59 + incidence_angle**(5/2)) * delta_swe

def swe_from_phase(delta_phase, incidence_angle = np.deg2rad(main_inc_angle), alpha = alpha, wavelength = nisar_wavelength):
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

def guneriussen_phase_from_depth(delta_d, epsilon, wavelength = nisar_wavelength, inc = np.deg2rad(40)):
    """
    Equation 6 from Guneriussen et al. (2001). Publisher Item Identifier S 0196-2892(01)08843-X

    Direct equation for phase change from snow depth change (delta_d, Z_s in paper), incidence angle, and permittivity.
    """
    return 4*np.pi/wavelength * delta_d * (np.cos(inc) - np.sqrt( epsilon - np.sin(inc)**2))

def calc_veg_permittivity_error(canopy_height, temperature_series, timedelta=12):
    veg_phase = []
    for i, t2 in enumerate(temperature_series):
        try:
            t1 = temperature_series.iloc[i-timedelta]
            e2 = canopy_permittivity(T=t2, h=canopy_height)
            e1 = canopy_permittivity(T=t1, h=canopy_height)
            veg_phase.append(float(vegetation_phase(H=canopy_height, epsilon_1=e1, epsilon_2=e2)))
        except IndexError:
            veg_phase.append(np.nan)

    return [swe_from_phase(phase) for phase in veg_phase]


def calc_dry_atmo_error(pressure_series, timedelta=12):
    dry_atmo_phase = []
    for i, p2 in enumerate(pressure_series):
        try:
            p1 = pressure_series.iloc[i-timedelta]
            delta_P = p2 - p1
            dry_atmo_phase.append(float(dry_atmosphere(delta_P_z_from_zref=delta_P)))
        except IndexError:
            dry_atmo_phase.append(np.nan)

    return [swe_from_phase(phase) for phase in dry_atmo_phase]


def calc_wet_atmo_error(pw_series, timedelta=12):
    wet_atmo_phase = []
    for i, pw2 in enumerate(pw_series):
        try:
            pw1 = pw_series.iloc[i-timedelta]
            delta_pw = pw2 - pw1
            wet_atmo_phase.append(float(precipitable_water(PW=delta_pw)))
        except IndexError:
            wet_atmo_phase.append(np.nan)

    return [swe_from_phase(phase) for phase in wet_atmo_phase]


def calc_ionosphere_error(tec_series, timedelta=12):
    ion_phase = []
    for i, tec2 in enumerate(tec_series):
        try:
            tec1 = tec_series.iloc[i-timedelta]
            delta_tec = tec2 - tec1
            ion_phase.append(float(ionospheric_advance(delta_TEC=delta_tec)))
        except IndexError:
            ion_phase.append(np.nan)

    return [swe_from_phase(phase)*1e16 for phase in ion_phase]


def calc_deformation_error(defo_series, timedelta=12):
    defo_phase = []
    for i, r2 in enumerate(defo_series):
        try:
            r1 = defo_series.iloc[i-timedelta]
            delta_r = r2 - r1
            defo_phase.append(float(land_surface_deformation(delta_R=delta_r)))
        except IndexError:
            defo_phase.append(np.nan)

    return [swe_from_phase(phase) for phase in defo_phase]