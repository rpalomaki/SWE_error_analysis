import numpy as np
import matplotlib.pyplot as plt
from constants import *

# Line of sight deformation

def land_surface_deformation(delta_R, wavelength = nisar_wavelength):
    """
    Phase change as a result of line of sight deformation (delta_R)
    """
    return 4 * np.pi / wavelength * delta_R

# Ionosphere

def ionospheric_advance(delta_TEC, K = 40.28, wavelength = nisar_wavelength, c = 3e8):
    """
    Phase change as a result of ionospheric Total Electron Count changes (delta_TEC), wavelength, and speed of light. 
    
    Equation 2 of Rosen et al. (2010). DOI: 10.1109/RADAR.2010.5494385
    """
    return 4 * np.pi * K * wavelength * delta_TEC / (c**2)

# Atmosphere

def dry_atmosphere(delta_P_z_from_zref, Rd = 287.05, g = 9.81, k1 = 0.776, incidence_angle = np.deg2rad(main_inc_angle), wavelength = nisar_wavelength):
    return (4 * np.pi / wavelength) * 1e-6 * k1 * Rd / (np.cos(incidence_angle) * g) * delta_P_z_from_zref

def precipitable_water(PW, incidence_angle = np.deg2rad(main_inc_angle), wavelength = nisar_wavelength):
    """
    The approximate phase delay as a function of precipitable water vapor in atmosphere.
    """
    return 4 * np.pi / wavelength * 6.5 * PW / np.cos(incidence_angle)

# Soil Moisture

def real_part_1_4GHz(S, C, mv, a0 = 2.862, a1 = -0.012, a2 = 0.001, b0 = 3.803, b1 = 0.462, b2 = -0.341, c0 = 119.006, c1 = -0.500, c2 = 0.633):
    """
    Imaginary polynomial of Hallikainen et al. (1985)'s Table II (row 1)

    Parameters are from row 1 of table II of Hallikainen et al. (1985) at 1.4 GHz
    and Equation 14 of of Hallikainen et al. (1985)
    DOI: 10.1109/TGRS.1985.289497

    Gives real component of permittivity based on volumetric moisture (mv), sand content (S), and clay content (C).
    S and C are dry soil percentage by weight with remaining fraction assumed to consist of silt. 
    """
    
    # Parameters are from row 1 of table II of Hallikainen et al. (1985) at 1.4 GHz
    # and Equation 14 of of Hallikainen et al. (1985)
    e_real = (a0 + a1*S + a2*C) + (b0 + b1*S + b2*C) * mv + (c0 + c1*S + c2*C) * mv**2
    return e_real

def imaginary_part_1_4GHz(S, C, mv, a0 = 0.356, a1 = -0.003, a2 = -0.008, b0 = 5.507, b1 = 0.044, b2 = -0.002, c0 = 17.753, c1 = -0.313, c2 = 0.206):
    """
    Imaginary polynomial of Hallikainen et al. (1985)'s Table II (row 11)

    Parameters are from row 1 of table II of Hallikainen et al. (1985) at 1.4 GHz
    and Equation 14 of of Hallikainen et al. (1985)
    DOI: 10.1109/TGRS.1985.289497

    Gives imaginary component of permittivity based on volumetric moisture (mv), sand content (S), and clay content (C).
    S and C are dry soil percentage by weight with remaining fraction assumed to consist of silt. 
    """
    e_imaj = (a0 + a1*S + a2*C) + (b0 + b1*S + b2*C) * mv + (c0 + c1*S + c2*C) * mv**2
    return e_imaj

def complex_e_1_4GHz(mv, S, C):
    """
    Combines real and imaginary permittivities calculated by table II in Hallikainen et al. (1985)
    DOI: 10.1109/TGRS.1985.289497

    Gives complex permittivity based on volumetric water content (mv), sand content (S), and clay content (C).
    S and C are dry soil percentage by weight with remaining fraction assumed to consist of silt. 
    """
    # subtract imaginary part to get realistic values
    return real_part_1_4GHz(S, C, mv) - 1j * imaginary_part_1_4GHz(S, C, mv)

def soil_vertical_wave_number_from_permittivity(e_prime, wavelength = nisar_wavelength, magnetic_permeability = 1.0, incidence_angle = np.deg2rad(main_inc_angle)):
    """
    Equation 5 from De Zan et al. (2014)
    Digital Object Identifier 10.1109/TGRS.2013.2241069

    Gives the soil perpendicular (z) wavenumber given a soils dielectric permittivity.
    """
    c = 3e8
    f = c / wavelength
    w = 2 * np.pi * f

    kx = (2 * np.pi / wavelength) * np.sin(incidence_angle)

    kz = np.sqrt(w**2 * e_prime * magnetic_permeability - kx**2)
    return kz

def phase_soil_wave_number(e1_prime, e2_prime, magnetic_permeability = 1.0):
    """
    Equation 12 from De Zan et al. (2014)
    Digital Object Identifier 10.1109/TGRS.2013.2241069

    Gives the phase change based on two soil dielectric permittivities
    """
    kz1= soil_vertical_wave_number_from_permittivity(e1_prime)
    kz2 = soil_vertical_wave_number_from_permittivity(e2_prime)
    return 1/(2*1j * kz1 - 2* 1j * np.conj(kz2))

def phase_from_soil_moisture(sand, clay, sm1, sm2):
    return np.angle(phase_soil_wave_number(e1_prime = complex_e_1_4GHz(S = sand, C = clay, mv = sm1), e2_prime=complex_e_1_4GHz(S = sand, C = clay, mv = sm2)))

# Vegetation Freezing

def v_h20(T, T_melt = 2):
    """
    Exponential change in ice to liquid water content based on temperature and a fitted parameter (T_melt)
    
    Equation 22 from Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542

    Parameter from Table 1 of Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542 

    """
    if T > 0: return 1
    else: return np.exp(T/T_melt)

def e_ice(T, wavelength = nisar_wavelength):
    """
    Ice's permittivity as a function of wavelength and temperature. 

    Real part: Equation 20 of Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542

    Imaginary from:
    "Notes on microwave radiation from snow samples and emission of layered snowpacks" Matzler (2004)
    Equations # 2, 3, 4 from Section 3 Dielectric Properties of water ice: imaginary part
    """
    # frequency in GHz
    f = 3e8/wavelength/1e9

    T = T + 273.16
    real = 3.1184 + 9.1e-4* (T - 273)
    
    omega = 300 / T - 1
    alpha = 0.00504 + 0.0062 * omega * np.exp(-22.1*omega)
    beta = (0.0207 / T) * (np.exp(335/T) / (np.exp(335/T) - 1)**2) + 1.16e-11 * f**2 + np.exp(-9.963 + 0.0372 * (T - 273.16))
    imaj = alpha/f + beta*f

    return real - 1j*imaj

def e_h20(T, S = 0, wavelength = nisar_wavelength):
    """
    Water's permitivitty as a function of wavelength, temperature, and salinity (S)

    From Klein et al. (1997)'s Equation 5, 8, 17.
    simplified by assumption of no salinity and low frequency
    """
    f = 3e8 / nisar_wavelength
    f = 1.4e9
    # e infinite for distlled water (paragraph below equation 7)
    e_inf = 4.9 # 20% error
    # standard debye equation - sufficent for low frequencies 
    alpha = 0
    
    e0 = 8.854e-12
    # angular frequnecy
    w = 2*np.pi*f
    # Equation 8
    e_s = 88.045 - 0.4147 * T + 6.295e-4 * T**2 + 1.075e-5 * T**3
    # distilled water so sigma = 0. Otherwise equation 9, 10, 11, 12 based on salinity
    sigma = 0
    # equation 17
    tau = 1.768e-11 - 6.086e-13 * T + 1.104e-14*T**2 - 8.111e-17*T**3
    # Equation 5
    e = e_inf + (e_s - e_inf) / (1 + (1j * w * tau)**(1-alpha)) - 1j * sigma / (w * e0)
    return e

def e_water_ice(T):
    """
    Mixing of water and ice permitivities based temperature (giving volume liquid to ice)

    Equation 19 from Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542
    """
    volume_frac_h20 = v_h20(T)
    e_water_part = e_h20(T)
    e_ice_part = e_ice(T)
    return volume_frac_h20*e_water_part + (1 - volume_frac_h20) * e_ice_part


def v_h20_in_wood(WC_wood = 0.3, p_wooddry = 300, p_water = 1000):
    """
    Calculates the volume fraction of water relative to wood.

    Uses WC_wood (fresh-wood gravimetric liquid water content), densities of dry wood and water

    Equation 18 from Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542

    Parameter from Table 1 of Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542
    """
    return WC_wood * p_wooddry / p_water

def e_wood(T, por = 0.5, WC_wood = 0.3, p_wooddry = 300, e_woodcells = 5.0 - 0.5 * 1j, e_air = 1.0006):
    """
    Calculates permittivity of wood given the temperature, wood liquid water content, permittivity of wood, porosity of wood.

    Equation 17 from Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542

    Parameter from Table 1 of Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542 
    """
    vh20 = v_h20_in_wood(WC_wood, p_wooddry)
    eh20 = e_water_ice(T)
    ewood = vh20 * eh20 + (1 - por) * e_woodcells + (por - vh20) * e_air
    return ewood

def vscc(CM_cdry = 10, h_c = 10, m_scc = 0.3, p_wooddry = 300):
    """
    Volume fraction of small canopy constituents. Uses p_wooddry (density of dry wood), 
    m_scc (gravimetric fraction of small canopy constituents), h_c (canopy height),
    CM_cdry (Column-mass of dry wood)

    Equation 16 from Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542

    Parameter from Table 1 of Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542 
    """
    # CM_cdry is based on 10m canopy height but if we change canopy height the
    # mass of dry wood changes proportionally (they are both 10 so those cancel.)
    table1_hc= 10
    table1_cm_cdry = 10
    if h_c != 10: CM_cdry = table1_cm_cdry*h_c/table1_hc

    return CM_cdry * m_scc / (h_c * p_wooddry)

def canopy_permittivity(T, WC_wood = 0.3, e_air = 1.0006, h = 10):
    """
    Calculate effective canopy permittivity from WC_wood (fresh-Wood gravimetric liquid Water-Content), 
    T (temperature), e_air (air permittivity), and canopy height.

    Equation 12 from Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542

    Parameter from Table 1 of Schwank et al. (2015)
    https://doi.org/10.1016/j.rse.2021.112542 
    """
    v_scc = vscc(h_c=h)
    ewood = e_wood(T, WC_wood = WC_wood)
    return e_air + ((ewood - e_air) * (ewood+ 5*e_air) * v_scc) / (3 * (ewood + e_air) - 2 * (ewood-e_air)  * v_scc)


def vegetation_phase(H, epsilon_1, epsilon_2, wavelength = nisar_wavelength, inc = np.deg2rad(main_inc_angle)):
    """
    Equation 6 from Guneriussen et al. (2001). Publisher Item Identifier S 0196-2892(01)08843-X

    Direct equation for phase change from snow depth change (delta_d, Z_s in paper), incidence angle, and permittivity.
    """
    return 4*np.pi/wavelength * H * (np.sqrt( epsilon_2 - np.sin(inc)**2) - np.sqrt( epsilon_1 - np.sin(inc)**2))
