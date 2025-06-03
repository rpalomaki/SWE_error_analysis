import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
import cartopy.crs as ccrs
import wget
import subprocess

# Larger figure size
fig_size = [14, 10]
plt.rcParams['figure.figsize'] = fig_size

# https://github.com/daniestevez/jupyter_notebooks/blob/master/IONEX.ipynb
def parse_map(tecmap, exponent = -1):
    tecmap = re.split('.*END OF TEC MAP', tecmap)[0]
    return np.stack([np.fromstring(l, sep=' ') for l in re.split('.*LAT/LON1/LON2/DLON/H\\n',tecmap)[1:]])*10**exponent
    
def get_tecmaps(filename):
    with open(filename) as f:
        ionex = f.read()
        return [parse_map(t) for t in ionex.split('START OF TEC MAP')[1:]]

def get_tec(tecmap, lat, lon):
    i = round((87.5 - lat)*(tecmap.shape[0]-1)/(2*87.5))
    j = round((180 + lon)*(tecmap.shape[1]-1)/360)
    return tecmap[i,j]

def ionex_filename(year, day, HHMM = '0000', TYP = 'FIN', TSMP = '02H', CNT = 'GIM', zipped = True):
    # TSMP could be '02H' for every 2 hours or '01D' for every day
    # TYP could be 'FIN' for final or 'RAP' for rapid solution
    # CNT could be GIM for global ionospheric TEC maps or ROT (rate of TEC index maps)
    if zipped == True: ext = '.gz'
    else: ext = ''
    return f'IGS0OPS{TYP}_{year:04d}{day:03d}{HHMM}_01D_{TSMP}_{CNT}.INX{ext}'
    # return '{}g{:03d}0.{:02d}i{}'.format(centre, day, year % 100, '.Z' if zipped else '')

def ionex_ftp_path(year, day):
    return 'https://cddis.nasa.gov/archive/gnss/products/ionex/{:04d}/{:03d}/{}'.format(year, day, ionex_filename(year, day))
    return 'ftp://cddis.gsfc.nasa.gov/gnss/products/ionex/{:04d}/{:03d}/{}'.format(year, day, ionex_filename(year, day, centre))

def ionex_local_path(year, day, directory = '/tmp', zipped = False):
    return directory + '/' + ionex_filename(year, day, zipped=zipped)

import os
def download_ionex(year, day, centre = 'esa', output_dir = '/tmp'):
    os.system(f'wget --auth-no-challenge -P {output_dir} "{ionex_ftp_path(year, day)}"')
    # wget.download(ionex_ftp_path(year, day, centre), output_dir)
    subprocess.call(['gzip', '-d', ionex_local_path(year, day, output_dir, zipped = True)])
    
def plot_tec_map(tecmap):
    proj = ccrs.PlateCarree()
    f, ax = plt.subplots(1, 1, subplot_kw=dict(projection=proj))
    ax.coastlines()
    h = plt.imshow(tecmap, cmap='viridis', vmin=0, vmax=100, extent = (-180, 180, -87.5, 87.5), transform=proj)
    plt.title('VTEC map')
    divider = make_axes_locatable(ax)
    ax_cb = divider.new_horizontal(size='5%', pad=0.1, axes_class=plt.Axes)
    f.add_axes(ax_cb)
    cb = plt.colorbar(h, cax=ax_cb)
    plt.rc('text', usetex=True)
    cb.set_label('TECU ($10^{16} \\mathrm{el}/\\mathrm{m}^2$)')

from tqdm import tqdm
for year in range(2016, 2026):
    print(year)
    for day in tqdm(range(1, 366)):
        download_ionex(year, day, output_dir='/Users/rdcrlzh1/Documents/thp/data/ionosphere')