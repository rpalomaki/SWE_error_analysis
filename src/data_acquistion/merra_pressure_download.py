from pathlib import Path
from pathlib import Path
from urllib.parse import urljoin
import os
import subprocess

output_dir = '/data/pressure/'
# def get_merra_name(year, month, day):
    # return f'MERRA2_401.inst1_2d_int_Nx.{year}{month:02d}{day:02d}.nc4'

def download_merra(year, month, output_dir):
    """
    Downloads MERRA files from gesdisc at NASA.

    See: https://disc.gsfc.nasa.gov/datasets/M2I1NXINT_5.12.4/summary
    https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Download%20Data%20Files%20from%20HTTPS%20Service%20with%20wget
    https://gmao.gsfc.nasa.gov/reanalysis/MERRA-2/data_access/
    """
    if isinstance(output_dir, str): output_dir = Path(output_dir)
    # filename = get_merra_name(year, month, day)
    # if output_dir.joinpath(filename).exists(): return output_dir.joinpath(filename)

    main_https = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXASM.5.12.4/'
    # filename = get_merra_name(year, month, day)
    https_fp = urljoin(main_https, f'{year:04d}/{month:02d}/')
    os.system(f'wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 --content-disposition -P {output_dir} "{https_fp}"')
    # subprocess.call(['gzip', '-d', output_dir.joinpath(filename)])
    
    # return output_dir.joinpath(filename)

# https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXINT.5.12.4/2016/01/MERRA2_100.inst1_2d_int_Nx.20160102.nc4
# https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I1NXINT.5.12.4/2016/01/MERRA2_400.inst1_2d_int_Nx.20160101.nc4

for year in range(2016, 2026):
    print(year)
    for month in range(1, 13):
        # for day in range(1, 32):
        download_merra(year, month, output_dir)

import xarray as xr
from pathlib import Path

fps = sorted(list(Path(output_dir).glob('*.nc4')))

from tqdm import tqdm
ams = []
pms = []
for fp in tqdm(fps):
    ams.append(xr.open_dataset(fp)['PS'].isel(time = 12))
    pms.append(xr.open_dataset(fp)['PS'].isel(time = 0))

ams = xr.concat(ams, 'time')
ams.to_netcdf(output_dir.joinpath('ams.nc'))

pms = xr.concat(pms, 'time')
pms.to_netcdf(output_dir.joinpath('pms.nc'))