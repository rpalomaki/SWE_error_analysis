from pathlib import Path
from urllib.parse import urljoin
import shutil
import os
import subprocess

def get_ionex_filename(year, day, old_format, HHMM = '0000', TYP = 'FIN', TSMP = '02H', CNT = 'GIM', cente = 'esa', zipped = True):
    """
    Generates new and old format ionex filenames

    Old format is before GPS week 2237 (~ November 2022)
    See: https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/atmospheric_products.html#iono
    """
    if old_format:
        ext = '.Z' if zipped else ''
        return f'{cente}g{day:03d}0.{year%100:02d}i{ext}'
    else:
        ext = '.gz' if zipped else ''
        return f'IGS0OPS{TYP}_{year:04d}{day:03d}{HHMM}_01D_{TSMP}_{CNT}.INX{ext}'

def download_ionex(year, day, output_dir, center = 'esa'):
    """
    Downloads IONEX files from CDDIS at NASA.

    Old format is before GPS week 2237 (~ November 2022)
    See: https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/atmospheric_products.html#iono

    center is only used before GPS week 2237.
    """
    if isinstance(output_dir, str): output_dir = Path(output_dir)
    if Path(output_dir.joinpath(get_ionex_filename(year, day, old_format = False, zipped = False))).exists(): return

    old_format = True if year <= 2022 and day <= 330 else False
    
    main_https = 'https://cddis.nasa.gov/archive/gnss/products/ionex/'
    filename = get_ionex_filename(year, day, old_format)
    https_fp = urljoin(main_https, f'{year:04d}/{day:03d}/{filename}')
    os.system(f'wget --auth-no-challenge -P {output_dir} "{https_fp}"')
    subprocess.call(['gzip', '-d', output_dir.joinpath(filename)])

    if old_format:
        old_format_fp = output_dir.joinpath(get_ionex_filename(year, day, old_format, zipped = False))
        new_format_fp = output_dir.joinpath(get_ionex_filename(year, day, old_format = False, zipped = False))
        shutil.move(old_format_fp, new_format_fp)
    
    return output_dir.joinpath(get_ionex_filename(year, day, old_format = False, zipped = False))

for year in range(2016, 2026):
    print(year)
    for day in range(1, 366):
        download_ionex(year, day, output_dir = '/Users/rdcrlzh1/Documents/SWE_error_analysis/local/ionosphere')