import pandas as pd
import geopandas as gpd
import numpy as np
from glob import glob
from pathlib import Path
import re
import warnings
import matplotlib.pyplot as plt
import rasterio as rio
import xarray as xr
import rioxarray as rxr
import sys
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
from matplotlib import patches
import matplotlib.ticker as mtick
sys.path.append('src')
from src.swe_retrievals import *
from src.plotting_functions import *

warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)
from pandas.errors import SettingWithCopyWarning
warnings.simplefilter('ignore', SettingWithCopyWarning)
warnings.simplefilter('ignore', UserWarning)


def compile_timeseries(station_id, time='am'):
    sites = pd.read_csv('/pl/active/palomaki-sar/insar_swe_errors/data/snotel/fig4_sites.csv')
    site = sites.loc[sites['station_id']==station_id]
    df = pd.read_csv(f'/pl/active/palomaki-sar/insar_swe_errors/data/snotel/hourly/{station_id}_hourly_2016_2025.csv',
                    index_col=0, parse_dates=True)
    df.index = df.index.tz_convert(site['timezone'].values[0]).tz_localize(None)
    tz_adjust = 6 #if site['timezone'].values[0] == 'America/Denver' else 7
    
    
    if time == 'am':
        df_out = pd.DataFrame(df.loc[df.index.hour==6, 'SOIL MOISTURE -2IN'])
    elif time == 'pm':
        df_out = pd.DataFrame(df.loc[df.index.hour==18, 'SOIL MOISTURE -2IN'])
        
    for i in df_out.index:
        if pd.isnull(df_out.loc[i, 'SOIL MOISTURE -2IN']):
            try:
                if pd.notnull(df.loc[i-pd.to_timedelta('1h'), 'SOIL MOISTURE -2IN']):
                    df_out.loc[i, 'SOIL MOISTURE -2IN'] = df.loc[i-pd.to_timedelta('1h'), 'SOIL MOISTURE -2IN']
            except KeyError:
                pass
        if pd.isnull(df_out.loc[i, 'SOIL MOISTURE -2IN']):
            try:
                if pd.notnull(df.loc[i+pd.to_timedelta('1h'), 'SOIL MOISTURE -2IN']):
                    df_out.loc[i, 'SOIL MOISTURE -2IN'] = df.loc[i+pd.to_timedelta('1h'), 'SOIL MOISTURE -2IN']
            except KeyError:
                pass
        
    # Keep temp, swe, SM columns from snotel data
    if df.iloc[0]['AIR TEMP_units'] == 'degF':
        df_out['airtemp_C'] = (df.loc[df_out.index, 'AIR TEMP'] - 32)*5/9
    else:
        df_out['airtemp_C'] = df.loc[df_out.index, 'AIR TEMP']
    
    df_out['swe_m'] = df.loc[df_out.index, 'SWE'] * 0.0254
    df_out.rename(columns={'SOIL MOISTURE -2IN':'soil_moisture_pct'}, inplace=True)
    
    
    # Dry atmosphere
    dry_atmo = xr.open_dataset(f'/pl/active/palomaki-sar/insar_swe_errors/data/atmo_dry/pressure_{time}.nc').sel(lat=site['lat'].values[0], lon=site['lon'].values[0], method='nearest')
    dry_atmo['time'] = dry_atmo['time'] - pd.to_timedelta(tz_adjust, 'h')
    df_out['surf_pres'] = dry_atmo.to_dataframe()['PS']
    
    # Wet atmosphere
    wet_atmo = xr.open_dataset(f'/pl/active/palomaki-sar/insar_swe_errors/data/atmo_wet/pw_{time}.nc').sel(lat=site['lat'].values[0], lon=site['lon'].values[0], method='nearest')
    wet_atmo['time'] = wet_atmo['time'] - pd.to_timedelta(tz_adjust, 'h')
    df_out['precip_water'] = wet_atmo.to_dataframe()['TQV']
    
    # Ionosphere
    ion = xr.open_dataset(f'/pl/active/palomaki-sar/insar_swe_errors/data/ionosphere/ion_{time}.nc').sel(y=site['lat'].values[0], x=site['lon'].values[0], method='nearest')
    # Ionosphere data is saved at the correct local hour for am/pm but does not have hour attached
    # Add hour into timestamp (instead of subtracting from UTC)
    if time == 'am':
        timedelta = 6
    elif time == 'pm':
        timedelta = 18
    ion['time'] = ion['time'] + pd.to_timedelta(timedelta, 'h')
    df_out['ion_tec'] = ion.to_dataframe()['__xarray_dataarray_variable__']
    
    # GNSS - only one daily value, add to both AM/PM data
    # Need to add timedelta to index calculated above
    try:
        gnss = pd.read_csv(f'/pl/active/palomaki-sar/insar_swe_errors/data/gnss/raw/{station_id}.tenv3', delim_whitespace=True, index_col=1, parse_dates=True, date_format='%y%b%d')
        gnss.index += pd.to_timedelta(timedelta, 'h')
        df_out[['gnss_east','gnss_north','gnss_up']] = gnss[['__east(m)','_north(m)','____up(m)']]
    except FileNotFoundError:
        df_out[['gnss_east','gnss_north','gnss_up']] = np.nan
    
    return df_out


def calc_all_errors(df, site, return_dict=False):
    # Separate by water year if necessary
    if 'wateryear' not in df.columns:
        df['wateryear'] = [i.year + 1 if i.month in [10,11,12] else i.year for i in df.index]
        
    df_out = None
    
    # Reset cumulative calculations every year
    for wateryear in df['wateryear'].unique():
        # Add extra month to extend rolling cumulative calculations
        dt_start = pd.to_datetime(f'{wateryear-1}-10-01')
        dt_end = pd.to_datetime(f'{wateryear}-11-01')
        df_wy = df.loc[dt_start:dt_end]
        # Handle missing days - insert nans at missing timestamps
        df_ = pd.DataFrame(index=pd.date_range(df_wy.index[0], df_wy.index[-1], freq='D'))
        df_wy = pd.concat([df_, df_wy], axis=1)
        # Calculate errors from pre-compiled data
        # Soil moisture
        df_wy['swe_change'] = df_wy['swe_m'].diff(periods=12)
        df_wy['soil_error'] = calc_soil_moisture_error(sand=site['sand']/10, 
                                                       clay=site['clay']/10, 
                                                       sm_series=df_wy['soil_moisture_pct']/100)
        # Veg permittivity changes
        df_wy['veg_error'] = calc_veg_permittivity_error(canopy_height=site['canopy_height'].values[0], 
                                                         temperature_series=df_wy['airtemp_C'])
        # Dry atmosphere
        df_wy['dry_atmo_error'] = calc_dry_atmo_error(pressure_series=df_wy['surf_pres'])
        # Wet atmosphere
        df_wy['wet_atmo_error'] = calc_wet_atmo_error(pw_series=df_wy['precip_water']/1000)
        # Ionosphere
        df_wy['ion_error'] = calc_ionosphere_error(tec_series=df_wy['ion_tec'])
        # Surface deformation
        df_wy['defo_error'] = calc_deformation_error(defo_series=df_wy['gnss_up'])

        # Total error
        df_wy['total_error'] = df_wy[['soil_error','veg_error','dry_atmo_error','wet_atmo_error','defo_error','ion_error']].sum(axis=1,min_count=1)
        df_wy['non_ion_error'] = df_wy[['soil_error','veg_error','dry_atmo_error','wet_atmo_error','defo_error']].sum(axis=1,min_count=1)

        # Set up 12-day cumulative sums
        # Need to strip timestamp from datetime index
        orig_index = df_wy.index
        df_wy.index = df_wy.index.normalize()
        
        resamp_dict = {f'12d_{i}':None for i in range(12)}
        error_cols = ['soil_error','veg_error','dry_atmo_error','wet_atmo_error','ion_error','defo_error']
        for i in range(12):
            df_tmp = df_wy[error_cols].iloc[i:].resample('12d').first()
            for c in error_cols:
                df_tmp[f'{c}_cumsum'] = df_tmp[c].cumsum()
    #         df_tmp['cumsum_no_ion'] = df_tmp[['soil_error','veg_error','dry_atmo_error','wet_atmo_error','defo_error']].cumsum().sum(axis=1)
    #         df_tmp['cumsum_with_ion'] = df_tmp[error_cols].cumsum().sum(axis=1)
            df_tmp = df_tmp[[c for c in df_tmp.columns if 'cumsum' in c]]
            resamp_dict[f'12d_{i}'] = df_tmp

        daily_errors_tmp = pd.DataFrame(index=df_wy.index, columns=resamp_dict[f'12d_0'].columns)   
        for c in daily_errors_tmp.columns:
            daily_errors_tmp[c] = pd.concat([resamp_dict[f'12d_{i}'][c] for i in range(12)]).sort_index()

        df_wy = pd.concat([df_wy, daily_errors_tmp], axis=1)
        
        # Calculate rolling 12d mean of cumsums
        rolling_means = df_wy[['defo_error_cumsum','soil_error_cumsum','veg_error_cumsum','dry_atmo_error_cumsum','wet_atmo_error_cumsum','ion_error_cumsum']].rolling('12d').mean()
        # Shift entire error curve to start at 0 error at the beginning
        for c in rolling_means.columns:
            try:
                initial_error = rolling_means.loc[rolling_means[c].notnull(), c].values[0]
            except:
                initial_error = 0
            rolling_means[c] -= initial_error
        rolling_means.rename(columns={c:f'{c}_rolling' for c in rolling_means.columns}, inplace=True)
        df_wy = pd.concat([df_wy, rolling_means], axis=1)
        
        df_wy['cumsum_no_ion'] = df_wy[['defo_error_cumsum','soil_error_cumsum','veg_error_cumsum','dry_atmo_error_cumsum','wet_atmo_error_cumsum']].sum(axis=1)
        df_wy['cumsum_with_ion'] = rolling_means.sum(axis=1)
        
        # Add wateryear values back into original df
        keep_cols = []
        for c in df_wy.columns:
            if c not in df.columns:
                keep_cols.append(c)
                
        df_wy.index = orig_index
        # Subset for only this wy to remove extra month 
        df_wy = df_wy.loc[df_wy['wateryear']==wateryear]
        if df_out is None:
            df_out = df[[c for c in df.columns if c not in keep_cols]]
            df_out[keep_cols] = np.nan
        df_out.loc[df_wy.index, keep_cols] = df_wy[keep_cols]
            
    
    if return_dict:
        return df_out, resamp_dict
    else:
        return df_out


def plot_swe_curves(station_id: int, wateryear: int, time: str = 'am', ax1=None,
                    ax2=None, extra_title_text = '', viz_end_monthday: str = '07-01', 
                    plot_legends: bool = True, return_data: bool = True):
    """
    679	Paradise
    398	Clackamas Lake
    1000	Annie Springs
    846	Virginia Lakes Rdg
    821	Tipton
    759	Silvies
    417	Corral Canyon
    803	Sunset
    490	Galena Summit
    577	Lewis Lake Divide
    828	Trial Lake
    935	Jackwhacker Gulch
    708	Quemazon
    """
    sites = pd.read_csv('/pl/active/palomaki-sar/insar_swe_errors/data/snotel/fig4_sites.csv')
    site = sites.loc[sites['station_id']==station_id]
    df = pd.read_csv(f'/pl/active/palomaki-sar/insar_swe_errors/data/compiled/{station_id}_{time}.csv',
                     index_col=0, parse_dates=True)
    # Subset for water year
    df['wateryear'] = [i.year + 1 if i.month in [10,11,12] else i.year for i in df.index]
    df = df.loc[df['wateryear']==wateryear]    
    df, resamp_dict = calc_all_errors(df, site, return_dict=True)
    df['cumsum_no_ion'] -= df['cumsum_no_ion'].values[0]
    df['cumsum_with_ion'] -= df['cumsum_with_ion'].values[0]
    
    # Clip df for visualization
    df = df.loc[:f'{wateryear}-{viz_end_monthday}']
    
    if station_id == 828 and wateryear == 2017:
        df.loc[df['swe_m'].isnull(), 'swe_m'] = 0
    
    
    # Total error plot
    df['swe_m'].plot(ax=ax1, lw=2, label='SNOTEL SWE')
#         (df['non_ion_error'] + df['swe_m']).plot(ax=ax1, label='SWE + non-ion errors')
#         (df['total_error'] + df['swe_m']).plot(ax=ax1, color='k', alpha=0.5, lw=0.7, label='SWE + all errors')
    (df['cumsum_no_ion'].rolling('12d').mean() + df['swe_m']).plot(ax=ax1, label='SWE + non-ion errors', lw=2)
    (df['cumsum_with_ion'].rolling('12d').mean() + df['swe_m']).plot(ax=ax1, color='k', zorder=0, alpha=0.5, lw=1.5, label='SWE + all errors')


    ax1.set_title(f'{site["station_name"].values[0]} {wateryear}  (Peak SWE = {df["swe_m"].max():.2f} m){extra_title_text}', linespacing=1.6)
    ax1.set_xlabel('')
    ax1.set_ylabel('SWE [m]')
    ax1.set_xlim(pd.to_datetime(f'{wateryear-1}-10-01'), df.index[-1])
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0, ha='center')
    if plot_legends:
        leg = ax1.legend(fontsize=10, loc='upper center')
        for legobj in leg.legend_handles:
            legobj.set_linewidth(2.0)

#         # Error components
#         percent_error = pd.DataFrame(index=df.index)
#         relative_swe_column = 'swe_m'    # 'swe_m' or 'swe_change'
#         for c in ['soil_error','veg_error','dry_atmo_error','wet_atmo_error','ion_error','defo_error']:
#             percent_error[c] = df[c]/df[relative_swe_column]*100

#         percent_error.loc[df[relative_swe_column] < 0.1] = np.nan
#         percent_error['defo_error'].plot(ax=ax2, label='Surface deformation')
#         percent_error['soil_error'].plot(ax=ax2, label='Soil permittivity')
#         percent_error['veg_error'].plot(ax=ax2, label='Veg permittivity')
#         percent_error['dry_atmo_error'].plot(ax=ax2, label='Dry atmosphere', alpha=0.5, lw=0.8, color='k')
#         percent_error['wet_atmo_error'].plot(ax=ax2, label='Wet atmosphere', alpha=0.6, lw=0.8, color=plt.get_cmap('tab10')(3))
#         ax2.hlines([-10,10], f'{wateryear-1}-10-01', f'{wateryear}-07-01', lw=1.5, color='k', ls='--')

      # cumsum without rolling mean 
#         df['defo_error_cumsum'].plot(ax=ax2, label='Surface deformation', lw=1)
#         df['soil_error_cumsum'].plot(ax=ax2, label='Soil permittivity', lw=1)
#         df['veg_error_cumsum'].plot(ax=ax2, label='Veg permittivity', lw=1)
#         df['dry_atmo_error_cumsum'].plot(ax=ax2, label='Dry atmosphere', alpha=0.5, lw=1, color='k')
#         df['wet_atmo_error_cumsum'].plot(ax=ax2, label='Wet atmosphere', alpha=0.6, lw=1, color=plt.get_cmap('tab10')(3))
#         ax2.hlines(0, f'{wateryear-1}-10-01', f'{wateryear}-07-01', lw=1.5, color='k', ls='--', zorder=0)

      # cumsum with rolling mean 
    colors = {'defo_error_cumsum':plt.get_cmap('tab10')(3), 'soil_error_cumsum':plt.get_cmap('tab10')(5),
              'veg_error_cumsum':plt.get_cmap('tab10')(2), 'dry_atmo_error_cumsum':plt.get_cmap('tab10')(1),
              'wet_atmo_error_cumsum':plt.get_cmap('tab10')(0), 'cumsum_no_ion':'k'}
    df['dry_atmo_error_cumsum'].rolling('12d').mean().plot(ax=ax2, label='Dry troposphere', color=colors['dry_atmo_error_cumsum'], lw=2)
    df['wet_atmo_error_cumsum'].rolling('12d').mean().plot(ax=ax2, label='Wet troposphere', color=colors['wet_atmo_error_cumsum'], lw=2)
    df['veg_error_cumsum'].rolling('12d').mean().plot(ax=ax2, label='Veg permittivity', color=colors['veg_error_cumsum'], lw=2)
    df['soil_error_cumsum'].rolling('12d').mean().plot(ax=ax2, label='Soil permittivity', color=colors['soil_error_cumsum'], lw=2)
    df['defo_error_cumsum'].rolling('12d').mean().plot(ax=ax2, label='Surface deformation', color=colors['defo_error_cumsum'], lw=2)
#     df['ion_error_cumsum'].rolling('12d').mean().plot(ax=ax2, label='Ionosphere', color=plt.get_cmap('tab10')(4), lw=2)
    
    df['cumsum_no_ion'].rolling('12d').mean().plot(ax=ax2, label='Total non-ion error', color=colors['cumsum_no_ion'], lw=3, zorder=10)
    ax2.hlines(0, f'{wateryear-1}-10-01', f'{wateryear}-07-01', lw=1.5, color='k', ls='--', zorder=0)

#         percent_error['ion_error'].plot(ax=ax2, label='Ionosphere')
#         ax.set_title(f'{site["station_name"].values[0]} {wateryear}')


    ax2.set_title(f'{site["station_name"].values[0]} {wateryear} cumulative errors{extra_title_text}', linespacing=1.6)
    ax2.set_xlabel('')
    ax2.set_xlim(ax1.get_xlim())
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0, ha='center')
#         ax2.set_ylim([-100,50])
    ax2.set_ylabel('SWE error [m]')
    if plot_legends:
        leg2 = ax2.legend(fontsize=9, ncol=2, loc='upper center')
        for legobj in leg2.legend_handles:
            legobj.set_linewidth(2.0)

    ax1.tick_params(axis='x', which='minor', bottom=False)
    ax2.tick_params(axis='x', which='minor', bottom=False)
    
    if return_data:
        return df, resamp_dict


def make_temp_var_plot(ax, df_am, df_pm, error_col, site, extra_title_dict=None, n_wy_days=274, plot_pm=False, shade=True):
    c = error_col
    plot_titles = {'defo_error':'Surface deformation','soil_error':'Soil permittivity','veg_error':'Vegetation permittivity','dry_atmo_error':'Dry troposphere','wet_atmo_error':'Wet troposphere','ion_error':'Ionosphere error'}
    am_gb = df_am.loc[df_am['dowy']<=n_wy_days].groupby('dowy')
    pm_gb = df_pm.loc[df_pm['dowy']<=n_wy_days].groupby('dowy')

    month_starts = [1,32,62,93,124,152,183,213,244,274] # Through July

    min_ = am_gb.quantile(0.25)[c].rolling(12).mean()
    max_ = am_gb.quantile(0.75)[c].rolling(12).mean()
    median = am_gb.median()[c].rolling(12).mean()
    ax.plot(median, lw=2, zorder=11, label='AM orbit')
    if shade:
        ax.fill_between(min_.index, min_, max_, alpha=0.25, color=plt.get_cmap('tab10')(0), zorder=10)
        ax.plot(min_, lw=1, alpha=0.7, color=plt.get_cmap('tab10')(0), zorder=12)
        ax.plot(max_, lw=1, alpha=0.7, color=plt.get_cmap('tab10')(0), zorder=13)


    
    if plot_pm:   
        min_ = pm_gb.quantile(0.25)[c].rolling(12).mean()
        max_ = pm_gb.quantile(0.75)[c].rolling(12).mean()
        median = pm_gb.median()[c].rolling(12).mean()
        ax.plot(median, lw=2, zorder=1, label='PM orbit')
        if shade:
            ax.fill_between(min_.index, min_, max_, alpha=0.25, color=plt.get_cmap('tab10')(1), zorder=0)
            ax.plot(min_, lw=1, alpha=0.7, color=plt.get_cmap('tab10')(1), zorder=2)
            ax.plot(max_, lw=1, alpha=0.7, color=plt.get_cmap('tab10')(1), zorder=3)
        leg = ax.legend(ncol=2, fontsize=9, loc='lower right')
        leg.set_zorder(100)
    
    ax.set_xticks(month_starts)
    ax.set_xticklabels(['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul'])
    if c == 'defo_error':
        st_name = site['station_name'].values[0]
        ax.set_title(st_name + f'{extra_title_dict[st_name]}\n\n' + plot_titles[c])
    else:
        ax.set_title(plot_titles[c])


def plot_temporal_variability(station_id: int, ax: plt.axes = None, extra_title_dict: dict = {}, return_data: bool = True):
    """
    679	Paradise
    398	Clackamas Lake
    1000	Annie Springs
    846	Virginia Lakes Rdg
    821	Tipton
    759	Silvies
    417	Corral Canyon
    803	Sunset
    490	Galena Summit
    577	Lewis Lake Divide
    828	Trial Lake
    935	Jackwhacker Gulch
    708	Quemazon
    """
    sites = pd.read_csv('/pl/active/palomaki-sar/insar_swe_errors/data/snotel/fig4_sites.csv')
    site = sites.loc[sites['station_id']==station_id]
    df_am = pd.read_csv(f'/pl/active/palomaki-sar/insar_swe_errors/data/compiled/{station_id}_am.csv',
                     index_col=0, parse_dates=True)
    df_pm = pd.read_csv(f'/pl/active/palomaki-sar/insar_swe_errors/data/compiled/{station_id}_pm.csv',
                     index_col=0, parse_dates=True)
    
    df_am = calc_all_errors(df_am, site)
    df_pm = calc_all_errors(df_pm, site)
    
    
    for df in [df_am, df_pm]:
    # Subset for water year
        df['wateryear'] = [i.year + 1 if i.month in [10,11,12] else i.year for i in df.index]
        df['dowy'] = [(r[0] - pd.to_datetime(f'{int(r[1]["wateryear"])-1}-10-01')) for r in df.iterrows()] # type: ignore
        df['dowy'] = df['dowy'].dt.days + 1


    # Plot
    if ax is None:
        fig, ax = plt.subplots(6, 1, figsize=(5,20))
    error_cols = ['defo_error','soil_error','veg_error','dry_atmo_error','wet_atmo_error','ion_error']
    for i, c in enumerate(error_cols):
        if c == 'ion_error':
            make_temp_var_plot(ax[i], df_am, df_pm, c, site, extra_title_dict=extra_title_dict, plot_pm=True, shade=True)
        else:
            make_temp_var_plot(ax[i], df_am, df_pm, c, site, extra_title_dict=extra_title_dict, plot_pm=False)
        
        
    if return_data:
        return (df_am, df_pm)


def calculate_avg_cumsum_errors(station_id: int):
    """
    679	Paradise
    398	Clackamas Lake
    1000	Annie Springs
    846	Virginia Lakes Rdg
    821	Tipton
    759	Silvies
    417	Corral Canyon
    803	Sunset
    490	Galena Summit
    577	Lewis Lake Divide
    828	Trial Lake
    935	Jackwhacker Gulch
    708	Quemazon
    """
    sites = pd.read_csv('/pl/active/palomaki-sar/insar_swe_errors/data/snotel/fig4_sites.csv')
    site = sites.loc[sites['station_id']==station_id]
    df_am = pd.read_csv(f'/pl/active/palomaki-sar/insar_swe_errors/data/compiled/{station_id}_am.csv',
                     index_col=0, parse_dates=True)
    df_pm = pd.read_csv(f'/pl/active/palomaki-sar/insar_swe_errors/data/compiled/{station_id}_pm.csv',
                     index_col=0, parse_dates=True)
    
    df_am = calc_all_errors(df_am, site)
    df_pm = calc_all_errors(df_pm, site)        
    
    for df in [df_am, df_pm]:
    # Subset for water year
#         df['wateryear'] = [i.year + 1 if i.month in [10,11,12] else i.year for i in df.index]
        df['dowy'] = [(r[0] - pd.to_datetime(f'{int(r[1]["wateryear"])-1}-10-01')) for r in df.iterrows()] # type: ignore
        df['dowy'] = df['dowy'].dt.days + 1
        
    # Groupby dowy
    am_gb = df_am.loc[df_am['dowy']<=274].groupby('dowy')
    pm_gb = df_pm.loc[df_pm['dowy']<=274].groupby('dowy')
    
    df_out = am_gb#.mean()[183]
    return df_out