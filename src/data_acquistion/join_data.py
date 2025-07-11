import pandas as pd
import xarray as xr

def compile_timeseries(station_id, time='am'):
    sites = pd.read_csv('/pl/active/palomaki-sar/insar_swe_errors/data/snotel/fig4_sites.csv')
    site = sites.loc[sites['station_id']==station_id]
    df = pd.read_csv(f'/pl/active/palomaki-sar/insar_swe_errors/data/snotel/hourly/{station_id}_hourly_2016_2025.csv',
                    index_col=0, parse_dates=True)
    df.index = df.index.tz_convert(site['timezone'].values[0]).tz_localize(None)
    
    
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
    dry_atmo['time'] = dry_atmo['time'] - pd.to_timedelta('6h')
    df_out['surf_pres'] = dry_atmo.to_dataframe()['PS']
    
    # Wet atmosphere
    wet_atmo = xr.open_dataset(f'/pl/active/palomaki-sar/insar_swe_errors/data/atmo_wet/pw_{time}.nc').sel(lat=site['lat'].values[0], lon=site['lon'].values[0], method='nearest')
    wet_atmo['time'] = wet_atmo['time'] - pd.to_timedelta('6h')
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
    
    return df_out
    

sites = pd.read_csv('/pl/active/palomaki-sar/insar_swe_errors/data/snotel/fig4_sites.csv')
for station in sites['station_id']:
    for time in ['am','pm']:
        data = compile_timeseries(station, time=time)
        data.to_csv(f'/pl/active/palomaki-sar/insar_swe_errors/data/compiled/{station}_{time}.csv')