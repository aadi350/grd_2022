import ee
import pandas as pd
import logging

logging.basicConfig(encoding='utf-8', level=logging.DEBUG)

ee.Initialize()

# OECS polygon
def get_oecs_polygon() -> ee.Geometry.Polygon:
    return ee.Geometry.Polygon(
        [[[-65.39205942840965, 17.34588919953531],
          [-63.63424692840965, 17.031019992676466],
          [-62.51364145965965, 15.067319140114463],
          [-62.95309458465965, 11.862481700190973],
          [-60.95358286590965, 11.453614365346695],
          [-59.92086802215965, 14.472416632307414],
          [-61.14206377672779, 18.228742495073664],
          [-64.6796614329778, 19.32090631615009]]]
        )


oecs = get_oecs_polygon()

def get_date_string(day:int, month:int, year:int) -> str:
    '''Takes day, month & year and joins into
    single string for use in Earth-engine APIS'''

    if day < 10:
        day = '0' + str(day)
    if month < 10:
        month = '0' + str(month)

    return f'{year}-{month}-{day}'


def average_by_month(dataset: ee.ImageCollection, band: str):
    months = ee.List.sequence(1, 12)
    by_month = ee.ImageCollection.fromImages(
        months.map(lambda m: dataset.filter(
                ee.Filter.calendarRange(m, m, 'month'))
                    .select(band).mean()
                    .set('month', m)
                )
            )

    return by_month

def select_temperature(dataset: ee.ImageCollection, depth=0, scale=0.001, shift=20):
    return dataset.select(f'water_temp_{depth}').map(lambda img: ee.Image(img).multiply(scale).add(shift))

def clip_to_geometry(dataset: ee.ImageCollection, poly: ee.Geometry):
    return dataset.map(lambda img: img.clip(poly))
#logging.debug(ee.Image.reduceToVectors(temp_0.first()).getInfo())

def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame.
    Modification of 
    https://developers.google.com/earth-engine/tutorials/community/intro-to-python-api-guiattard
    """
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)
    logging.debug(df.head())
    # Remove rows without data inside.
    #df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    return df

def load_hycom_region(start_date, end_date, region=oecs, band='water_temp_2'):
    IMAGE_COLLECTION = 'HYCOM/sea_temp_salinity'
    dataset = ee.ImageCollection(IMAGE_COLLECTION).filterDate(start_date, end_date)
    full_dataset = ee.ImageCollection(IMAGE_COLLECTION)

    oecs = get_oecs_polygon()
    averaged = average_by_month(dataset, band)
    clipped = clip_to_geometry(averaged, region)

    return clipped

def get_rect(averaged_data, band, scale, shift):
    single_band = averaged_data.select(band).map(lambda img: ee.Image(img).multiply(scale).add(shift))
    collection_area_of_salinity = single_band.getRegion(oecs, scale=4000).getInfo()
    df_area_of_interest = ee_array_to_df(collection_area_of_salinity, [band])

    return df_area_of_interest

def get_temp_rect(averaged_data):
    band = 'water_temp_2'
    scale=0.001
    shift=20

    temp_rect = get_rect(averaged_data, band, scale, shift)
    temp_rect.drop('time', axis=1, inplace=True)
    logging.debug(temp_rect.id.unique()) 

    return temp_rect

def get_temp_rect_year(year):

    clipped_avg = load_hycom_region(f'{year}-01-01', f'{year}-12-31')
    temp_rect = get_temp_rect(clipped_avg)
   

    temp_rect.columns = ['month','lon', 'lat', 'water_temp_2']

    temp_rect['year'] = year
    return temp_rect


def get_sal_rect(averaged_data):
    band = 'salinity_2'
    scale=0.001
    shift=20

    sal_rect = get_rect(averaged_data, band, scale, shift)
    sal_rect.drop('time', axis=1, inplace=True)
    logging.debug(sal_rect.id.unique()) 

    return sal_rect

def get_sal_rect_year(year):

    clipped_avg = load_hycom_region(f'{year}-01-01', f'{year}-12-31', band='salinity_2')
    sal_rect = get_sal_rect(clipped_avg)
   

    sal_rect.columns = ['month','lon', 'lat', 'salinity_2']

    sal_rect['year'] = year

    return sal_rect

data = pd.DataFrame(columns = ['month', 'lon', 'lat', 'salinity_2', 'year'])

for year in range(2000, 2022):
    year_data = get_sal_rect_year(year)
    data = data.append(year_data)
    logging.debug(year_data.head())

data.to_csv('res/salinity_2.csv', index=False)

