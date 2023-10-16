import os
import ee
import glob
# import shutil
import requests
import geemap
import numpy as np
# import natsort
# import folium
import Plugin_download
import rasterio
from rasterio.merge import merge
from rasterio.features import shapes
from shapely.ops import unary_union
from codes import shoreline
# from rasterio.mask import mask
from natsort import natsorted
from pyproj import CRS
import geopandas as gpd
# import shapely
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.polygon import LinearRing
import warnings
warnings.filterwarnings("ignore")


service_account = 'brian-python@ee-briansimiyu.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, './ee-brianchelloti-f3bf0b9227be.json')
ee.Initialize(credentials)

# Define the working CRS
epsg_code = "3857"


# Load the roi shapefile
roi = './SHP4/plugin_test555.shp'
roi = geemap.shp_to_ee(roi)
roi = roi

# Create temporary folder
if not os.path.exists('./output/WaterMask/pre-processed-images'):
  os.makedirs('./output/WaterMask/pre-processed-images')

# ------ Main Code Execution ------ #
# Split the geometry to small grid 
# grid = geemap.fishnet(roi, h_interval=0.05, v_interval=0.05, delta=1)
grid = roi
gridList = grid.toList(grid.size())
grid_num = grid.toList(grid.size()).length()

# Create list of each grid feature
ls_feature = []
for i in range(grid_num.getInfo()):
  feature = ee.Feature(gridList.get(i)).geometry()
  ls_feature.append(feature)

  
# Create temporary folder
if not os.path.exists('./output/Tiles'): 
    os.makedirs('./output/Tiles')
    
# 2. Define start date and end date
date = ['2020-01-01', '2020-12-31']

# ------ Main Code Execution ------ #
start_date = date[0]
end_date = date[1]

# Create filename
start_yyyy = start_date[:4]
start_mm = start_date[5:7]
end_yyyy = end_date[:4]
end_mm = end_date[5:7]
file_name = start_yyyy+start_mm+'_'+end_yyyy+end_mm

#Download image by grid
for i in range(grid_num.getInfo()):
  # Extract image from GEE and apply pre-process function
  image = Plugin_download.waterMask_download(roi, ls_feature[i], start_date, end_date)

  BandIDs = ['waterMask']
  # Download pre-processed image
  
  download_id = ee.data.getDownloadId({
      'image': image,
      'bands': BandIDs,
      'region': ls_feature[i],
      'scale': 10,
      'format': 'GEO_TIFF',
      'crs' : 'EPSG:'+str(epsg_code),
  })

  response = requests.get(ee.data.makeDownloadUrl(download_id))
  with open('./output/Tiles/image_grid_'+str(i)+'.tif', 'wb') as fd:
    fd.write(response.content)
    
    # ------ Merge all grid image to one image ------ #
# Make a search criteria to select the image files
q = os.path.join('./output/Tiles/image*.tif') 

# sorted files by name
fp = natsorted(glob.glob(q)) 

# List for storing the raster image
src_files = []

# Open each raster files by iterating and then append to our list
for raster in fp:
    # open raster file
    files = rasterio.open(raster)

    # add each file to our list
    src_files.append(files)

# Merge function returns a single mosaic array and the transformation info
mosaic, out_trans = merge(src_files)

# Set metadata
out_meta = src_files[0].meta.copy()
out_meta.update({"driver": "GTiff",
                "dtype": "float32",
                "nodata": 0,
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "count": 1,
                "crs": CRS.from_epsg(int(epsg_code))})
                
# Write the mosaic raster
output = os.path.join('./output/WaterMask/pre-processed-images'+'/WaterMask_'+file_name+'.tif')
with rasterio.open(output, "w", tiled=True, compress='lzw', **out_meta) as dest:
    dest.write(mosaic.astype(np.float32))
    
    # Create folder
if not os.path.exists('./output/WaterMask/post-processed-images'):
  os.makedirs('./output/WaterMask/post-processed-images')
if not os.path.exists('./output/shoreline/geojson'):
  os.makedirs('./output/shoreline/geojson')
  
# ------ Shoreline extraction ------ #
# Read input image data
Image = rasterio.open('./output/WaterMask/pre-processed-images'+'/WaterMask_'+file_name+'.tif')
if Image.read().any() == 0:
  print('Warning: The image is empty, so shoreline cannot be extracted.')

else:
  print('Extracting shoreline')
  
rescale_image, transform = shoreline.resampling(image=Image, scale_factor=5)

# Georeference
horizontal_step = 0     # (+ Positive) Move to right // (- Negative) Move to left
vertical_step = 0       # (+ Positive) Move to top // (- Negative) Move to bottom

ncol = -horizontal_step
nrow = vertical_step
buffer_rate = 0

    # Slice column number
if ncol < 0:
  # Negative
  rescale_image = np.delete(rescale_image, np.s_[rescale_image.shape[2]-abs(ncol):rescale_image.shape[2]], axis=2)
  new_col = np.empty((rescale_image.shape[0],rescale_image.shape[1], abs(ncol)))
  new_col.fill(np.nan)
  rescale_image = np.array([np.c_[new_col[i], rescale_image[i]] for i in range(rescale_image.shape[0])])
else:
  # Positive
  rescale_image = np.delete(rescale_image, slice(0,ncol), axis=2)
  new_col = np.empty((rescale_image.shape[0],rescale_image.shape[1], ncol))
  new_col.fill(np.nan)
  rescale_image = np.c_[rescale_image, new_col]

# Slice row number
if nrow < 0:
  # Negative
  rescale_image = np.delete(rescale_image, np.s_[rescale_image.shape[1]-abs(nrow):rescale_image.shape[1]], axis=1)
  new_row = np.empty((rescale_image.shape[0], abs(nrow), rescale_image.shape[2]))
  new_row.fill(np.nan)
  rescale_image = np.array([np.r_[new_row[i], rescale_image[i]] for i in range(rescale_image.shape[0])])
else:
  # Positive
  rescale_image = np.delete(rescale_image, slice(0,abs(nrow)), axis=1)
  new_row = np.empty((rescale_image.shape[0], abs(nrow), rescale_image.shape[2]))
  new_row.fill(np.nan)
  rescale_image = np.array([np.r_[rescale_image[i], new_row[i]] for i in range(rescale_image.shape[0])])

# Set metadata
out_meta = Image.meta.copy()
out_meta.update({"driver": "GTiff",
                "dtype": "float32",
                "nodata": 0,
                "height": rescale_image.shape[1],
                "width": rescale_image.shape[2],
                "transform": transform,
                "count": 1,
                "crs": Image.crs
                }
                )
# Write the clip raster
output = os.path.join('./output/WaterMask/post-processed-images/WaterMask_'+file_name+'.tif')
with rasterio.open(output, "w",tiled=True, compress='lzw', **out_meta) as dest:
    dest.write(rescale_image.astype(np.float32))
    
rescale_image = rasterio.open('./output/WaterMask/post-processed-images/WaterMask_'+file_name+'.tif')

mask = rescale_image.read(1)

    # ------ Save result as GeoJSON file ------ #
# Export result
polygons = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(mask.astype('uint8'), mask=None, transform=transform)))
geometry = list(polygons)

# Create new geodataframe
geom_dataframe  = gpd.GeoDataFrame.from_features(geometry)

# Set projection of dataframe
geom = geom_dataframe.set_crs(Image.crs)

# Remove no-data geometries
geom = geom[geom.raster_val != 0]

# Extract  boundaries
# list_poly = []
# for p in geom['geometry']:
#     for interior in p.interiors:
#         list_poly.append(Polygon(interior))

list_poly = []
for p in geom['geometry']:
  list_poly.append(Polygon(p.exterior))

smooth_poly = []
for i in range(len(list_poly)):
  poly_line = list_poly[i]
  outbuffer = poly_line.buffer(10, resolution=5, cap_style=1, join_style=1, mitre_limit=2, single_sided=True)
  inbuffer = outbuffer.buffer(-10.5, resolution=5, cap_style=1, join_style=1, mitre_limit=2, single_sided=True)
  simplified = inbuffer.simplify(1, preserve_topology=False) # False: Use Douglas-Peucker algorithm
  inbuffer2 = simplified.buffer(buffer_rate, resolution=5, cap_style=1, join_style=1, mitre_limit=2, single_sided=True)
  
  if type(inbuffer2) == MultiPolygon:
    ring = [LinearRing(inbuffer2.geoms[k].exterior) for k in range(len(inbuffer2.geoms))]
    ring = unary_union(ring)
    smooth_poly.append(ring)
  else:
    ring = LinearRing(inbuffer2.exterior)
    smooth_poly.append(ring)

# Create new geodataframe for exterior boundaries
geo_shoreline = gpd.GeoDataFrame({'geometry':smooth_poly}, crs=Image.crs)
geo_shoreline = geo_shoreline.dropna().reset_index(drop=True)
geo_shoreline['id'] = geo_shoreline.index

# Save to geojson file
outfp = './output/shoreline/geojson/shoreline_'+file_name+'.json'
geo_shoreline.to_file(outfp, driver='GeoJSON')