import ee

################## Sentinel-1 ####################
# Mask cloud function for Sentinel
def maskCloudSeninel(image):
  # Bits 10 and 11 are clouds and cirrus, respectively.
  cloudBitMask = (1 << 10)
  cirrusBitMask = (1 << 11)
  # Get the pixel QA band
  qa = image.select('QA60')
  # Both flags should be set to zero, indicating clear conditions
  mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0))
  return image.updateMask(mask).divide(10000)

# Create no-cloud image and clip area of interest
def Sentinel_no_clouds(aoi, start_date, end_date):
  # importing image collection and filtering
  # Rescale and mask cloud
  S2_collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
  img_col = S2_collection.filterDate(start_date, end_date).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
  img_maskCloud = img_col.map(maskCloudSeninel).median()
  # Clip region
  dataset = img_maskCloud.clip(aoi)
  return dataset

def waterMask_download(aoi, aoi_grid, start_date, end_date):
    
    # Import Sentinel-1 and filter the data series
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
           # .filter(ee.Filter.eq('instrumentMode', 'IW'))
           # .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
          .filterBounds(aoi)
          .filterDate(start_date, end_date)
           # .filter(ee.Filter.contains(leftField=".geo", rightValue=aoi))
          .map(lambda image: image.clip(aoi))
          .map(lambda image: image.addBands(image.select('VV').focal_median(float('50'), 'circle', 'meters').rename('VV_smoothed'))))


    # Define the Otsu function
    def otsu(histogram):
        counts = ee.Array(ee.Dictionary(histogram).get('histogram'))
        means = ee.Array(ee.Dictionary(histogram).get('bucketMeans'))
        size = means.length().get([0])
        total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
        sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
        mean = sum.divide(total)
        
        indices = ee.List.sequence(1, size)
        
        def compute_bss(i):
            aCounts = counts.slice(0, 0, i)
            aCount = aCounts.reduce(ee.Reducer.sum(), [0]).get([0])
            aMeans = means.slice(0, 0, i)
            aMean = aMeans.multiply(aCounts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(aCount)
            bCount = total.subtract(aCount)
            bMean = sum.subtract(aCount.multiply(aMean)).divide(bCount)
            return aCount.multiply(aMean.subtract(mean).pow(2)).add(bCount.multiply(bMean.subtract(mean).pow(2)))
        
        bss = indices.map(compute_bss)
        
        return means.sort(bss).get([-1])
    
    
    # Define a function to add a water mask
    def add_water_mask(image):
        # Compute histogram
        histogram = image.select('VV').reduceRegion(
            reducer=ee.Reducer.histogram(255, 2)
            .combine('mean', None, True)
            .combine('variance', None, True),
            geometry=aoi,
            scale=10,
            bestEffort=True)
        
        # Calculate threshold using the Otsu function
        threshold = otsu(histogram.get('VV_histogram'))
        
        # Get the water mask
        water_mask = image.select('VV_smoothed').lt(threshold).rename('waterMask')
        water_mask = water_mask.updateMask(water_mask)  # Remove all pixels equal to 0
        return image.addBands(water_mask).select("waterMask").clip(aoi).copyProperties(image, ['system:time_start', 'system:time_end'])
    
    # Apply the water mask function to the Sentinel-1 ImageCollection
    s1 = s1.map(add_water_mask)
    
    s1 = s1.sort('system:time_end',False).first()
    # s1 = s1.mean()
    
    return s1