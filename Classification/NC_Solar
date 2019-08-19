//1. SET UP FUNCTIONS FOR LATER
// Import cloud masking functions from defenders modules
var Clouds = require('users/defendersofwildlifeGIS/Modules:Clouds')

//Function to process S2 imagery masking for clouds, water and shadow
function maskclouds(img){
  var clouds = Clouds.basicQA(img);
  clouds = Clouds.sentinelCloudScore(clouds);
  var waterMask = Clouds.waterScore(img).select('waterScore').lte(0.5);
  var shadowMask = img.select('B11').gt(900);
  //var darkC = Clouds.darkC(img, ['B4', 'B3', 'B2'])
  //var nd = img.normalizedDifference(['B2', 'B12']).rename('nd');
  //var darkbands = img.select(['B8', 'B11', 'B12']).reduce(ee.Reducer.sum()).rename('darkbands');
  //var out = img.addBands([clouds, water, nd, darkbands]);
  return clouds.updateMask(clouds.select('cloudScore').lte(25));//.and(shadowMask).and(waterMask));
}

//Function to add derived variables to multispectral data
//TO DO pan sharpen swir1 & swir2 bands?
var addVariables = function(image) {
  var r = image.select('B4');
  var b = image.select('B2');
  var g = image.select('B3');
  var n = image.select('B8');
  //var time = image.date().difference('2000-01-01', 'year');
  var vndvi = image.normalizedDifference(['B3', 'B4']).rename('vndvi');
  var gli = g.multiply(2).subtract(b).subtract(r).divide(g.multiply(2).add(r).add(b)).rename('gli');
  //var gli = image.expression(
  //  '((2* b("G")) - b("R")- b("B"))/((2 * b("G")) + b("R") + b("B"))
  //  ).rename('gli');
  var lightness = r.max(b).max(g).add(r.min(g).min(b)).rename('light');
  var luminosity = r.multiply(0.21).add(g.multiply(0.72).add(b.multiply(0.07))).rename('luminosity');
  var rgbavg = image.select(['B4', 'B3', 'B2']).reduce(ee.Reducer.mean());
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('ndvi');
  var ndsi = image.normalizedDifference(['B2', 'B11']).divide(image.select(['B8'])).rename('ndsi');
  var ndwi = image.normalizedDifference(['B8', 'B11']).rename('ndwi')
  return(image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
  .addBands([vndvi, ndvi, ndsi, ndwi, gli, lightness, luminosity, rgbavg]));
};

//Function to process horizontal polarization sentinel 1 data
function hprocess(img_col, region, type){
  var area = img_col.filterBounds(region);
  var time = area.map(add_time);
  var med;
  if (type == 'recent'){
    med = time.qualityMosaic('system:time_start')
    .clipToCollection(region);
  } else if (type == 'median'){
      med = time.median()
      .clipToCollection(region);
  }
  var norm = med.normalizedDifference(['HH', 'HV']).rename('norm');
  var diffh = med.select('HH').subtract(med.select('HV')).rename(['HHminHV']); 
  var sumh = med.select('HH').add(med.select('HV')).rename(['HVpluHH']);
  var ratioh = med.select('HH').divide(med.select('HV')).rename(['HHdivHV']);
  return med.addBands(ratioh).addBands(sumh);
  //var final = norm(nd, bands_class, aoi);
}
//Function to select and process vertical polarization sentinel 1 data
function vprocess(img_col, region, type){
  var area = img_col.filterBounds(region);
  var time = area.map(add_time);
  var med;
  if (type == 'recent'){
    med = time.qualityMosaic('system:time_start')
    .clip(region);
  } else if (type == 'median'){
      med = time.median()
      .clip(region);
  }
  //var norm = med.normalizedDifference().rename('norm');
  var diffv = med.select('VV').subtract(med.select('VH')).rename(['VVminVH']);
  var sumv = med.select('VV').add(med.select('VH')).rename(['VVpluVH']);
  var ratiov = med.select('VV').divide(med.select('VH')).rename(['VVdivVH']);
  var norm = diffv.divide(sumv).rename('norm');
  return med.addBands(ratiov).addBands(sumv).addBands(diffv).addBands(norm);
  //var final = norm(nd, bands_class, aoi);
}
// Comine H and V polarization sentinel 1 data
function vdiff(img){
  var vvsubvh = img.select(['VV']).subtract(img.select(['VH'])).rename('VVsubVH')
  return img.addBands(vvsubvh);
}

function adcomp(ascend, descend){
  var vv = ee.ImageCollection(ascend.select(['VV']), descend.select(['VV'])).mean();
  var vh = ee.ImageCollection(ascend.select(['VH']), descend.select(['VH'])).mean();
  var vvvh = ee.ImageCollection(ascend.select(['VVsubVH']), descend.select(['VVsubVH'])).mean();
  return ee.Image.cat(vv, vh, vvvh);
}

function create_composite_img(aoi){
  var collection = S2.filterDate('2019-01-01','2019-04-28')
                  .filterBounds(aoi)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskclouds)
                  .map(addVariables);
                  
  first = collection.median().clip(aoi)
  variance = colleciton.reduce(ee.Reducer.variance()).clip(aoi)
  var v = sentinel1
  // Filter to get images with VV and VH dual polarization.
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  // Filter to get images collected in interferometric wide swath mode.
    .filter(ee.Filter.eq('instrumentMode', 'IW'));
  
  var h = sentinel1
  // Filter to get images with HH and HV dual polarization.
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HH'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HV'))
  // Filter to get images collected in interferometric wide swath mode.
    .filter(ee.Filter.eq('instrumentMode', 'IW'));
    
  var vhAscending = v.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
  var vhDescending = v.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

  var ascend=vhAscending
  .filterDate('2019-03-01','2019-04-28').filterBounds(aoi).map(vdiff);

  var descend=vhDescending
  .filterDate('2019-02-01','2019-04-28').filterBounds(aoi).map(vdiff);

  s1comp = adcomp(ascend, descend).clip(aoi)
  
  return first.addBands(variance).addBands(s1comp);
}

// 2. COLLECT AND CURATE IMAGERY
//Collect and process sentinel 1 data
var v = sentinel1
  // Filter to get images with VV and VH dual polarization.
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  // Filter to get images collected in interferometric wide swath mode.
  .filter(ee.Filter.eq('instrumentMode', 'IW'));
  
var h = sentinel1
  // Filter to get images with HH and HV dual polarization.
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HH'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'HV'))
  // Filter to get images collected in interferometric wide swath mode.
  .filter(ee.Filter.eq('instrumentMode', 'IW'));

// Filter to get images from different look angles.
var vhAscending = v.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
var vhDescending = v.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

var ascend=vhAscending
.filterDate('2019-03-01','2019-04-28').filterBounds(geometry).map(vdiff);

var descend=vhDescending
.filterDate('2019-02-01','2019-04-28').filterBounds(geometry).map(vdiff);

var currentcomp = adcomp(ascend, descend).clip(geometry);

// Create a three month colleciton of S2 images in the study area,
// cloud mask and add variables

var collection = S2.filterDate('2019-01-01','2019-04-28')
                  .filterBounds(geometry)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskclouds)
                  .map(addVariables);

// compute temporal variance of all pixels
var variance = collection.reduce(ee.Reducer.variance());

// create median composite for visualization
var first = collection.median().clip(geometry);

var data = first.addBands(variance).addBands(currentcomp);
// Alternative features 
// CREATE 2-pixel radius kernel
var list1 = [1,0,1,0,1];
var list2 = [0,0,0,0,0];
var list3 = [1,0,0,0,1];
var weights = [list1, list2, list3, list2, list1];

var kernel2 = ee.Kernel.fixed({
  width: 5,
  height: 5,
  weights: weights
})

//CREATE 4-pixel radius kernel
var list1 = [1,0,0,0,1,0,0,0,1];
var list2 = [0,0,0,0,0,0,0,0,0]
var weights = [list1, list2, list2, list2, list1, list2, list2, list2, list1]
var kernel4 = ee.Kernel.fixed({
  width: 9,
  height: 9,
  weights: weights
})

var neighborhood = first.select(['B2', 'B3', 'B4', 'B8']).reduceNeighborhood({
  reducer: ee.Reducer.mean().combine({
    reducer2: ee.Reducer.variance(),
    sharedInputs: true}),
  kernel: ee.Kernel.square(1, 'pixels')
});

var neighborhood2 = neighborhood.neighborhoodToBands(kernel2)
var neighborhood4 = neighborhood.neighborhoodToBands(kernel4)

var rings = neighborhood2.addBands(neighborhood4)


// 3. SET UP FEATURES FOR SAMPLING TRAINING DATA
// set landcover property to '1' for solar panels
var solar = ee.FeatureCollection(table).map(function(ft){return ft.set('landcover', 1)})
var good_solar = solar.filterMetadata('Status', 'equals', 'Established');
var solar_buff = good_solar.map(function(ft){return ft.buffer(-50)})
var solar_cent = good_solar.map(function(ft){return ft.centroid(10)})

// Remap nlcd into broad categories
var nlcd = NLCD.select('landcover').clip(geometry);
Map.addLayer(nlcd, {}, 'nlcd');
nlcd = nlcd.remap({
  from: [23, 24, 41, 42, 43, 90, 95, 81, 82, 21, 11],
  to: [2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5],
  bandName: 'landcover'
}).rename(['landcover']);

// Create observations of features at solar panel
var solar_ft = data.sampleRegions({
  //Using manually curated points resulted in better performance
  collection: solar_pts,
//Alternatively, we can automatically collect sample points from within well
//-curated training areas
  //colleciton: solar_buff.filterBounds(geometry),
  properties: ['landcover'],
  tileScale: 16,
  scale: 10,
  geometries: true
  });
  
  
//Create a stratified sample of 1000 observations
// of features in each broad landcover class
var nat_ft = data.addBands(nlcd).stratifiedSample({
  numPoints: 1000,
  classBand: 'landcover',
  region: geometry,
  tileScale: 16,
  scale: 10,
  geometries: true
});

//Adding manual 'not-solar' points in areas where the model initially mis-identified
// solar fields improves performance
var non_ft = data.sampleRegions({
  collection: non_pts,
  properties: ['landcover'],
  tileScale: 16,
  scale: 10,
  geometries: true
})

//Combine all observations 
nat_ft = nat_ft.filter(ee.Filter.bounds(solar).not());

///Check our sample sizes for each class of observations
print("natural features:", nat_ft.size());
print("solar observations:", solar_ft.size());
var ft = nat_ft.merge(solar_ft).merge(non_ft).randomColumn();

Export.table.toAsset({
  collection: ft,
  description: 'solar_samples_5e3d8aef33f3fb0d30c068ce8d69b2be'
})

//Split observations into training and testing data
var validation = ft.filter(ee.Filter.lt('random', 0.3));
var training = ft.filter(ee.Filter.gte('random', 0.7));

// 4. TRAIN CLASSIFIER AND MAKE PREDICTIONS
//Train a random forest classifier using training data
var classifier = ee.Classifier.randomForest(20).train(training, 'landcover', data.bandNames());

//Test classifier performance on validation data and create error matrix
var holdout = validation.classify(classifier)
.errorMatrix('landcover', 'classification');

print('variance', holdout.accuracy());

//Classify image
var result = data.clip(geometry).classify(classifier);

var eroded = result.focal_mode(1, 'square', 'pixels')

// Display the results.
var data = first.addBands(variance).clip(geometry);
Map.addLayer(data, {bands:['B4', 'B3', 'B2'], min:500, max:1500},'S2');
Map.addLayer(eroded, {bands:'classification', palette:['#000000', '#ffffff', '#099500', '#fff700', '#005ce7'], min:1, max:5}, 'result');
Map.addLayer(solar_buff, {color: '#ff00eb'}, 'solar fields');

//TEST UNMIXING APPROACH
var nat_dict = nat_ft.reduceColumns({
  reducer: ee.Reducer.median().repeat(8).group(8, 'landcover'),
  selectors: ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'ndvi', 'ndsi', 'landcover']
})

var solar_dict = solar_ft.reduceColumns({
  reducer: ee.Reducer.median().repeat(8),
  selectors: ['B2', 'B3', 'B4', 'B8', 'B11', 'B12','ndvi', 'ndsi']
})


var urbanList = ee.Dictionary(ee.List(nat_dict.get('groups')).get(0)).get('median');
var greenList = ee.Dictionary(ee.List(nat_dict.get('groups')).get(1)).get('median');
var openList = ee.Dictionary(ee.List(nat_dict.get('groups')).get(2)).get('median');
var waterList = ee.Dictionary(ee.List(nat_dict.get('groups')).get(3)).get('median');
var solarList = solar_dict.get('median')
//print(solarList);

var miximg = first.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'ndvi', 'ndsi'])
.unmix(
  [urbanList, greenList, openList, waterList, solarList],
  true)

//Map.addLayer(miximg, {}, 'unmix')
