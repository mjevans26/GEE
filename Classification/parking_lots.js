//SET UP FEATURES FOR SAMPLING TRAINING DATA
plots = ee.FeatureCollection(plots).map(function(ft){return ft.set('landcover', 1)});
bldgs = ee.FeatureCollection(bldgs).map(function(ft){return ft.set('landcover', 2)});
rds = ee.FeatureCollection(rds).map(function(ft){return ft.set('landcover', 3).buffer(10)});
rds = rds.reduceToImage(['landcover'], ee.Reducer.first());
var outbnd = ee.FeatureCollection(LI).union(10);

var imperv = plots.merge(bldgs).filterBounds(geometry);
var doi = '2017-03-01';
var today = ee.Date(Date.now());
var centroids = plots.filter(ee.Filter.gt('AREA', 10)).map(function(ft){return ft.centroid()});
var random = ee.FeatureCollection.randomPoints(geometry, 1000);

Map.addLayer(imperv, {}, 'bldgs&rds');
//Map.addLayer(centroids, {}, 'centroids');
//Map.addLayer(random, {}, 'randpts');

//get most recent not cloudy s2 image

//.filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 10)
//.sort('system:time_start', false).first());
/*
var before = ee.Image(S2.filterDate('2016-07-01', '2016-08-31').sort('CLOUDY_PIXEL_PERCENTAGE', true).first());
var after = ee.Image(S2.filterDate('2017-07-01', '2017-08-31').sort('CLOUDY_PIXEL_PERCENTAGE', true).first());
*/

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
.filterDate('2018-05-01','2018-07-30');

var descend=vhDescending
.filterDate('2018-05-01', '2018-07-30');
/*
var d_past=vhDescending
.filterDate('2016-07-01','2016-08-31');

var d_current=vhDescending
.filterDate('2017-07-01', '2017-08-31');
*/
function adcomp(ascend, descend){
  var vv = ee.ImageCollection(ascend.select(['VV']), descend.select(['VV'])).mean();
  var vh = ee.ImageCollection(ascend.select(['VH']), descend.select(['VH'])).mean();
  return ee.Image.cat(vv, vh);
}

//var pastcomp = adcomp(a_past, d_past).clip(geometry);
var currentcomp = adcomp(ascend, descend).clip(LI);

//Define functions to mask clouds for S2 and LS8 data
function add_time(img) {i
  return img.addBands(img.metadata('system:time_start'));
}

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
/*
var dpast = vprocess(d_past, geometry, 'recent');
var dcurrent = vprocess(d_current, geometry, 'recent');
var apast = vprocess(a_past, geometry, 'recent');
var acurrent = vprocess(a_current, geometry, 'recent');

//Map.addLayer(acurrent, {}, 'acurrent');
//Map.addLayer(dcurrent, {}, 'dcurrent');
//Map.addLayer(apast, {}, 'apast');
//Map.addLayer(dpast, {}, 'dpast');
//Map.addLayer(apast.subtract(acurrent), {}, 'adiff');
//Map.addLayer(pastcomp.subtract(currentcomp), {}, 'compdiff');
function cor(img){
  var imageA = img.select('VV');
  var imageB = img.select('VH');
  var correl = ee.Algorithms.CrossCorrelation(imageA, imageB,0,3);
  return correl.select(3);
}

*/

// This example uses the Sentinel-2 QA band to cloud mask
// the collection.  The Sentinel-2 cloud flags are less
// selective, so the collection is also pre-filtered by the
// CLOUDY_PIXEL_PERCENTAGE flag, to use only relatively
// cloud-free granule.

// Function to mask clouds using the Sentinel-2 QA band.
function maskS2clouds(image) {
  var qa = image.select('QA60');

  // Bits 10 and 11 are clouds and cirrus, respectively.
  var cloudBitMask = ee.Number(2).pow(10).int();
  var cirrusBitMask = ee.Number(2).pow(11).int();

  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudBitMask).eq(0).and(
             qa.bitwiseAnd(cirrusBitMask).eq(0));

  // Return the masked and scaled data.
  return image.updateMask(mask).divide(10000)
  .copyProperties(image, ['system:time_start']);
}

//Function to add time variables for
//note if there is one cycle per year multiply t by ncycles
var addVariables = function(image) {
  var time = image.date().difference('2000-01-01', 'year');
  return image.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
  //.addBands(ee.Image.constant(1))
  //.addBands(ee.Image.constant(time).rename('t'))
  //.addBands(ee.Image.constant(time.multiply(2*Math.PI).sin()).rename('sin'))
  //.addBands(ee.Image.constant(time.multiply(2*Math.PI).cos()).rename('cos'))
  .addBands(image.normalizedDifference(['B8', 'B4']).rename('ndvi'))
  .addBands(image.normalizedDifference(['B2', 'B11']).divide(image.select(['B8'])).rename('ndsi'))
  .float();
  
};

// Map the function over one year of data and take the median.
var collection = S2.filterDate(today.advance(-1, 'year'), today)
                  // Pre-filter to get less cloudy granules.
                  .filterBounds(LI)
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .map(maskS2clouds)
                  .map(addVariables);

var first = ee.Image(collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()).clip(LI);
Map.addLayer(first, {bands:['B4', 'B3', 'B2'], min:0.02, max:0.15}, 'S2');

var months = ee.List.sequence(1,12,3).map(function(month){
  return collection.filter(ee.Filter.calendarRange(month, ee.Number(month).add(2), 'month')).median();
});

function combineBands(image, result){
  return ee.Image(result).addBands(image);
}

var empty = ee.Image().select();

var composite = ee.Image(months.iterate(combineBands, empty));

// Display the results.
var data = composite.addBands(currentcomp);//.clip(geometry);
//Map.addLayer(data, {}, 'data');
Map.addLayer(currentcomp, {bands:['VV', 'VH', 'VV'], min: -15, max: -3}, 'S1');
var nlcd = NLCD.select('landcover');//.clip(geometry);
//Map.addLayer(nlcd, {}, 'nlcd');

nlcd = nlcd.remap({
  from: [41, 42, 43, 81, 82, 21, 11],
  to: [3, 3, 3, 4, 4, 4, 5],
  bandName: 'landcover'
}).rename(['landcover']).addBands(data);
print(nlcd.bandNames());

Map.addLayer(nlcd, {}, 'ncld');
//nat = nlcd.where(nlcd.eq(41).or(nlcd.eq(42)).or(nlcd.eq(43)), ee.Image(3));
//bare = nlcd.where(nlcd.eq(81).or(nlcd.eq(82)), ee.Image(4));

var imperv_ft = data.clip(geometry).sampleRegions({
  collection: imperv,
  properties: ['landcover'],
  tileScale: 16,
  scale: 30
  });
  
var nat_ft = nlcd.stratifiedSample({
  numPoints: 5000,
  classBand: 'landcover',
  region: geometry,
  tileScale: 16,
  scale: 100
});


//Map.addLayer(composite, {bands: ['B3', 'B2', 'B1'], min: 0, max: 0.3});
print("natural features:", nat_ft.size());
print("impervious features:", imperv_ft.size());
var ft = nat_ft.merge(imperv_ft).randomColumn();

var validation = ft.filter(ee.Filter.lt('random', 0.3));
var training = ft.filter(ee.Filter.gte('random', 0.7));

//tileScale explanation: specifies how big a tile to use to transfer data,
//will slow things down, but avoid memory error

//print(training.reduceColumns(ee.Reducer.frequencyHistogram(), ['landcover']));

var classifier = ee.Classifier.randomForest(20).train(training, 'landcover', data.bandNames());

//print(classifier.confusionMatrix().accuracy());

var holdout = validation.classify(classifier).errorMatrix('landcover', 'classification');
print(holdout.accuracy());

var result = data.clip(LI).classify(classifier);
Map.addLayer(result, {bands:'classification', palette:['black', 'white', 'green', 'yellow', 'blue'], min:1, max:5}, 'result');

var segments = result.select('classification').eq(1)
.where(rds.eq(3), 0)
.focal_min(1, 'square', 'pixels', 2);//.focal_max(1, 'square', 'pixels', 2);//.focal_min(1, 'square', 'pixels');
//var segments = ee.Algorithms.Image.Segmentation.SNIC(result, 10, 10, 4);
Map.addLayer(output, {}, 'parking lots');

var bands = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12', 'ndvi', 'ndsi',
'B2_1', 'B3_1', 'B4_1', 'B8_1', 'B11_1', 'B12_1', 'ndvi_1', 'ndsi_1',
'B2_2', 'B3_2', 'B4_2', 'B8_2', 'B11_2', 'B12_2', 'ndvi_2', 'ndsi_2',
'B2_3', 'B3_3', 'B4_3', 'B8_3', 'B11_3', 'B12_3', 'ndvi_3', 'ndsi_3', 'VV', 'VH', 'landcover'];

var outputFeatures = Array.from(bands);

print(outputFeatures);
var link = '91ebc2da1d448a5cc0008d9d1ce02f41';
var train_desc = 'tf_plot_train_' + link;
var test_desc = 'tf_plot_test_' + link;

Export.table.toDrive({
  collection: training, 
  description: train_desc, 
  fileFormat: 'TFRecord', 
  selectors: outputFeatures
});

Export.table.toDrive({
  collection: validation, 
  description: test_desc, 
  fileFormat: 'TFRecord', 
  selectors: outputFeatures
});


// Export the image to TFRecord format.  Note:
// print(ee.Image(1).reduceRegion('count', exportRegion, 30)); // 5620989

var image_desc = 'tf_plot_image_' + link;

// Already exported
// Export.image.toCloudStorage({
//   image: image.select(bands),
//   description: image_desc,
//   scale: 30,
//   fileFormat: 'TFRecord',
//   bucket: 'nclinton-training-temp', 
//   region: exportRegion,
//   formatOptions: {
//     'patchDimensions': [256, 256],
//     maxFileSize: 104857600,
//     compressed: true,
//   },
// });

var polys = segments.reduceToVectors({
  geometry: outbnd,
  scale: 10,
  eightConnected: true,
  labelProperty: 'class',
  maxPixels: 1e11,
  geometryType: 'polygon'
}).filter(ee.Filter.eq('class', 1));

Map.addLayer(polys);

Export.table.toAsset({
  collection: polys,
  description: "LI_plots_85a2518c78a1b01856574f802998e359",
})

Export.image.toDrive({
  image: result,
  description: "LI_class_c5cd47e574915f2d17af56300be04e6d",
  region: result.geometry(),
  scale:10,
  maxPixels: 1e11
});
