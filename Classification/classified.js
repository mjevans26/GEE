// Import the required modules
var Phenology = require('users/defendersofwildlifeGIS/Modules:Phenology')
var Clouds = require('users/defendersofwildlifeGIS/Modules:Clouds')
var Terrain = require('users/defendersofwildlifeGIS/Modules:Terrain')
var Calibrate = require('users/defendersofwildlifeGIS/Modules:Calibration')
var MAD = require('users/defendersofwildlifeGIS/Modules:MAD')
var MAD = require('users/defendersofwildlifeGIS/Modules:MAD')

//Define adjusted MAD functions for calibration
function canon(cor, before, after, length){
  var labels = ee.List.sequence(1, length).map(function(item){
    return ee.String("V").cat(ee.Number(item).toInt().format());
  });
  var decomp = cor.matrixSingularValueDecomposition();
  //create n * min(n, m) matrix of canonical covariates a
  var U = ee.Array(decomp.get("U"));
  //create m * min(n, m) matrix of canonical covariates b
  var V = ee.Array(decomp.get("V"));
  //get diagonal elements of SVD equivalent to CCA correlations
  var S = ee.Array(decomp.get("S")).matrixDiagonal();
  //turn images into array images with 1 x nbands matrices at each pixel
  var before2D = before.toArray().toArray(1).arrayTranspose(); 
  var after2D = after.toArray().toArray(1).arrayTranspose();
  var a = before2D.matrixMultiply(ee.Image(U)).arrayProject([1]).arrayFlatten([labels]);
  var b = after2D.matrixMultiply(ee.Image(V)).arrayProject([1]).arrayFlatten([labels]);
  return ee.Dictionary({'a':a, 'b':b});
}
function mad(before, after, aoi){
  var length = before.bandNames().length().min(after.bandNames().length());
  var corMat = MAD.corrmat(before, after, aoi);
  var cca = canon(corMat, before, after, length);
//  var cvs = ee.Image(cca.get('img'));
//  var ccachi = chisq(cvs, aoi, 100);
//  var rhos = cca.get('rhos');
  return cca;
  //return ee.Dictionary({'img': cvs.addBands(ccachi), 'rhos':rhos});
}

// Define visualization palettes for the different classes
var wwfpalette = ['f2f2f2', 'f2f2f2', '003dea', 'e7ce1d', 'e7ce1d', 'ab6c1d', '00adff', 'e7ce1d', 'e7ce1d', 'e7ce1d', 'e7ce1d', 'b6db00', 'b6db00', '3ebb00', '008700']
var palette = ['f2f2f2','00A600','003dea','B6DB00','E6E600','E7CE1D','1DB000','3EBB00','63C600','8BD000','dedede','edb694','efc2b3','f1d6d3','f2d12f']
var valpalette = ['dac041', '9f591d','c6c6c6', 'eaffce', 'ff8d00', 'caff29', 'd7ffc6', '0008ff', '77d448', '559733', '8500ff', '02bdc6']  

// Second round of ground truth data had columns 'Name', 'Lat', "long'

var legend= ee.Dictionary({
  'Agriculture': 'dac041',
  'Degraded / fallow land': '9f591d',
  'Dry river bed': 'c6c6c6',
  'Eastern_Wet_alluvial_grasslands': 'eaffce',
  'Grassland with invasives': 'ff8d00',
  'Grassland with woodland succession':'caff29', 
  'Lower_alluvial_savannah_woodlands':'d7ffc6',
  'River / water':'0008ff',
  'Sub tropical mixed moist deciduous forests':'77d448',
  'Sub tropical semi evergreen forests':'559733',
  'Swamp_areas':'8500ff',
  'Wetland':'02bdc6'
})

var names = ['class 0', 'class 1', 'water', 'class 3', 'class 4', 'class 5', 'class 6', 'class 7', 'class 8', 'class 9', 'class 10', 'class 11', 'class 12', 'class 13', 'class 14']
var vizParams = {'palette': wwfpalette, 'min':0, 'max':14}

var data = table.merge(table2).merge(table3).filter(ee.Filter.inList('class', ['Settlement', 'Savanna with invasives', 'River / water', 'Dry river bed']).not()).merge(riverbed18).merge(water18).merge(ag)
var classes = data.aggregate_array('class').distinct()
print(classes)


var training = classes.iterate(function(curr, prev){
  var i = classes.indexOf(curr)
  var subset = data.filterMetadata('class', 'equals', curr).map(function(ft){return ft.set('class_num', i)})
  return(ee.FeatureCollection(prev).merge(subset))
}, ee.FeatureCollection([]))

// Function to rescale pixels of image
function rescale(img){
  var bandNames = img.bandNames()
  var min = img.reduce(ee.Reducer.min())
  var max = img.reduce(ee.Reducer.max())
  var rescaled = img.subtract(min).divide(max.subtract(min)).rename(bandNames)
  return rescaled
}

// Function adding spectral indices to an image
function addVariables(img){
  var bandNames = img.bandNames()
  var nir = img.select('B8')
  var red = img.select('B4')
  var blue = img.select('B2')
  img = img.addBands(red.add(blue).rename('B4B2'))
  var rgb = img.select(['B4', 'B3', 'B2'])
  var evi = nir.subtract(red).divide(nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)).multiply(2.5).rename('evi')
  var light = rgb.reduce(ee.Reducer.max()).add(rgb.reduce(ee.Reducer.min())).divide(2).rename('light')
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
  var ndwi = img.normalizedDifference(['B3', 'B8']).rename('ndwi')
  //var evi2 = nir.subtract(red).divide(nir.add(red.multiply(2.4)).add(1)).multiply(2.5)
  var bi = img.normalizedDifference(['B4B2', 'B3']).rename('bi')
  var ndwi2 = img.normalizedDifference(['B8', 'B11']).rename('ndwi2')
  return img.addBands(ndvi).addBands(ndwi).addBands(bi).addBands(light).addBands(evi).set('system:time_start', img.get('system:time_start'))
}

var end = '2020-11-30'

var start = '2015-11-30'

var s2Bands = ee.List(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])

var collection = S2.filterBounds(boundary).filterDate(start, end)

var indices = ee.List(['ndvi', 'ndwi', 'bi', 'evi'])

var varsCollection = collection.map(Clouds.basicQA).select(s2Bands).map(addVariables);

var winter18 = varsCollection.filterDate('2016-12-01', '2017-02-28').select(s2Bands.add('ndvi')).qualityMosaic('ndvi')//.median()
var summer18 = varsCollection.filterDate('2016-03-01', '2016-05-31').select(s2Bands.add('ndvi')).qualityMosaic('ndvi')//.median()
var monsoon18 = varsCollection.filterDate('2016-06-01', '2016-09-30').select(s2Bands.add('ndvi')).qualityMosaic('ndvi')//.median()
var postmonsoon18 = varsCollection.filterDate('2016-10-01', '2016-11-30').select(s2Bands.add('ndvi')).qualityMosaic('ndvi')//.median()

var seasons18 = ee.Image.cat([
  winter18.select(s2Bands),
  summer18.select(s2Bands),
  monsoon18.select(s2Bands),
  postmonsoon18.select(s2Bands)
  ])

var winter19 = varsCollection.filterDate('2019-12-01', '2020-02-28').select(s2Bands.add('ndvi')).qualityMosaic('ndvi')//.median()
var summer19 = varsCollection.filterDate('2020-03-01', '2020-05-31').select(s2Bands.add('ndvi')).qualityMosaic('ndvi')//.median()
var monsoon19 = varsCollection.filterDate('2020-06-01', '2020-09-30').select(s2Bands.add('ndvi')).qualityMosaic('ndvi')//.median()
var postmonsoon19 = varsCollection.filterDate('2020-10-01', '2020-11-30').select(s2Bands.add('ndvi')).qualityMosaic('ndvi')//.median()

var seasons19 = ee.Image.cat([
  winter19.select(s2Bands),
  summer19.select(s2Bands),
  monsoon19.select(s2Bands),
  postmonsoon19.select(s2Bands)
  ])

//Try mad calibration
/*
var madcal = ee.Dictionary(mad(seasons18, seasons19, geometry))
var img19 = ee.Image(madcal.get('b'))
var img18 = ee.Image(madcal.get('a'))
*/

// get band names of composite for classification
var input_features = seasons19.bandNames()


//Try regression calibrations

//var calibrated = Calibrate.calibrate_regress(seasons19, seasons18, input_features, geometry, 100)

// Now rescale to 0-1 per pixel

seasons18 = ee.Image.cat([
  rescale(seasons18.select(input_features.slice(0,6))),
  rescale(seasons18.select(input_features.slice(6,12))),
  rescale(seasons18.select(input_features.slice(12,18))),
  rescale(seasons18.select(input_features.slice(18,24)))
  ])

seasons19 = ee.Image.cat([
  rescale(seasons19.select(input_features.slice(0,6))),
  rescale(seasons19.select(input_features.slice(6,12))),
  rescale(seasons19.select(input_features.slice(12,18))),
  rescale(seasons19.select(input_features.slice(18,24)))
  ])

Map.addLayer(postmonsoon19, {bands:['B4', 'B3', 'B2'], min: 0, max: 1250}, 'S219')
Map.addLayer(postmonsoon18, {bands:['B4', 'B3', 'B2'], min: 0, max: 1250}, 'S218')

// FEATURE ENGINEERING

// Use image segmentation to generate clusters that we can use to increase training data


var snic20 = ee.Algorithms.Image.Segmentation.SNIC({
  image: seasons19,
  size: 50
  })

// Add SNIC output cluster ids to imagery data
var img_data = seasons19.addBands(snic20.select('clusters')).clip(boundary)

//generate 5 points within each output cluster

var random = img_data.stratifiedSample({
  numPoints: 3,
  classBand: 'clusters',
  region: boundary,
  seed: 9,
  scale: 10,
  tileScale: 6
})


//Map.addLayer(snic20)

// Add the cluster membership to training features
//var features = snic20.sampleRegions({
//  collection: training,
//  properties: ['class_num'], 
//  scale: 10
//})

//print(training)
//Map.addLayer(snic19, {bands:['B4_mean', 'B3_mean', 'B2_mean']}, 'snic19')

/*
var clusters = snic20.select('clusters').reduceToVectors({
  scale:10
}).filterBounds(ee.FeatureCollection(training).geometry())

//define a spatial join to match clusters to ground-truth

var join = ee.Join.saveFirst({
  matchKey: 'training'
})

// Define a spatial filter as geometries that intersect.
var spatialFilter = ee.Filter.intersects({
  leftField: '.geo',
  rightField: '.geo',
  maxError: 10
});

var joined = join.apply(clusters, training, spatialFilter)

//print(joined)
*/

/*
In addition to the provided bands, we can experiment with different derived spectral features to be
used as predictors in classification analyses. These include:

1. Principal components
3. Harmonic coefficients
*/

// 1. PCA Transformation
/*
function get_eigens(img, bnds, aoi){
  var arrayImage = img.select(bnds).toArray()
  
  var covar = arrayImage.reduceRegion({
    reducer: ee.Reducer.covariance(),
    geometry: aoi,
    scale: 10,
    maxPixels: 1e13
  });
  
  var covarArray = ee.Array(covar.get('array'));
  
  var eigens = covarArray.eigen();
  
  //Since the eigenvalues are appended to the eigenvectors, slice the two apart and discard the eigenvectors:
  var eigenVectors = eigens.slice(1, 1);
  
  //Perform the matrix multiplication
  var pcs = ee.Image(eigenVectors)
  .matrixMultiply(arrayImage.clip(aoi).toArray(1));
  
  //Finally, convert back to a multi-band image and display the first PC
  var pcNames = ee.List.sequence(1, bnds.size()).map(function(int){
    return ee.String('pc').cat(ee.Number(int).format('%d'))
  })
  
  var pcImage = pcs
    // Throw out an an unneeded dimension, [[]] -> [].
    .arrayProject([0])
    // Make the one band array image a multi-band image, [] -> image.
    .arrayFlatten([pcNames]);
    
  return pcImage
}

var pcImage19 = get_eigens(seasons19, s2Bands, boundary)
var pcImage18 = get_eigens(seasons18, s2Bands, boundary)
//Map.addLayer(pcImage19, {'bands':['pc1', 'pc2', 'pc3']}, 'PC19');
//Map.addLayer(pcImage18, {'bands':['pc1', 'pc2', 'pc3']}, 'PC18');
*/

// 2. Harmonic Regression
/*

//Output of linear regression is a 2-band image
var model = varsCollection.select(['offset', 'time', 'cos', 'sin', 'ndvi', 'ndwi', 'bi', 'light'])
  .reduce(ee.Reducer.linearRegression(4, 4));
  
//These coefficeints are 2d Nx by Ny
var coeff = model.select('coefficients').toArray()

//resid is 1D array of length Ny  
var resid = model.select('residuals').arrayGet([0]);

var ximage = coeff.arraySlice(0, 1, 2).arrayTranspose(0,1).arrayProject([0]).arrayFlatten([['x_ndvi', 'x_ndwi', 'x_bi', 'x_light']])
var cosimage = coeff.arraySlice(0, 2, 3).arrayTranspose(0,1).arrayProject([0]).arrayFlatten([['sin_ndvi', 'sin_ndwi', 'sin_bi', 'sin_light']])
var sinimage = coeff.arraySlice(0, 3, 4).arrayTranspose(0,1).arrayProject([0]).arrayFlatten([['cos_ndvi', 'cos_ndwi', 'cos_bi', 'cos_light']])
var img = ximage.addBands(cosimage).addBands(sinimage).addBands(seasons19).clip(boundary)
*/

// SUPERVISED CLASSIFICATION
// We can run supervised classification on different sets of features.

// 1. Using PCA values


// Extract PCA values at ground truth points
var features = seasons19.sampleRegions({
  collection: training,
  properties: ['class_num'],
  scale: 10
})

print(features.size())

var features18 = seasons18.sampleRegions({
  collection: training,
  properties: ['class_num'],
  scale:10
})

Export.table.toCloudStorage({
  collection: features,
  description: 'manas_training_tfrecords',
  bucket:'cvod-203614-mlengine',
  fileNamePrefix: ''
  
})

// 2. Using scaled reflectance values from all seasons

// Extract pixel values at ground truth points

/*
var features = img_data.sampleRegions({
  collection: training,
  properties: ['class_num'],
  scale: 10
})
*/

// Filter new points & assign class id to new features based on cluster id
// Only need to do once then export
/*
var clusters = features.aggregate_array('clusters').distinct()

var new_features = random.filter(ee.Filter.inList('clusters', clusters))

var keys = features.distinct(['clusters', 'class_num'])
new_features = random.remap(keys.aggregate_array('clusters'), keys.aggregate_array('class_num'), 'clusters')

Export.table.toAsset({
  collection: new_features,
  description: 'snic-generated-training-data',
  assetId: 'snic_training_data'
})

var training_features = features.merge(new_features)
*/

// 3. Using scaled reflectance values from postmonsoon
/*
// Extract pixel values at ground truth points
var features = seasons19.sampleRegions({
  collection: training,
  properties: ['class_num'],
  scale: 10
})
*/
var classifier = ee.Classifier.smileRandomForest(20)
var trained = classifier.train({
  features: features, 
  classProperty: 'class_num',
  // use all feature properties as predictors except cluster id
  inputProperties: seasons19.bandNames()//input_features
  })

var trained18 = classifier.train({
  features: features18,
  classProperty: 'class_num',
  inputProperties: seasons18.bandNames()
})

var confusion = trained.confusionMatrix()
print(confusion)
/*
var medians = features.reduceColumns({
  reducer: ee.Reducer.median().repeat(seasons19.bandNames().size()).group(seasons19.bandNames().size(), 'class_num'),
  selectors: seasons19.bandNames().add('class_num')
})

var keys = ee.List(medians.get('groups')).map(function(dict){return ee.Dictionary(dict).get('class_num')})
*/

//var classified = ee.Image('users/defendersofwildlifeGIS/Manas/Manas_supervised_21Jan21')
var classified = seasons19.clip(boundary).classify(trained)
var classified18 = seasons18.clip(boundary).classify(trained18)

// Classify PCA images
//var classified = pcImage19.clip(boundary).classify(trained)
//var classified18 = pcImage18.clip(boundary).classify(trained)
var suppal = ['02bdc6', 'dac041', '8500ff', '9f591d', 'ff8d00', 'eaffce', 'd7ffc6', 'caff29', '559733', '77d448', 'c6c6c6', '0008ff',]

Map.addLayer(classified, {'palette': suppal, 'min':0, 'max':11}, 'supervised')
Map.addLayer(classified18, {'palette': suppal, 'min':0, 'max':11}, 'supervised18')

// Export classified image

Export.image.toAsset({
  image: classified,
  description: 'Manas_supervised20_14Feb21',
  scale:10,
  region: boundary,
  maxPixels: 1e13
})

Export.image.toAsset({
  image: classified18,
  description: 'Manas_supervised16_14Feb21',
  scale: 10,
  region: boundary,
  maxPixels: 1e13
})


// SPECTRAL MIXING APPROACH
/*
// Calculate median values of each class
// Extract pixel values at ground truth points

var features = winter19.sampleRegions({
  collection: training,
  properties: ['class'],
  scale: 10
})

//print(features)

// Returns a list of dictionaries
var medians = features.reduceColumns({
  reducer: ee.Reducer.median().repeat(winter19.bandNames().size()).group(winter19.bandNames().size(), 'class'),
  selectors: winter19.bandNames().add('class')
})


var lists = ee.List(medians.get('groups')).map(function(dict){return ee.Dictionary(dict).get('median')})
var keys = ee.List(medians.get('groups')).map(function(dict){return ee.Dictionary(dict).get('class')})
//print(keys)

var img19 = winter19.clip(geometry)
var img18 = winter18.clip(geometry)
var unmixed19 = img19.unmix(lists, true).rename(keys)
var unmixed18 = img18.unmix(lists, true).rename(keys)
//print(unmixed.bandNames())
// TO ID A CERTAIN HABITAT TYPE
var arrayImg19 = unmixed19.toArray()
var classes19 = arrayImg19.arrayArgmax().arrayFlatten([['class']])
var arrayImg18 = unmixed18.toArray()
var classes18 = arrayImg18.arrayArgmax().arrayFlatten([['class']])
Map.addLayer(classes19, {'palette': valpalette, 'min':0, 'max':11}, 'unmixing 19')
Map.addLayer(classes18, {'palette': valpalette, 'min':0, 'max':11}, 'unmixing 18')
*/

// Make Legend
function makeRow(color, name) {
       var colorBox = ui.Label({
        style: {
          backgroundColor: '#' + color,
          padding: '8px',
          margin: '0 0 4px 0'
        }
      });
       var description = ui.Label({
        value: name,
        style: {margin: '0 0 4px 6px'}
      });
       return ui.Panel({
        widgets: [colorBox, description],
        layout: ui.Panel.Layout.Flow('horizontal')
      });
}

var legendTitle = ui.Label({
  value: 'Legend',
  style: {'fontWeight':'bold'}
})
var legend = ui.Panel({
  widgets: [legendTitle],
  style: {'position': 'bottom-left'}
})

classes.evaluate(function(list){
  
  for (var i = 0; i < 12; i++) {
    var lc = list[i]
    legend.add(makeRow(suppal[i], lc))
  }

})

var subset = data.filterMetadata('class', 'equals', 'Dry river bed')
print(subset.size())
Map.addLayer(boundary)
Map.addLayer(subset, {color:'blue'}, 'fallow pts')
Map.add(legend)

//Export individual classes as polygon KML
/*
var binary = classified.eq(9)
.focal_mode(1, 'square', 'pixels')
//.focal_max(1, 'square', 'pixels')

var polys = binary.updateMask(binary)
.reduceToVectors({
  scale:10,
  geometry: boundary,
  eightConnected: true,
  maxPixels: 1e13,
  tileScale: 8
})

var date = '20Jan21'

Export.table.toDrive({
  collection: polys,
  description: 'deciduous_forest' + date,
  fileFormat: 'KML'
})
*/
// Export training data as TFRecords

Export.table.toCloudStorage({
  collection: features,
  bucket: 'cvod-203614-mlengine',
  fileNamePrefix: 'landcover/data/seasons_raw',
  fileFormat: 'TFRecord',
  selectors: ['B2', 'B2_1', 'B2_2', 'B2_3']
})

/*
// Export a test image as TFRecords
Export.image.toCloudStorage({
  image: seasons19,
  description: 'manas_test2',
  bucket: 'cvod-203614-mlengine',
  fileNamePrefix: 'landcover/data/predict/test2/test2',
  scale: 10,
  fileFormat: 'TFRecord',
  region: geometry,
  formatOptions: {
    'patchDimensions': [256, 256],
    maxFileSize: 104857600,
    compressed: true,
  },
});
*/
