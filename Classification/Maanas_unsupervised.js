//Define Imports
// Area of interest around Maanas park
var geometry = 
    /* color: #d63000 */
    /* shown: false */
    ee.Geometry.Polygon(
        [[[90.54414589375006, 26.815995087871755],
          [90.5468924757813, 26.785963735625458],
          [90.59152443378912, 26.75101930485026],
          [90.57847816914068, 26.724650605025396],
          [90.56749184101568, 26.653485968662928],
          [90.8819754835938, 26.63998421664039],
          [90.8984549757813, 26.634460312526723],
          [90.91356117695318, 26.627094691558753],
          [90.969179463086, 26.62218401382214],
          [90.99389870136724, 26.64428040181993],
          [91.00969154804693, 26.636915413999333],
          [91.02823097675787, 26.663918047034365],
          [91.04265053242193, 26.673122030060668],
          [91.06393654316412, 26.67005411823117],
          [91.10032875507818, 26.706863612328252],
          [91.14084084003912, 26.701342946146017],
          [91.155947041211, 26.74427440545754],
          [91.16418678730474, 26.74611396316001],
          [91.17654640644537, 26.75960314343375],
          [91.23559792011724, 26.751632457678262],
          [91.24658586506746, 26.780135599830718],
          [91.26512367695318, 26.806802661323157],
          [91.23937586086004, 26.815405375760232],
          [91.15191341107595, 26.814193902667654],
          [91.10152534960662, 26.82275384543421],
          [90.88539952331928, 26.81394970199623]]])

//Bring in Sentinel-2 data
var S2 = ee.ImageCollection("COPERNICUS/S2");

//Load modules for masking clouds and modeling phenology
var Phenology = require('users/defendersofwildlifeGIS/Modules:Phenology')
var Clouds = require('users/defendersofwildlifeGIS/Modules:Clouds')

var end = '2019-11-30'
var start = '2018-11-30'
var collection = S2.filterBounds(geometry).filterDate(start, end)

function addVariables(img){
  var nir = img.select('B8')
  var red = img.select('B4')
  var blue = img.select('B2')
  img = img.addBands(red.add(blue).rename('B4B2'))
  var rgb = img.select(['B4', 'B3', 'B2'])
  var light = rgb.reduce(ee.Reducer.max()).add(rgb.reduce(ee.Reducer.min())).divide(2).rename('light')
  var ndvi = img.normalizedDifference(['B8', 'B4']).rename('ndvi')
  var ndwi = img.normalizedDifference(['B3', 'B8']).rename('ndwi')
  //var evi2 = nir.subtract(red).divide(nir.add(red.multiply(2.4)).add(1)).multiply(2.5)
  var bi = img.normalizedDifference(['B4B2', 'B3']).rename('bi')
  var ndwi2 = img.normalizedDifference(['B8', 'B11']).rename('ndwi2')
  return Phenology.addHarmonics(img.addBands(ndvi).addBands(ndwi).addBands(bi).addBands(ndwi2).addBands(light), 1)
}

var bands = ['ndvi', 'ndwi', 'bi', 'light']

var varsCollection = collection.map(addVariables).map(Clouds.basicQA);

var winter = varsCollection.filterDate('2018-12-01', '2019-02-28').select(bands).median()
var summer = varsCollection.filterDate('2019-03-01', '2019-05-31').select(bands).median()
var monsoon = varsCollection.filterDate('2019-06-01', '2019-09-30').select(bands).median()
var postmonsoon = varsCollection.filterDate('2019-10-01', '2019-11-30').select(bands).median()
var seasons = winter.addBands(summer).addBands(monsoon).addBands(postmonsoon)

print(varsCollection.size())

//Output of linear regression is a 2-band image
var model = varsCollection.select(['offset', 'time', 'cos', 'sin', 'ndvi', 'ndwi', 'bi', 'light'])
  .reduce(ee.Reducer.linearRegression(4, 4));

//These coefficeints are 2d Nx by Ny
var coeff = model.select('coefficients').toArray()

var ximage = coeff.arraySlice(0, 1, 2).arrayTranspose(0,1).arrayProject([0]).arrayFlatten([['x_ndvi', 'x_ndwi', 'x_bi', 'x_light']])
var cosimage = coeff.arraySlice(0, 2, 3).arrayTranspose(0,1).arrayProject([0]).arrayFlatten([['sin_ndvi', 'sin_ndwi', 'sin_bi', 'sin_light']])
var sinimage = coeff.arraySlice(0, 3, 4).arrayTranspose(0,1).arrayProject([0]).arrayFlatten([['cos_ndvi', 'cos_ndwi', 'cos_bi', 'cos_light']])
var image = ximage.addBands(cosimage).addBands(sinimage).addBands(seasons).clip(geometry)

var training = image.sample({
  region: geometry,
  scale: 10,
  numPixels: 10000,
  tileScale: 12
})
var clusterer = ee.Clusterer.wekaXMeans(2, 15).train(training)
var result = image.cluster(clusterer, 'class')
var output = result.focal_mode(1, 'square', 'pixels')
var vizParams = {palette: ['00A600','1DB000','3EBB00','63C600','8BD000','B6DB00','E6E600','E7CE1D','f2f2f2','003dea','dedede','edb694','efc2b3','f1d6d3','f2d12f'], min:0 , max:14}
Map.addLayer(varsCollection.median(), {bands:['B4', 'B3', 'B2'], min:250, max:2500}, 'image')
Map.addLayer(output, vizParams, 'class')

var vector = output.reduceToVectors({
  reducer: ee.Reducer.first(),
  geometry: geometry,
  scale: 10,
  maxPixels: 1e13,
  tileScale: 12
})

Export.table.toDrive({
  collection: vector,
  description: 'Maanas_unsupervised',
  fileFormat: 'KML'
})
