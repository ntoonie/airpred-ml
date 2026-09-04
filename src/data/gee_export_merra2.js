// AIRPRED - Export MERRA-2 aerosol diagnostics for all 10 NCR cities.
// Run this in the Google Earth Engine Code Editor (code.earthengine.google.com),
// then go to the Tasks tab and Run each of the 10 export tasks.
//
// *** RESOLVE THE FORMULA QUESTION BEFORE RUNNING THIS (implementation guide
// Section 7): the thesis's stated formula uses SO4CMASS (column mass density,
// kg/m2). NASA GMAO's published surface-PM2.5 formula uses SO4SMASS (surface
// mass concentration, kg/m3) instead. Pick the band list below accordingly. ***

var cities = {
  'Manila': [120.9842, 14.5995],
  'Quezon_City': [121.0437, 14.6760],
  'Caloocan': [120.9663, 14.6760],
  'Valenzuela': [120.9830, 14.7011],
  'Pasig': [121.0851, 14.5764],
  'Makati': [121.0244, 14.5547],
  'Mandaluyong': [121.0359, 14.5794],
  'Navotas': [120.9417, 14.6667],
  'Pasay': [120.9972, 14.5378],
  'San_Juan': [121.0333, 14.6000]
};

// EDIT this band list once the Section 7 formula question is resolved:
//   thesis-as-written: ['BCSMASS', 'OCSMASS', 'SO4CMASS', 'DUSMASS25', 'SSSMASS25']
//   NASA GMAO formula: ['BCSMASS', 'OCSMASS', 'SO4SMASS', 'DUSMASS25', 'SSSMASS25']
var BANDS = ['BCSMASS', 'OCSMASS', 'SO4CMASS', 'SO4SMASS', 'DUSMASS25', 'SSSMASS25'];

var collection = ee.ImageCollection('NASA/GSFC/MERRA/aer/2')
  .filterDate('2022-08-01', '2025-01-01')
  .select(BANDS);

Object.keys(cities).forEach(function (cityName) {
  var coords = cities[cityName];
  var point = ee.Geometry.Point(coords);
  var cityData = collection.map(function (image) {
    var value = image.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: point,
      scale: 50000
    });
    return ee.Feature(null, value)
      .set('datetime', image.date().format('YYYY-MM-dd HH:mm:ss'))
      .set('city', cityName)
      .set('latitude', coords[1])
      .set('longitude', coords[0]);
  });

  Export.table.toDrive({
    collection: ee.FeatureCollection(cityData),
    description: 'MERRA2_PM25_' + cityName,
    folder: 'AIRPRED_MERRA2',
    fileNamePrefix: 'MERRA2_' + cityName,
    fileFormat: 'CSV'
  });
});

print('Export tasks created for all 10 cities. Go to the Tasks tab to run them.');
