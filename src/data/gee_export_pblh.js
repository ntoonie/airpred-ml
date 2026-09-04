// AIRPRED - Export MERRA-2 PBLH (Planetary Boundary Layer Height) for all 10
// NCR cities. Same pattern as gee_export_merra2.js, different collection.
//
// Purpose: fill the ~6-month boundary_layer_height gap (Jan-Jun 2024) in the
// ERA5 pull. PBLH covers 1980-2026 continuously, so this window has real data.
//
// NOTE (implementation guide / chat findings): this collection's native grid
// (~69km x 55km) is even coarser than the aerosol collection's ~50km grid, so
// expect PBLH to also come back nearly/exactly identical across all 10 NCR
// cities, same as the PM2.5 branch already does. This does not block using it
// as a gap-filler -- it just means the filled period won't carry per-city BLH
// variation the way the rest of the ERA5 record does. Document this explicitly.

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

// Pull a WIDER date range than just the gap -- we need overlap with ERA5's
// good months too, to measure the systematic bias between the two sources
// before trusting a direct substitution (see preprocessing.py compare step).
var collection = ee.ImageCollection('NASA/GSFC/MERRA/flx/2')
  .filterDate('2022-08-01', '2025-01-01')
  .select(['PBLH']);

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
      .set('city', cityName);
  });

  Export.table.toDrive({
    collection: ee.FeatureCollection(cityData),
    description: 'MERRA2_PBLH_' + cityName,
    folder: 'AIRPRED_MERRA2_PBLH',
    fileNamePrefix: 'MERRA2_PBLH_' + cityName,
    fileFormat: 'CSV'
  });
});

print('PBLH export tasks created for all 10 cities. Go to the Tasks tab to run them.');