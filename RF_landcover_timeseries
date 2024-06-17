## This code calculates a random forest land cover classification time series analysis using Landsat sensors, and imported training data/region of interest (ROI) with Google Earth Engine in JavaScript.

/////////////////////////////////////////////////////////////////////////////
//////// Preprocess Imagery and Define Region of Interest (ROI) ///////////// 
/////////////////////////////////////////////////////////////////////////////

// Add the Savannah River Basin ROI Shapefile to the map
Map.centerObject(ROI, 10);

var outline = {fillColor: '00000000',width: 2.0} // turning the ROI into an outline
Map.addLayer(ROI.style(outline), {}, 'ROI');

// Specify the start date, 1st day of Month (Year, Month, Day)
var startGrowing = ee.Date.fromYMD(2013, 5, 1);

// Specify the start date, 1st day of Month (Year, Month, Day)
var startNonGrowing = ee.Date.fromYMD(2013, 5, 1);


// Set  up the cloud masking function
function maskL8srClouds(l8) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  // Get the pixel QA band.
  var qa = l8.select('QA_PIXEL');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return l8.updateMask(mask);
}

// Scale factors function
function applyScaleFactors(l8) {
  var opticalBands = l8.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = l8.select('ST_B.*').multiply(0.00341802).add(149.0);
  return l8.addBands(opticalBands, null, true)
    .addBands(thermalBands, null, true);
}

/////////////////////////////////////////////////////////////////////////////
//////////////// Create Growing Season Composite For Loop /////////////////// 
/////////////////////////////////////////////////////////////////////////////

// Iterate over each year from the start date
for (var i = 0; i < 11; i++) {
// Calculate the end date for each month
var finishGrowing = startGrowing.advance(4, 'month');

//  Create growing season composite 
  var compositeGrowing = l8
    .filterBounds(ROI)
    .filterDate(startGrowing, finishGrowing)
    .map(maskL8srClouds)
    .map(applyScaleFactors)
    .mean()
    .clip(ROI);

// Set visualization parameters
var visParams = {
  bands: ['SR_B4', 'SR_B3', 'SR_B2'],
  min: 0,
  max: 0.2 //brightening the image a little
}

Map.addLayer(compositeGrowing, visParams,'Growing Season Composite - year' + (i + 1), false);

// Filter the NLCD collection to get the image for your region of interest
var nlcdData = NLCD
              .filterBounds(ROI)
              .first()
              .select('landcover');

// Define the urban and water class codes from NLCD
var urbanWaterCodes = [11, 21, 23, 24];

// Create a binary mask for urban and water areas
var maskUrbanWater = nlcdData.neq(urbanWaterCodes[0])
  .or(nlcdData.eq(urbanWaterCodes[1]))
  .or(nlcdData.eq(urbanWaterCodes[2]))
  .or(nlcdData.eq(urbanWaterCodes[3]))
//  .rename('urban_water_mask'); delete, I don't see what this does

// Update the composite mask with the NLCD mask
var maskedComposite = compositeGrowing.updateMask(maskUrbanWater);

Map.addLayer(maskedComposite, visParams, 'masked', false);


// Set visualization parameters for NDVI, moved it down here so the NVDI is using 
// the masked image
var visParamsNDVI = {
  min: 0,
  max: 1,
  palette: ['white','green']
};
var ndviGrowing = maskedComposite.normalizedDifference(['SR_B5','SR_B4'])

Map.addLayer(ndviGrowing, visParamsNDVI,'Yearly NDVI - year' + (i + 1), false);

// Classify training data and bands for the growing season
var label = 'Class';
var bands = ['SR_B1','SR_B2', 'SR_B3', 'SR_B4','SR_B5','SR_B6','SR_B7'];


// Merge all training feature collections
var training = Marsh.merge(Unhealthy_Vegetation).merge(Healthy_Vegetation).merge(Evergreen);
var growingSeasonTraining = maskedComposite.sampleRegions({
      collection: training, 
      properties: ['Class'], 
      scale: 30,
      tileScale: 4
    });

// Designate training data for growing season
var trainingDataGrowing = growingSeasonTraining.randomColumn();
var trainSetGrowing = trainingDataGrowing.filter(ee.Filter.lessThan('random', 0.7));
var testSetGrowing = trainingDataGrowing.filter(ee.Filter.greaterThanOrEquals('random', 0.7));

// Define palette for classification
var landcoverPalette = {
    min: 0,
    max: 3,
    palette: 
    ['5c5337', // Marsh, brown
    '8b0000', // Unhealthy Vegetation, dark red
    '316025', // Healthy Vegetation, green
    'FFD700'], // Evergreen, gold
};

// Create the classifier for the growing season
var classifierGrowing = ee.Classifier.smileRandomForest(10)
  .train({
    features: trainSetGrowing,
    classProperty: label,
    inputProperties: bands
  });

// Classify the image with the water mask for the growing season
var classifiedGrowing = maskedComposite.classify(classifierGrowing);

// Add growing season composite to the map
Map.addLayer(classifiedGrowing, landcoverPalette,'Growing Season Classification - year' + (i + 1), true);

// Update the start date for the next iteration
  startGrowing = finishGrowing;

var classificationWithNoData = classifiedGrowing.unmask(-1);

// Export the Growing Season Composite image for each year
Export.image.toDrive({
 image: classificationWithNoData,
 description: 'Growing_Season_Composite - year' + (i + 1),
 folder: 'GEE_Data',
 region: ROI,
 crs: "EPSG: 4326",
 skipEmptyTiles: true,
 scale: 30, // Resolution in meters per pixel
 formatOptions: {
 cloudOptimized: true
}});
}
  
/*Generating stats on the model
Currently this only generates states for the last version of the model,
which is for the final year. This is because your classifier is part of the loop
and gets rewritten for each year. As such, you don't have a way to compare 
the models on a year to year basis. I think this is fine because the landscape
is not so dynamic that we can't assume that the classifiers aren't the "same"
from year to year. 
*/

// Classify the test FeatureCollection.
var test = testSetGrowing.classify(classifierGrowing);// running the model on the test portion

// Print some info about the classifier.
print('RF, explained', classifierGrowing.explain());

// Print the confusion matrix.
var confusionMatrix = test.errorMatrix(label, 'classification');
print('Confusion Matrix', confusionMatrix);


//Confusion Matrix
var accuracy = confusionMatrix.accuracy()
var consumersAccuracy = confusionMatrix.consumersAccuracy()
var producersAccuracy = confusionMatrix.producersAccuracy()
var kappa = confusionMatrix.kappa()


//https://www.nateko.lu.se/sites/nateko.lu.se.sv/files/assessment_of_classification_accuracy2.pdf
print('Overall Accuracy',accuracy)
print('User Accuracy',consumersAccuracy)
print('Producer Accuracy',producersAccuracy)
//https://stats.stackexchange.com/questions/82162/cohens-kappa-in-plain-english
print('kappa',kappa)

/*
If you want to work with the regression tree in R, you have to follow these steps:
1. On the console tab in the right pane, click the JSON to right of the object
2. Click the leading curly bracket and it will select everything in that area
3. Use crtl + C to copy
4. Open notepad and paste it there
5. The file can be named anything, but you have to add .json to the end of the file name
6. Also change the file type to all files, instead of text.
7. Use the instructions in the R script to open the files

This will create a chart summarizing the total area for each class
You should find a way to loop this this so that you can get one for every year. 
Then you can do a timeseries analysis. 
*/

var areaChart = ui.Chart.image.byClass({
  image: ee.Image.pixelArea().addBands(classifiedGrowing),
  classBand: 'classification', 
  region: ROI,
  scale: 30,
  reducer: ee.Reducer.sum(),
  classLabels: ['Marsh', 'Unhealty','Healthy','Evergreen']
});

print('Area of Each Class (m^2)',areaChart)
