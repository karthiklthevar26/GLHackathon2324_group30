from flask import Flask, request, jsonify
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

app = Flask(__name__)

# Create a Spark session
spark = SparkSession.builder.master("local").appName("HousePriceApp").getOrCreate()

# Load model
model_path = "model/houseprice_model"
loaded_model = PipelineModel.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the postman request
        data = request.get_json(force=True)

        # Logging for debugging
        app.logger.info(f'Received JSON data: {data}')

        # Ensure that all required feature columns are present in the received JSON
        required_columns = ['Id','MSSubClass','LotArea','OverallQual','YearBuilt','YearRemodAdd','BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold','New','MSZoning_index','Street_index','LotShape_index','LandContour_index','Utilities_index','LotConfig_index','LandSlope_index','Neighborhood_index','Condition1_index','Condition2_index','BldgType_index','HouseStyle_index','RoofStyle_index','RoofMatl_index','Exterior1st_index','Exterior2nd_index','ExterQual_index','ExterCond_index','Foundation_index','Heating_index','HeatingQC_index','CentralAir_index','KitchenQual_index','Functional_index','PavedDrive_index','SaleType_index','SaleCondition_index','features','SalePrice']
        for column in required_columns:
            if column not in data:
                return jsonify({'error': f'No "{column}" key found in the input JSON'}), 400

        # Extract features from the JSON data
        features = [data[column] for column in required_columns]

        # Logging for debugging
        app.logger.info(f'Features: {features}')

        # Convert the features to a PySpark DataFrame with a single Row
        features_row = Row(features=Vectors.dense(features))
        features_df = spark.createDataFrame([features_row])

        # Make a prediction using the loaded model
        prediction = loaded_model.transform(features_df).select('prediction').collect()[0]['prediction']

        return jsonify({'HousePricePrediction': prediction})

    except Exception as e:
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000, debug=True)
