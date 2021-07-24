from flask import Flask, request
import pandas as pd
from _collections import OrderedDict
import joblib

app = Flask(__name__)

@app.route("/api/eanalyzer")
def get() :
    noInfected = float(request.args["NoInfected"])
    nearToDistrict = float(request.args["NearToDistrict"])
    noOfPeople = float(request.args["NoOfPeople"])

    folder_name = "model/"
    file_path = (folder_name + "model.joblib")
    file = open(file_path, "rb")
    model_load = joblib.load(file)

    request_data = OrderedDict([("total_infected",noInfected),("near_district",nearToDistrict),("no_of_people",noOfPeople)])
    reshaped_data = pd.Series(request_data).values.reshape(1,-1)
    prediction = model_load.predict(reshaped_data)
    return str(prediction)

if __name__ == "__main__":
    app.run()
