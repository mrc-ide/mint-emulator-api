from flask import Flask, request
from utils import validate_inputs, get_emulator_model, HTTP_BAD_REQUEST

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>A Flask API to interact with trained ML emulators for MINT.</p>"


@app.route("/get_prevalence", methods=["POST"])
def get_prevalence():
    """
    Main function to get prevalence time series from inputs by evaluating inputs against
    a trained emulator model. Flow:
    1. Validate inputs. Return early if any inputs are missing.
    2. Convert inputs to format for emulator (floats / dummy variables).
    3. Select emulator model as specified by user.
    4. Evaluate the emulator for No Intervention, Pyrethoid-ITN, Pyrethoid-PBO-ITN, and Pyrethoid-pyrrole ITN scenarios.
    5. Return formatted JSON to user.
    :return: prevalence time series for different intervention scenarios (JSON object)
    """
    raw_inputs = request.form
    if not validate_inputs(raw_inputs):
        return "Inputs not valid. Ensure { bitingPeople, bitingIndoors, seasonality, currentPrevalence, " \
               "levelOfResistance, sprayInput, netUse, irsUse } present in request form.", HTTP_BAD_REQUEST

    return raw_inputs
