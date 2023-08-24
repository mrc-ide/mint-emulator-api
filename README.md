# mint-emulator-api
A simple flask API to interact with trained emulator models for MINT. 

## Installation and running in debug mode
First, install the dependencies using `pip install -r requirements.txt`, then run the development server using the
command `python -m flask --app api --debug run` from the `src/` folder. The API should now be exposed on `localhost:5000`
and can be interacted with using Postman or `cURL`.

## API structure
The API current contains one route `/get_prevalence`, which requires a `POST` request with a JSON body of input parameters.
It then validates the inputs, converts them into tensor format for use in Pytorch, and evaluates the selected emulator
against the inputs to return prevalence time series for the following intervention sets: No intervention, Pyrethroid-only ITN, 
Pyrethroid-pyrrole ITN, and Pyrethroid-PBO ITN. The trained models are saved in the `trained_models/` directory, and exposes
the feed-forward neural network (FFNN), bidirectional recurrent neural network (BiRNN), gated recurrent unit (GRU), and
long short-term memory (LSTM) architectures for use.

### Input JSON
The input parameters are described below, and should adhere to this format for correct processing. Values in brackets
refer to the valid values for the given parameter.
```
{
    "bitingPeople": <"high", "low">,
    "bitingIndoors": <"high", "low">,
    "seasonality": <"seasonal", "perennial">,
    "currentPrevalence": <"0%" to "100%">,
    "levelOfResistance": <"0%" to "100%">,
    "sprayInput": <"0%" to "100%">,
    "netUse": <"0%" to "100%">,
    "irsUse": <"0%" to "100%">,
    "emulatorModel": <"LSTM", "BiRNN", "GRU", "FFNN">
}
```
### Response/Output JSON
The response JSON is structured with a nested dictionary of the prevalence time series of each intervention type, alongside
some metadata.
```
{
    "emulator": true,
    "emulator_type": <"LSTM", "BiRNN", "GRU", "FFNN">,
    "no_intervention": [[...prevalence_time_series (61 values)...]],
    "pyrethroid_only_itn": [[...prevalence_time_series (61 values)...]],
    "pyrethroid_pbo_itn": [[...prevalence_time_series (61 values)...]],
    "pyrethroid_pbo_pyrrole_itn": [[...prevalence_time_series (61 values)...]]
}
```
