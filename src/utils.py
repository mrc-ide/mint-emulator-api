import torch
from models import FFNN, BiRNN, LSTM, GRU

def get_emulator_model(label="LSTM"):
    model_info = {
        "LSTM": {"path": "../trained_models/MINT_LSTMmodel.pth", "model_class": LSTM},
        "GRU": {"path": "../trained_models/MINT_GRUmodel.pth", "model_class": GRU},
        "BiRNN": {"path": "../trained_models/MINT_BiRNNmodel.pth", "model_class": BiRNN},
        "FFNN": {"path": "../trained_models/MINT_FFNNmodel.pth", "model_class": FFNN},
    }
    model_spec = model_info.get(label)
    if model_spec:
        # Instantiate model
        model = model_spec["model_class"]()
        # Load trained weights
        model.load_state_dict(torch.load(model_spec["path"]))
        return model
    return "Model not found - ensure label is one of { LSTM, GRU, FFNN, BiRNN }"


def validate_inputs(inputs):
    params = [
        "bitingPeople",
        "bitingIndoors",
        "seasonality",
        "currentPrevalence",
        "levelOfResistance",
        "sprayInput",
        "netUse",
        "irsUse",
        "emulatorModel"
    ]

    # Set difference - TODO also validate that inputs are correct data type
    if len(set(params).difference(inputs.keys())) != 0:
        return False

    return True


def percent_to_float(percent):
    """
    Convert percent string to float
    :param percent:
    :return:
    """
    return float(percent.strip("%")) / 100


def format_inputs(raw_inputs):
    """
    Convert raw inputs from web application for use in emulator.
    :param raw_inputs:
    :return formatted_inputs:
    """

    # Note the order of inputs required for the emulator:
    # 'bitingPeople_high', 'bitingPeople_low', 'bitingIndoors_high',
    # 'bitingIndoors_low', 'seasonality_perennial', 'seasonality_seasonal',
    # 'intervention_irs', 'intervention_irs-llin',
    # 'intervention_irs-llin-pbo', 'intervention_irs-pyrrole-pbo',
    # 'intervention_llin', 'intervention_llin-pbo', 'intervention_none',
    # 'intervention_pyrrole-pbo', 'currentPrevalence', 'levelOfResistance',
    # 'itnUsage', 'sprayInput', 'netUse', 'irsUse'

    formatted_inputs = {}

    # First handle string -> float variables
    current_prevalence = percent_to_float(raw_inputs['currentPrevalence'])
    level_of_resistance = percent_to_float(raw_inputs['levelOfResistance'])
    net_use = percent_to_float(raw_inputs['netUse'])
    itn_usage = percent_to_float(raw_inputs['itnUsage'])
    spray_input = percent_to_float(raw_inputs['sprayInput'])
    irs_use = percent_to_float(raw_inputs['irsUse'])
    numerical_inputs = torch.tensor([current_prevalence, level_of_resistance, itn_usage, spray_input, net_use, irs_use], dtype=torch.float32)
    # Construct dummy variables for categorical inputs
    biting_people = [1, 0] if raw_inputs["bitingPeople"] == "high" else [0, 1]
    biting_indoors = [1, 0] if raw_inputs["bitingIndoors"] == "high" else [0, 1]
    seasonality = [1, 0] if raw_inputs["seasonality"] == "perennial" else [0, 1]
    categorical_inputs = torch.tensor(biting_people + biting_indoors + seasonality, dtype=torch.float32)

    # Vectors for no intervention, pyrethroid only ITN only, pyrethroid-PBO ITN only, and pyrethroid-pyrrole ITN only
    pyrethroid_only_itn = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0], dtype=torch.float32)
    pyrethroid_pbo_itn = torch.tensor([0, 0, 0, 0, 0, 1, 0, 0], dtype=torch.float32)
    no_intervention = torch.tensor([0, 0, 0, 0, 0, 0, 1, 0], dtype=torch.float32)
    pyrethroid_pbo_pyrrole_itn = torch.tensor([0, 0, 0, 0, 0, 0, 0, 1], dtype=torch.float32)

    # Construct input sets in correct order using Pytorch tensors
    formatted_inputs['no_intervention'] = torch.hstack([categorical_inputs, no_intervention, numerical_inputs]).reshape(1, 20)
    formatted_inputs['pyrethroid_only_itn'] = torch.hstack([categorical_inputs, pyrethroid_only_itn, numerical_inputs]).reshape(1, 20)
    formatted_inputs['pyrethroid_pbo_itn'] = torch.hstack([categorical_inputs, pyrethroid_pbo_itn, numerical_inputs]).reshape(1, 20)
    formatted_inputs['pyrethroid_pbo_pyrrole_itn'] = torch.hstack([categorical_inputs, pyrethroid_pbo_pyrrole_itn, numerical_inputs]).reshape(1, 20)
    return formatted_inputs



HTTP_BAD_REQUEST = 400
