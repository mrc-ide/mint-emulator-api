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
        "irsUse"
    ]

    # Set difference - TODO also validate that inputs are correct data type
    if len(set(params).difference(inputs.keys())) != 0:
        return False

    return True
