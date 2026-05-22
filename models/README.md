# Model Artefact Versioning

The loan approval app uses a versioned model package.

A model version includes:

- the trained model file
- the preprocessing scaler
- the label encoders
- metadata describing the model version

The `.pkl` preprocessing files are versioned with the model because predictions depend on the scaler and encoders matching the model used during training.

Current production model version:

```text
baseline-model