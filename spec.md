# Specifications

## Inputs
- Trained model
- Hierarchy over features

## Outputs
- CSV, listing for all nodes in hierarchy:
    - Accuracy of model with given node perturbed
    - p-values of paired statistical test comparing perturbed model loss to baseline (unperturbed) model loss

## Client API
- Client must create an object (say 'model') that implements the following calls:
  <pre> test_instance(feature_vector) </pre>
- and pass it to mihifepe using the bind_model function:
  <pre> bind_model(model) </pre>
