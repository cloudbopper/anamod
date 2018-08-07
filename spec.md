# Specifications

## Inputs
- Trained model
- Hierarchy over features

## Outputs
- CSV, listing for all nodes in hierarchy:
    - Accuracy of model with given node perturbed
    - p-values of paired statistical test comparing perturbed model loss to baseline (unperturbed) model loss

## Client API
- The client must create a model object that implements the following method:

  - **model.predict(input_instance, target_label)**
  
    (returns a tuple (loss, prediction) representing the model's output loss and prediction
    for the given (input_instance, target_label) pair)
    
- and a hierarchy whose nodes implement the following methods:
  - **node.is_leaf** (returns true if node is a leaf node, else false)
  - **node.children** (returns list of children if internal node, else empty list)
- and pass them to mihifepe using the binding functions:
  - **mihifepe.bind_model(model)**
  - **mihifepe.bind_hierarchy(hierarchy_root)**
