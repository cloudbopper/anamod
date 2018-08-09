Specifications
==============

Inputs
------
- Test data
- Trained model
- Hierarchy over features

Outputs
-------
- CSV, listing for all nodes in hierarchy:
    - Accuracy of model with given node perturbed
    - p-values of paired statistical test comparing perturbed model loss to baseline (unperturbed) model loss

Client API
----------

### Data types
The library deals with models that accept one or both of the following types of input:
- **Static input**: Represented as a single vector per instance of length *L*
- **Dynamic input**: Represented as an input sequence of variable length *V*, each element of which is a vector of fixed length *W*.

Models commonly take only static input, but models such as Recurrent Neural Networks (RNNs) work with input sequences. Models comprising bigger networks with RNN sub-networks may take both kinds of inputs.

### Test Data
[TODO]

### Trained model
The client must create a model object that implements the following method:

    model.predict(target, input_vector=None, input_sequence=None)
which must return a tuple (loss, prediction) representing the model's output loss and prediction
for the given (target, instance) pair.

The test data type must match the data type of the *predict* function (e.g. if the model requires both static and dynamic input, the input test data must provide both for every instance).

### Hierarchy over features
[REVISE] The client must provide a hierarchy whose nodes implement the following methods:

    node.is_leaf (returns true if node is a leaf node, else false)
    node.children (returns list of children if internal node, else empty list)

### Binding interface
These binding functions must be invoked to pass the created objects to the library:

    mihifepe.bind_model(model)
    mihifepe.bind_hierarchy(hierarchy_root)

Examples
--------
[TODO]
