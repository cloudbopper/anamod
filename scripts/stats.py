import cloudpickle as cp
model_wrapper = cp.load(open("model_wrapper.cpkl", "rb"))
model = model_wrapper.ground_truth_model
print("%0.5f, %0.5f" % (model._aggregator._means[0], model._aggregator._stds[0]))