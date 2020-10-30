# Debugging

import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import pearsonr
from pprint import pprint
import numpy as np
from anamod.compute_p_values import compute_empirical_p_value

num_instances = baseline_loss.shape[0]
x = list(range(num_instances))
fidx = 15  # 5

# Increase in loss upon perturbation (num_instances X num_permutations)
delta = perturbed_loss - baseline_loss[:, np.newaxis]

var1 = delta > 0  # Whether or not loss increased upon perturbation

var2 = np.sum(var1, axis=1)  # Number of permutations for each instance for which the loss increased upon perturbation

var3 = np.mean(delta, axis=1)  # Mean change in loss for every instance

# Histogram of above value. Should be approx uniform for irrelevant feature, but higher mass on right for relevant feature 
px.histogram(x=var2).show()

# Plot showing baseline losses
px.line(x=x, y=baseline_loss).show()

# Plot showing mean change in loss for each instance - note larger and more numerous positive than negative peaks
px.line(x=x, y=var3).show()

# Noise
outputs = model.predict(data)
noise = outputs - targets

# Ground truth model, temporally aggregated data
gmodel = model.ground_truth_model
agg_data = gmodel._aggregator.operate(data)

# Noise excluding feature 5 - appears to be significantly higher than regular noise and outputs
# Incorrect due to feature standardization
# data2 = np.copy(data)
# data2[:, 5, :] = 0
# outputs2 = model.predict(data2)
# noise2 = outputs2 - targets

# Feature lists
relevant_features_list = [int(x) for x in gmodel.relevant_feature_names]
relevant_features_set = set(relevant_features_list)
irrelevant_features_set = set(range(30)).difference(relevant_features_set)
irrelevant_features_list = list(irrelevant_features_set)

# Sanity check - should match targets
var5 = gmodel._polynomial_fn(agg_data.transpose(), 0)
assert np.all(np.around(var5, 10) == np.around(targets, 10))

# Noise excluding feature 5 - correct version
agg_data2 = np.copy(agg_data)
agg_data2[:, fidx] = 0
outputs2 = gmodel._polynomial_fn(agg_data2.transpose(), model.noise_multiplier)
noise2 = outputs2 - targets

# Sanity check - should match targets
agg_data3 = np.copy(agg_data)
agg_data3[:, irrelevant_features_list] = 0
outputs3 = gmodel._polynomial_fn(agg_data3.transpose(), 1)
assert np.all(np.around(outputs3, 10) == np.around(targets, 10))

# Check outputs for feature 5 only TODO
# var4 = list(irrelevant_features_set.differenc({5}))

# Sanity check - should match outputs2/noise2
agg_data4 = gmodel._aggregator.operate(data_perturbed)
agg_data4[:, fidx] = 0
outputs4 = gmodel._polynomial_fn(agg_data4.transpose(), model.noise_multiplier)
noise4 = outputs4 - targets
assert np.all(outputs4 == outputs2)

# Plot baseline targets, feature 5 outputs, other feature outputs together
fig = go.Figure()
# fig.add_trace(go.Scatter(x=x, y=targets, name="Targets"))
fig.add_trace(go.Scatter(x=x, y=noise2, name=f"Noise except {fidx}"))
agg_data5 = np.copy(agg_data)
all_except_5 = list(range(30))
all_except_5.remove(fidx)
agg_data5[:, all_except_5] = 0
outputs5 = gmodel._polynomial_fn(agg_data5.transpose(), model.noise_multiplier)
fig.add_trace(go.Scatter(x=x, y=outputs5, name=f"Noise for {fidx}"))
fig.show()
# Correlation between both noise terms
corr1 = pearsonr(noise2, outputs5)
# Inferences:
# - Compare to plot showing mean change in loss for each instance (line 27)

# - For instances where the two noise terms are nearly equal (and with same sign), loss goes down on perturbation
#   since permuting 5 regresses it towards the mean (0) i.e. away from other noise, hence decreasing the total noise magnitude

# - For instances where the two noise terms have opposite signs, the loss tends to go up on perturbation
#   since permuting 5 regresses it towards the mean (0) and hence towards the other noise, hence increasing the total noise magnitude

# - The two noise terms appear to have a significant negative correlation (-0.2),
#   hence the terms tend to be opposed, hence the loss usually goes up

############# Key observation #############
# - Note that this analysis does not rely on permutations, but it tells it where the loss is likely to go upon permutation.
#   Which seems to be the reason that the FDR does not decrease despite increasing the number of permutations
#   The baseline configuration essentially decides the average direction of change for the loss upon perturbation

# - In general, an irrelevant feature's noise terms may have the same or opposite sign as the other noise terms with equal probability (VERIFY)
#   Does this mean that the expected probability of a false positive may be 50% for a given feature? How can we address this issue?

############# Bug identified ##############
# - It appears that the RNG is being deep-copied before creating features, so features may have correlations
#   This may be the cause of the significant correlation between noise terms (such as for feature 5), resulting in false positives
#   RNG deep copying was added four months ago in change introducing per-feature aggregation functions (commit 32a49e12c01c939cc36a763c039f282d463f3b89)
#   In this simulation, features 3, 4 and 5 are correlated - they all have a MC state with identical probabilities

# Check correlations between time series for feature 5 and every other feature for a given instance
# May not be very useful since individual time series may correlate if all time series are increasing
# Correlation should be close to 0 across instances however
import code; code.interact(local=vars())
corr3 = [pearsonr(data[0, fidx, :], data[0, idx, :]) for idx in range(30)]
pprint(corr3)
# Check correlations across instances
corr4 = [pearsonr(agg_data[:, fidx], agg_data[:, idx]) for idx in range(30)]
pprint(corr4)

# Plot features 4 and 15 together, they seem to have relatively high correlation
# Note: feature 15 is a Bernoulli r.v., seems likely independent of 4
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=agg_data[:, 4], name="4"))
fig.add_trace(go.Scatter(x=x, y=agg_data[:, 15], name="15"))
fig.show()

# Time series comparisons - surprisingly high correlation
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=data[0, 4, :], name="4"))
fig.add_trace(go.Scatter(x=x, y=data[0, 15, :], name="15"))
fig.show()

from scipy.stats import pearsonr
pearsonr(data[0, 4, :], data[0, 15, :])

# Check correlation between signs of noise terms
except5 = noise2 > 0
only5 = outputs5 > 0
corr2 = pearsonr(except5, only5)

# Compute p-values using given test statistic
test_statistic = "mean_loss"
pvalues = [compute_empirical_p_value(baseline_loss, perturbed_losses[f"{idx}"], test_statistic) for idx in range(30)]
imp = set(np.where(np.array(pvalues) < 0.1)[0])
irrel = imp.difference(relevant_features_set)
rel = imp.difference(irrelevant_features_set)

# Plot showing multiple losses
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=baseline_loss, name="baseline"))
fig.add_trace(go.Scatter(x=x, y=perturbed_losses[f"{fidx}"][:, 0], name="perturbed"))
fig.show()

fig = px.scatter(x=baseline_loss, y=perturbed_losses[f"{fidx}"][:, 0], width=800, height=800)
fig.update_xaxes(range=[0, 5])
fig.update_yaxes(range=[0, 5])
fig.show()