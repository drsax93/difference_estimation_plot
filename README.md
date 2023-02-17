## Difference Estimation Plot

Customised version of difference estimation plot - inspired by the DABEST package by Acclab.
It performs much faster because it doesn't produce the ordered swarmplots but plots a density-informed scatterplot instead.

Basic usage:

```
import difference_estimation_polot as dpl
data_ is a dict or pd.DataFrame
fig, stats, p = dpl.estimation_plot(data_)
```

See the notebook or the `estimation_plot()` info and docs for further examples.