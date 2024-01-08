# gnn-clustering-citation-cartels

Contains node clustering experiments focusing on a co-authorship graph generated from a list of retracted Hindawi papers (discussed in [this ForBetterScience article](https://forbetterscience.com/2023/01/03/hindawi-garbage-sorting-system-based-on-citations/)). A paper's clustering is propagated to all its authors. 

The goal is to see if an end-to-end GNN+MLP pipeline can recover the provided ground-truth clusterings.

## Prelim. Results

| Method              | NMI ↑  | Conductance ↓ | Modularity ↑ |
|---------------------|--------|---------------|--------------|
| Spectral Clustering | 0.0167 | 0             | 0.0881       |
| GNN+MLP             | 0.3282 | 0.0790        | 0.6431       |

Std. devs. not reported yet.
GNN+MLP results are reported with upsampling and neutral cluster masking.
Spectral Clustering result is reported with neutral cluster masking only; does not scale well enough to support the larger upsampled graph. 