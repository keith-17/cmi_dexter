# Deep learning efficiency and conditional search review

The largest avoidable search cost was treating every hyperparameter as relevant
to every model. A BERT candidate with `gru_units=32` and an otherwise identical
BERT candidate with `gru_units=128` build the same shared backbone: those GRU
settings are inactive. They should produce one cross-validation evaluation.
Changing `bert_d_model` changes the model and must remain a distinct candidate.

This review uses the supplied PDFs as reference material, not as instructions.
The books support the training and memory principles below; they do not describe
the conditional Bayesian or evolutionary implementation added to this project.

## How the search avoids wasted candidates

`canonical_candidate` in `src/multibranch_utils.py` removes inactive settings and
normalizes equivalent aliases. Candidate validation and score caching use this
effective parameter set, rather than the original flat list of knobs. Validation
runs before expensive extraction or model construction, rejecting incompatible
active choices such as a BERT width that is not divisible by its head count.

The rules follow the actual model graph. BERT still uses modality-specific branch
CNNs and fusion, so their parameters remain active. Its optional ToF
`spatial_cnn_gru` encoder also has a separate, active `tof_gru_units` setting.
Removing the shared backbone's inactive GRU settings must not remove that ToF
GRU, or discard the CNN settings used before the transformer.

### Bayesian search

`HierarchicalBayesSearchCV` compiles a flat parameter dictionary into separate
architecture spaces, including the enabled/disabled BiGRU cases. An optimizer
therefore does not spend its ordinary BERT proposals adjusting shared GRU widths.
Other conditional controls are considered together when deciding which dimensions
can be active; large controller products use conservative pruning.

For one input parameter dictionary, `n_iter` is shared across these architecture
spaces, rather than multiplied by the number of architectures. The search visits
spaces in rotation, with randomized initial order. Explicit lists retain their
per-entry budgets. Architecture-category priors do not weight this round-robin
allocation; supply explicit per-space budgets when architectures need unequal
allocation. Only distinct, preflight-valid candidates submitted to CV
consume the budget. A subsequent training failure still counts as an attempted
CV candidate; preflight cannot predict every runtime failure.

Repeated effective candidates reuse their observed scores. Invalid proposals
provide finite penalties to the optimizer without running model training.
Synthetic rejection/cache feedback is capped at 16 observations per optimizer;
an exhausted all-invalid space records just one penalty observation. This avoids
fitting a large Gaussian process merely to explain rejected points. Surrogate
history defaults to one retained model (`optimizer_kwargs['model_queue_size']`
can override it).
Proposals are sequential and use `optimizer.ask()`; this avoids the constant-liar
optimizer copies made by batched `ask(n_points=1)`. Each proposal round tries at
most 32 optimizer/random proposals. For discrete subspaces with at most 4,096 raw
coordinates, a lazy Cartesian iterator can then find an unseen candidate or prove
exhaustion without allocating the full Cartesian table. Continuous or larger
spaces stop that subspace after bounded unsuccessful retries and report the
proposal cap; this is not proof that no valid unseen candidate exists.

An all-invalid search retains a finite-penalty result row for reporting. It does
not train a model to manufacture a successful result.

The [full dependency audit](search_dependency_audit.md) goes beyond the backbone
example. Compiling the saved V1 Bayesian settings now removes **56 inactive
dimensions**: 141 supplied dimensions become 85 potentially active dimensions.
It also covers optimizer, loss, scheduling, augmentation, feature extraction,
spectral overrides and unused random-forest frame settings.

### Evolutionary search

`HierarchicalGASearchCV` draws architecture and feature controllers before their
dependent genes. Individuals contain active genes only. Following crossover or a
controller mutation, inactive genes are removed and newly active genes are drawn
from their descriptors. Switching from BERT to BiGRU therefore initializes GRU
genes instead of retaining unrelated inactive values.

A duplicate offspring triggers a fresh conditional proposal after the first
attempt, even when mutation probability is zero. After at most 32 attempts, a
small finite conditional tree can supply a remaining candidate. This enumeration
is capped at 4,096 visited tree nodes, rather than expanding every inactive axis.
When it proves the finite valid space exhausted, later generations stop and any
remaining population slots reuse cached results. Larger or continuous spaces
retain the bounded proposal fallback and cache; their exhaustion is not claimed.

## Reading the search diagnostics

| Attribute | Interpretation |
| --- | --- |
| `unique_candidate_count_` | Bayesian: distinct preflight-valid candidates evaluated by CV. Evolutionary: distinct canonical evaluation records, including a possible invalid fallback penalty record. It is not a count of individual fold fits. |
| `duplicate_proposal_count_` | Repeated coordinates or effective candidates encountered while proposing; a high count indicates limited remaining diversity or a heavily conditional space. |
| `reused_candidate_count_` | Cached Bayesian phenotype scores or evolutionary evaluation results reused without another CV fit. A repeated Bayesian coordinate detected before score lookup need not increment this count. |
| `rejected_candidate_count_`, `rejected_candidate_reasons_` | Preflight rejections and their reasons. These count rejection events, including retries, rather than necessarily distinct configurations. |
| `search_exhausted_` | A bounded finite enumeration/coordinate count proved the relevant search space exhausted. Bayesian sets this only when all compiled spaces are exhausted. |
| `proposal_limit_reached_` | Bounded retries failed in a space whose exhaustion could not be proved. Unseen valid candidates may remain. |

The result cache avoids repeating extraction and training for the same effective
candidate within one search run. It does not cache complete fitted neural models
or share learned preprocessing across CV folds.

## What the books support

Sources:

- [Advanced Deep Learning with TensorFlow 2 and Keras](<../context/Advanced Deep Learning with TensorFlow 2 and Keras (1)-compressed.pdf>).
- [Deep learning Book, notes by Shelly Sheynin](<../context/Deep learning book.pdf>).
  This is the supplied 157-page notes document, distinct from the separate
  Goodfellow textbook also present in `context/`.

PDF page numbers below are one-based. The Keras book's printed pages are 21 less
than its PDF pages. For the Sheynin notes, the second locator is the contents-page
number, not a separately verified printed footer.

| Source and verified location | Principle and application |
| --- | --- |
| Keras, PDF p32 / printed p11, Listing 1.3.2; PDF p56 / printed p35, Listing 1.5.1 | The examples cast model inputs to float32. Notebook sensor loading and extractor chunk tensors already use float32, reducing input storage relative to float64. Some signal calculations intentionally retain higher precision. |
| Keras, PDF p43 / printed p22, Optimization; PDF p428 / printed p407, SSD model training | Mini-batches bound work per training step, and batch size should fit available GPU memory. Choose batch size together with chunk length and model width. The book's particular batch sizes are examples, not universal settings. |
| Keras, PDF pp45-46 / printed pp24-25, Performance evaluation and Model summary | Larger networks need not improve generalization. Inspect model summaries and compare validation quality before expanding depth or width. |
| Keras, PDF p423 / printed p402, Data generator model in Keras | Load or construct batches when full input tensors cause memory bottlenecks. This motivates a future streaming extractor; the current estimator still extracts full tensors eagerly. |
| Sheynin, PDF p9 / contents p8, Data Preprocessing | Learn normalization statistics on training data and apply them to validation/test data. The estimator splits sequences before fitting its extractor and transforms the validation partition separately. |
| Sheynin, PDF p28 / contents p27, Layer Normalization | LayerNorm is discussed for recurrent models and transformers. The transformer blocks already use LayerNorm. |
| Sheynin, PDF p29 / contents p28, Early stopping | Use validation behavior to control training duration. Existing callbacks monitor `val_loss` and restore the best weights when internal validation is available; without it they monitor training loss. |
| Sheynin, PDF p113 / contents p112, Vision Transformers: Disadvantages; PDF p123 / contents p122, Swin Transformer | Global self-attention has quadratic cost in token count. Applied to this temporal model, the token count is the padded chunk length. |

## Remaining RAM costs

The BERT encoder applies global self-attention over the full padded chunk.
Branch convolutions preserve timestep count, and masking does not shorten the
attention tensors. Doubling chunk length can approximately quadruple the
attention-score component of memory, before accounting for layers, heads, batch
size and implementation details. Shorter chunks and smaller batches are the
available configuration controls; token downsampling or local attention would
be separate model changes requiring validation.

Training still eagerly constructs full training and validation tensors and
modality arrays. Releasing intermediate combined tensors reduces duplication,
but mini-batch Keras fitting alone does not make preprocessing stream data.
Future work could extract batches or read disk-backed chunks while preserving
training-only normalization, masks, sequence labels and whole-sequence splits.
Streaming and token reduction are recommendations, not changes implemented in
this search update.

The existing serial TensorFlow search settings also avoid simultaneous model
fits competing for GPU memory. These changes reduce redundant work; actual
end-to-end training time and GPU peak memory still require measurement on the
target runtime and dataset.

## Regression checks

Run `.\.venv\Scripts\python.exe -m unittest discover -s tests -q`.
The conditional-search tests verify that BERT never draws inactive shared-GRU
descriptors, distinct active candidates fill the available budget, mixed
architectures share one Bayesian budget, callbacks stop the whole search,
architecture/auxiliary switches restore newly active genes, finite exhaustion
terminates, and the surrogate history and invalid-proposal feedback stay bounded.
They use real search engines and small recording estimators, avoiding GPU model
training simply to verify search scheduling. Keras execution tests remain skipped
when the existing local TensorFlow installation cannot import.
