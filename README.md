# kaggle-quora-insincere-questions
Kaggle Quora Insincere Questions Classification

## Ideas
- [x] Train on the entire train set

To do ensemble of multiple different models, waiting for k-fold training is not reasonable.

- [x] Optimize for loss/f1

F1 is dependent on a threshold. Rather, optimize the model for loss is very straight forward. Test both.

- [x] Balanced sampling

The data set is very skewed. Balanced sampling would help. You can gradually increase the ratio of
positive samples. Try both static/dynamic sampling ratio scheduling.

- [ ] Cyclic learning rate

For faster convergence

- [x] Ensemble of embeddings

Glove, Paragram, fastText embeddings are ready. Try both (1) average of embeddings and (2) ensemble of
models with different embeddings.

- [ ] Transformer

Of course, do it. Refer to - https://qiita.com/halhorn/items/c91497522be27bde17ce

- [ ] Other NLP features - number of words

Might be effective.

- [x] Pre-processing of text

Try this kernel - https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing

- [ ] Pseudo-labeling

Test data is small, but the 2nd stage data set is fairly large enough to carry on pseudo-labeling.

- [ ] Snapshot Ensemble

0.003 for one epoch, 5 snapshots over next 3 epochs (6000 steps / 5 cycles = 1200 steps each).
For each snapshot, compute optimized threshold. Take average of each threshold over 5-folds and
use them for test with the same number of snapshots.

- [ ] Majority Voting

Ensemble technique for threshold-needed metrics like F1. Compute binary labels for each model
and just take average and threshold with 0.5. Make sure you have odd number of models.


## Logs

Evaluation logs are stored in Google Spread Sheets.

### Main Sheet

| Date | Eval Type | Script Name | Model Name | Validation F1 Majority | Validation F1 Optimized | Threshold |
| --- | --- | --- | --- | --- | --- | --- |
| 20190101-00:00:00 | Snapshot | 000_baseline | StackedRNN | 0.688 | 0.689 | 0.38 |
| 20190101-00:00:00 | Epoch | 000_baseline | StackedRNN | 0.688 | 0.689 | 0.38 |

### Snapshot Sheet

| Date | Eval Type | Tag | Script Name | Model Name | Epoch Info | Epoch Info | ... |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 20190101-00:00:00 | Snapshot | Score | 000_baseline | StackedRNN | 0.688 | 0.689 | ... |
| 20190101-00:00:00 | Snapshot | Threshold | 000_baseline | StackedRNN | 0.38 | 0.39 | ... |
| 20190101-00:00:00 | Epoch | Score | 000_baseline | StackedRNN | 0.688 | 0.689 | ... |
| 20190101-00:00:00 | Epoch | Threshold | 000_baseline | StackedRNN | 0.38 | 0.39 | ... |