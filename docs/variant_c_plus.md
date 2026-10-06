# Variant C+ (improved AIRPRED / CMA-TCN+)

The original Variant C (`VariantC_AIRPRED`) is **unchanged**. C+ is a separate class, `VariantC_Plus`, switched on through variant keys and config flags. The original checkpoints still load and give bit-identical outputs.

## 1. What the original C does

```
PM2.5 (B,48,1)  -> TCNEncoder (4 blocks, dilations 1/2/4/8, channels 64/64/128/128) -> h_pm25 (B,48,128) -> Query
Meteorology (B,48,7) -> TCNEncoder (same design)                                      -> h_met  (B,48,128) -> Key, Value
CrossModalAttentionFusion: MHA (d_model 128, 4 heads) -> +residual -> LayerNorm -> FFN(256) -> +residual -> LayerNorm -> Linear(128->256)
ForecastHead: global average pool over time -> Linear(256->128) -> GELU -> Dropout(0.1) -> Linear(128->24)
```
641,176 parameters. No attention dropout, no dropout in the FFN, no normalisation of the TCN outputs.

## 2. What changed, and why

| # | Change | Flag | Purpose |
|---|---|---|---|
| 1 | **Multi-scale TCN readout.** Each TCN block's output is tapped, projected by a 1x1 conv to `d_model / n_taps` channels and concatenated. Block *k* has receptive field 5, 13, 29, 61 steps (K=3, 2 convs per block), so the representation mixes short-, medium- and long-range history from one network. | `multi_scale`, `ms_taps` | Temporal dependency modelling. The original only reads the last block, so it must pass short-range detail through all four blocks. Costs about 12k parameters per branch, not a duplicated TCN. |
| 2 | **LayerNorm on both branch outputs before attention.** | `input_norm` | Training stability. TCN outputs are post-ReLU (non-negative, unbounded); normalising keeps attention logits in range and puts the multi-scale taps on one scale. |
| 3 | **Configurable attention dropout and FFN dropout.** | `attention_dropout`, `ffn_dropout` | Generalisation. Fine-tuning showed the original C overfits quickly. |
| 4 | **Optional bidirectional cross-modal attention** (meteorology also queries PM2.5), streams concatenated and projected. | `bidirectional_attention` | Cross-modal interaction. Off by default; costs about 165k parameters. |
| 5 | **Optional residual forecasting** (`last PM2.5 + learned correction`). | `--residual` | Starts from persistence. Any variant can use it. |

Already present in the original and kept as is: residual connections and post-LayerNorm around attention and the FFN, and PM2.5 as Query with meteorology as Key/Value.

**Honest note:** the original fusion block already had residuals and LayerNorm, so "add residuals and normalisation" amounted to adding the branch-output LayerNorm and the dropouts. I did not add components that would only duplicate what was there.

### Architecture

```
Original:
  PM2.5  -> TCN -> Q ─┐
  Meteo  -> TCN -> K,V┴-> Cross-Modal Attention -> Forecast

Improved (CPLUS):
  PM2.5 -> Multi-Scale TCN -> LayerNorm ─┐
                                         ├-> Cross-Modal Attention -> Fusion -> Forecast
  Meteo -> Multi-Scale TCN -> LayerNorm ─┘
           (PM2.5 = Q, meteorology = K,V)

Optional (CPLUSBI): a second attention block with meteorology = Q, PM2.5 = K,V;
  both outputs concatenated -> Linear -> Forecast
```

## 3. Variant keys (ablations)

| Key | Multi-scale | LayerNorm + dropouts | Bidirectional | Meaning |
|---|---|---|---|---|
| `C` | no | no | no | Original AIRPRED |
| `CMS` | yes | no (pinned off) | no | C + multi-scale TCN |
| `CATT` | no | yes (config values) | no | C + improved attention |
| `CPLUS` | yes | yes | no (config) | C+ (multi-scale + improved attention) |
| `CPLUSBI` | yes | yes | yes | C+ with bidirectional attention |

Residual forecasting is a flag, not a variant: `C --residual` is "C + residual forecasting", `CPLUS --residual` is "full improved C+".

A key pins only the flags in the table; every other value is read from `configs/config.yaml` under `model.c_plus`.

## 4. Configuration (`configs/config.yaml`, new `model.c_plus` block)

```yaml
model:
  # ... existing keys unchanged ...
  c_plus:
    tcn_channels: [64, 64, 128, 128]
    dilations: [1, 2, 4, 8]
    kernel_size: 3
    tcn_block_dropout: 0.0
    d_model: 128
    num_attention_heads: 4
    ffn_dim: 256
    fused_dim: 256
    multi_scale: true
    ms_taps: null              # null = all blocks; e.g. [1, 3] = two scales
    input_norm: true
    attention_dropout: 0.1     # candidates 0.0 / 0.1 / 0.2
    ffn_dropout: 0.1
    bidirectional_attention: false
```
A, B, C, D ignore this block. `training.weight_decay` already existed in the config but `train.py` never used it; it is now passed to Adam (default 0.0, so old runs are unchanged).

## 5. Commands

Original C (baseline, unchanged; **this retrains and overwrites `variant_c_seed*_best.pt`**, so skip it if you already have those):
```
python train_local.py --variants C --seeds 42 123 2024 7 99
```
Improved C+:
```
python train_local.py --variants CPLUS --seeds 42 123 2024 7 99
```
Ablations (the same seeds for all, one run):
```
python train_local.py --variants C CMS CATT CPLUS --seeds 42 123 2024 7 99 --keep-checkpoints
python train_local.py --variants C --residual --seeds 42 123 2024 7 99          # C + residual
python train_local.py --variants CPLUS --residual --seeds 42 123 2024 7 99      # full improved C+
python train_local.py --variants CPLUSBI --seeds 42 123 2024 7 99               # optional bidirectional
```
Hyperparameter candidates (tag every run; the tag goes into checkpoint, CSV and JSON names):
```
python train_local.py --variants CPLUS --seeds 42 --lr 3e-4 --weight-decay 1e-5 --attn-dropout 0.2 --max-epochs 100 --patience 10 --tag _lr3e-4_wd1e-5_ad0.2
```
Fine-tuning on OpenAQ works with the new keys (checkpoint must be `variant_cplus_seed42_best.pt` etc.):
```
python finetune_openaq.py --variants CPLUS --seeds 42 123 7 --attn-dropout 0.1
```

Every `train_local.py` run writes `results/multiseed_val_rmse<tag>.csv` and `..._config.json` (all hyperparameters and parameter counts). C+ runs without a tag go to `multiseed_val_rmse_cplus.csv`, so the baseline CSV is never overwritten.

## 6. Experimental search space (candidates, not claimed optimal)

| Hyperparameter | Values |
|---|---|
| learning rate | 1e-3, 3e-4, 1e-4, 3e-5 |
| weight decay | 0, 1e-5, 1e-4 |
| attention dropout | 0, 0.1, 0.2 |
| max epochs / patience | 100 / 10 |

That is 36 combinations, so search in stages: all 36 with one seed (42), then re-run the top 3 on 5 seeds. **Choose by validation RMSE only.** `train_local.py` never reads the test split; look at test once, at the end, for the one chosen configuration of each variant.

**Fairness:** if C+ gets this search and A, B, C, D do not, the comparison is tilted. Give every variant the same lr and weight-decay grid (attention dropout applies only to the attention variants), or report C+ with the same default recipe as the baselines as well.

## 7. Parameter counts

| Variant | Parameters | vs C |
|---|---|---|
| C (original) | 641,176 | |
| CMS | 666,008 | +3.9% |
| CATT | 641,688 | +0.1% |
| CPLUS | 666,520 | +4.0% |
| CPLUSBI | 831,768 | +29.7% |

## 8. Compatibility notes

- `forward(x_pm25, x_met)` and the `(B, 24)` output are unchanged; preprocessing, datasets and evaluation are untouched.
- C+ checkpoints are ordinary state dicts, named `variant_<key>_seed<S>[tag]_best.pt`. They cannot be loaded into the original `VariantC_AIRPRED` (different layers), and `main.py` does not serve them; it still loads the original/fine-tuned A–D.
- Residual-trained checkpoints (`..._resid_best.pt`) hold base weights only; at inference they need the `ResidualForecast` wrapper (`build_variant(..., residual=True)`).
- `return_attention=True` returns the PM2.5-to-meteorology map, shape `(B, heads, L, L)`; the reverse map is available via `fusion(..., return_reverse=True)`. Code that reads `model.fusion.mha` still works.
- `--attn-dropout` in `finetune_openaq.py` now sets every attention module (C has one, CPLUSBI two).
- `train_local.py` default behaviour (variants A–D) is unchanged; C+ keys are opt-in.
