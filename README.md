# Cross-Domain FSL via Noise-enhanced Supervised Autoencoder (NSAE)

An unofficial PyTorch re-implementation of the ICCV 2021 paper *"Boosting the Generalization Capability in Cross-Domain Few-Shot Learning via Noise-Enhanced Supervised Autoencoder"* (Liang et al.).

## Overview

This repository trains a supervised autoencoder on top of a ResNet-10 feature extractor and evaluates it on few-shot classification episodes. During pre-training the model jointly optimizes three objectives: a classification loss on the encoded features, a pixel-wise reconstruction loss between the input and the decoder output, and a classification loss on the *re-encoded reconstruction* — encouraging features that generalize across domains. At test time each novel episode is fine-tuned on its support set and then classified two ways: directly with the linear classifier and with a k-NN label-propagation refinement over the query features.

> Note: This is an independent re-implementation for study/reproduction purposes. It is not the authors' official code. See [Citation](#citation).

## Repository structure

| File | Purpose |
| --- | --- |
| `train.py` | Single entry point: pre-trains the NSAE, then runs episodic fine-tuning + evaluation. |
| `network.py` | `NSAE_model` — ResNet feature extractor + linear classifier + convolutional decoder, and the combined loss. |
| `backbone.py` | ResNet backbones (`ResNet10` is used); modified from `facebookresearch/low-shot-shrink-hallucinate`. |
| `io_utils.py` | Argument parsing (`parse_args`) and the backbone model registry. |
| `configs.py` | Dataset root paths and the checkpoint save directory. |
| `utils.py` | `AverageMeter`, learning-rate schedule, and small helpers. |
| `datasets/miniImageNet_few_shot.py` | Data managers for base training and episodic sampling. |
| `datasets/additional_transforms.py` | Extra image augmentations (e.g. `ImageJitter`). |

## Requirements

There is no `requirements.txt` in the repo. Based on the imports, you need Python 3 and:

- `torch` and `torchvision` (PyTorch >= 1.7 is required for `nn.Unflatten`, used in the decoder)
- `numpy`
- `scipy` (`scipy.linalg.sqrtm`)
- `scikit-learn` (`sklearn.neighbors.NearestNeighbors`)
- `pandas`
- `Pillow`

A CUDA-capable GPU is required — the model and tensors are moved to CUDA unconditionally (`.cuda()`).

## Datasets

- **Base (pre-training):** miniImageNet, expected at `./data/miniImagenet/train` (an `ImageFolder`-style directory, one subfolder per class). The base loader assumes 64 classes.
- **Cross-domain targets:** `configs.py` also defines paths for the [BSCD-FSL benchmark](https://github.com/IBM/cdfsl-benchmark) domains — `CropDisease`, `EuroSAT`, `ISIC`, and `ChestX` — and `--dtarget` selects among them. However, only the miniImageNet loader is implemented in this repository, so as written the fine-tuning/evaluation episodes are also drawn from miniImageNet. Add the corresponding target-domain loaders to evaluate true cross-domain transfer.

Edit the paths at the top of `configs.py` to match your local layout before running.

## Usage

Everything runs from a single script — `train.py` first pre-trains the autoencoder on the base set, then fine-tunes and evaluates on episodes:

```bash
python train.py --train_aug
```

Common flags (safe to pass, since they are shared by both parse passes):

- `--model` — backbone architecture (default `ResNet10`, the only registered backbone)
- `--train_aug` — enable training-time data augmentation
- `--lamda1` — weight of the reconstruction (MSE) loss (default `1.0`)
- `--lamda2` — weight of the classification loss on the reconstruction (default `1.0`)
- `--k_lp`, `--delta`, `--alpha` — label-propagation hyper-parameters used at evaluation

Checkpoints are written to `./models/checkpoints/<model>_bsr[_aug]/` (controlled by `save_dir` in `configs.py`).

During evaluation the script prints two accuracies per episode and a final mean with a 95% confidence interval:

- **BSR** — accuracy of the fine-tuned linear classifier on the query set
- **BSR+LP** — accuracy after k-NN label propagation over the query features

> Configuration note: `train.py` calls `parse_args('train')` and `parse_args('finetune')` on the *same* command line, and the two parsers do not share their script-specific arguments. Passing a train-only flag (`--num_classes`, `--save_freq`, `--start_epoch`, `--stop_epoch`) or a finetune-only flag (`--dtarget`, `--test_n_way`, `--n_shot`) will make the other parse pass fail. To change the number of pre-training epochs, classes, or the fine-tuning length, edit the defaults in `io_utils.py` and the `finetune_epochs_recon` / `finetune_epochs` values (currently set to `1`) near the bottom of `train.py`.

## Citation

This is an unofficial re-implementation. Please cite the original authors:

```bibtex
@InProceedings{Liang_2021_ICCV,
    author    = {Liang, Hanwen and Zhang, Qiong and Dai, Peng and Lu, Juwei},
    title     = {Boosting the Generalization Capability in Cross-Domain Few-Shot Learning via Noise-Enhanced Supervised Autoencoder},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {9424-9434}
}
```

Paper: [ICCV 2021 (CVF Open Access)](https://openaccess.thecvf.com/content/ICCV2021/papers/Liang_Boosting_the_Generalization_Capability_in_Cross-Domain_Few-Shot_Learning_via_Noise-Enhanced_ICCV_2021_paper.pdf) · [Supplementary](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Liang_Boosting_the_Generalization_ICCV_2021_supplemental.pdf)

## Acknowledgements

The ResNet backbone and dataset loading code are adapted from [facebookresearch/low-shot-shrink-hallucinate](https://github.com/facebookresearch/low-shot-shrink-hallucinate).

## License

Released under the MIT License. See [LICENSE](LICENSE).
