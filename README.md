# Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning


[![arXiv](http://img.shields.io/badge/arxiv-2511.22640-red?logo=arxiv)](https://www.arxiv.org/abs/2511.22640)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/riccardodesanti/generative-exploration/blob/development/notebooks/tutorial_fdc.ipynb)

This repository contains the official implementation of the Flow Density Control algorithm, a method for optimizing general utilities beyond entropy-regularized reward maximization. 


## Installation

Create conda environment:
```bash
conda env create -f environment.yaml
```
Activate environment:
```bash
conda activate genexp
```
In root directory, install package:
```bash
pip3 install -e .
```

## Usage

To use FDC on your own models, follow the installation instructions above and check the ``notebooks/tutorial_fdc.ipynb`` notebook for a typical first example using the Manifold Exploration functional. Check the `src/genexp/trainers` folder to see all implemented utilities/divergences.

For more fine-grained control (especially if your model generates something other than a PyTorch Tensor):

### Wrap your model with either our FlowModel class or our DiffusionModel class (or subclass them)

We define a prototypical `FlowModel` class (and `DiffusionModel` class extending it) in `src/genexp/models.py` in order to wrap simple models. We assume your model is a velocity field predictor in the case of flow models, and a noise predictor in the case of diffusion models. If your flow/diffusion model does something different, simply subclass the respective class, and implement the `velocity_field`/`score_func` according to your model's functionality. Note that we use alpha/beta according to their definitions in the [Adjoint Matching paper](https://arxiv.org/abs/2409.08861), see section 2.

### Choose one of our Sampler classes (or subclass your own)

If your model outputs a PyTorch tensor, we recommend using our `EulerMaruyamaSampler` by simply passing the data shape. Otherwise, subclass the `Sampler` class, and the `Sample` object, in order to define exactly part of your data is used in the Adjoint Matching procedure. No matter how complex your data is, by implementing the `adjoint` property of your `Sample` object you define a tensor with respect to which gradients can be computed in FDC.

## Citation

If you use this code in your research, please include the following citations in your work:


```
@inproceedings{de2025flow,
      title={Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning}, 
      author={Riccardo De Santi and Marin Vlastelica and Ya-Ping Hsieh and Zebang Shen and Niao He and Andreas Krause},
      year={2025},
      booktitle={Advances in Neural Information Processing Systems},
      url={https://arxiv.org/abs/2511.22640}, 
}

@inproceedings{de2025provable,
      title={Provable Maximum Entropy Manifold Exploration via Diffusion Models}, 
      author={Riccardo De Santi and Marin Vlastelica and Ya-Ping Hsieh and Zebang Shen and Niao He and Andreas Krause},
      year={2025},
      booktitle={Proceedings of the 42nd International Conference on Machine Learning},
      url={https://arxiv.org/abs/2506.15385}, 
}
```
