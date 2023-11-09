# BannerGen - A Library for Multi-Modality Banner Generation 

## Introduction
Salesforce BannerGen library aims to help graphical designers 
- simpilfy workflow
- scale produtivity
- bring forward creative ideas

which are achieved by leveraging advanced generative AI technologies. Specifically, BannerGen is composed of 3 proprietary multi-modal banner generation methods, namely
- LayoutDETR
- LayoutInstructPix2Pix
- Framed Template Retrieve Adapter

## Blog Post and Citing BannerGen
You can find more details in our [blog post](https://bannergen.placeholder).
**If you're using BannerGen in your research or applications, please cite it with the following BibTeX**:
```bibtex
@inproceedings{yu2023layoutdetr,
    title = "LayoutDETR: Detection Transformer Is a Good Multimodal Layout Designer",
    author = "Ning Yu and 
    Chia-Chih Chen and 
    Zeyuan Chen and 
    Rui Meng and 
    Gang Wu and
    Paul Josel and
    Juan Carlos Niebles and
    Caiming Xiong and
    Ran Xu",
    year = 2023,
    eprint= 2212.09877,
    archivePrefix = arXiv,
    primaryClass = cs.CV,
}
```
# Table of Contents
  - [Library Design](#libdesign)
  - [Getting Started](#libdesign)
    - [Installation](#installation)
    - [Model Download](#model-download)
    - [Usage](#usage)
  - [Ethical and Responsible Use](#ethical-and-responsible-use)
  - [Contact Us](#contact)
  - [License](#license)
## Library Design
<img src="./LibraryDesign.png">

## Getting Started
### Environment
This library has been tested on Ubuntu 20.04 including Python 3.8 and PyTorch 2.1.0 environment. A single A100 GPU is employed for banner generation. Nevertheless, the peak GPU memory usage is 18GB, any NVIDIA GPU with larger memory should suffice. For more information about our base image configuration, please refer to the Dockerfile.


### Installation
```bash
git clone git@github.com:salesforce/BannerGen.git
cd BannerGen
chmod +x setup.sh
./setup.sh
```

### Model Download
You can login to your google account to download BannerGen models [here](https://console.cloud.google.com/storage/browser/sfr-bannergen-data-research). Please point `banner_gen.py` `--model_path` to the local directory where you downloaded the models. The purpose of each model file can be looked up in `BANNER_GEN_MODEL_MAPPER`dictionary in `banner_gen.py`.

### Usage
`banner_gen.py` serves as a demo file to illustrate how to initialize headless browser for rendering and how to import, configure, and call the 2 essential fuctions in each of the 3 banner generation methods. These 2 functions are `load_model` and `generate_banners`. To test a specific method simply assign `--model_name` and point `--model_path` to where you downloaded the model files. Rest of the arguments will be set to the default values and data stored in the repo `test` directory.

To test your own images and/or different types of banner texts, simply assign image path `--image` and the corresponding text types. Here we support header, body, and button as text inputs. 

- Test LayoutDETR
  - python banner_gen.py --model_path=/export/share/chiachihchen/BANNERS --model_name=LayoutDETR
- Test InstructPix2Pix
  - python banner_gen.py --model_path=/export/share/chiachihchen/BANNERS --model_name=InstructPix2Pix
- Test RetrieveAdapter
  - python banner_gen.py --model_path=/export/share/chiachihchen/BANNERS --model_name=RetrieveAdapter
- Check result banner image and html files in ./result

## Ethical and Responsible Use
We note that models in BannerGen provide no guarantees on their multimodal abilities; ill-aligned or biased generations may be observed. In particular, the datasets and pretrained models utilized in BannerGen may contain socioeconomic biases. We plan to improve the library by investigating and mitigating these potential biases and inappropriate behaviors in the future.


## Contact us
If you have any questions, comments or suggestions, please do not hesitate to contact us at [{ning.yu, ran.xu}@salesforce.com](mailto:ning.yu@salesforce.com?subject=[GitHub]%20Source%20Han%20Sans) .

## License
Please refer to [BSD 3-Clause License](LICENSE.txt). We do NOT own the licenses to the fonts stored in `RetrieveAdapter/templates/css/fonts`. To use the fonts in your own work, please acquire the employed font licenses from the respective owners.
