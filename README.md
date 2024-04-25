# GegenConv
This repository contains the code for [Gegenbauer Graph Neural Networks for Time-varying Signal Reconstruction](https://doi.org/10.1109/TNNLS.2024.3381069) published at the Transaction of Neural Networks and Learning Systems (TNNLS).

Authors: [Jhon A. Castro-Correa](https://www.eecis.udel.edu/~jcastro/), [Jhony H Giraldo](https://sites.google.com/view/jhonygiraldo), [Mohsen Badiey](https://www.eecis.udel.edu/~badiey/), [Fragkiskos D. Malliaross](https://fragkiskos.me)

---

<div align="center">
    <a href="./">
        <img src="./assets/pipeline.png" width="95%" alt="Pipeline"/>
    </a>
</div>

---

## Getting started

### Create a virtual environment
If you have Python's `virtualenv` package installed (through `apt` on Ubuntu), you can make a virtual environment via the following:

```bash
# change your path and name for your virtual env (remove [])
python3 -m virtualenv ~/path/to/virtual/env/storage/[name]

# then source your environment (you need to do this each time you run!)
# again, remove the [] from name
source ~/path/to/virtual/env/storage/[name]
# this should give a nice `([name])` at the beginning of each terminal line
```

On the other hand, if you have installed Anaconda on your system, you can make the virtual environment via the following:

```bash
# change your path and name for your virtual env (remove [])
conda create --name [myenv]

# then source your environment (you need to do this each time you run!)
# again, remove the [] from name
conda activate [myenv]
# this should give a nice `([name])` at the beginning of each terminal line
```

### Clone this repository

```bash
git clone https://github.com/jcastro295/GegenGNN.git  
```

### Prerequisites

Our code requires Python >= 3.10.

You also need the additional packages listed in the [requirements.txt](requirements.txt) file. You can install the requirements using:

```bash
pip install -r requirements.txt
```

## Run the code 

With the requirements installed, the scripts are ready to run and  used. Make a **copy** of the `settings.file.toml` file. Then edit the copy with your desired settings. Then you can run the script by calling the following:

```bash
python3 [filename.py] -f [YOUR_SETTINGS.TOML]
```

## Credits

if you use our code, please consider citing our work:

```bibtex
@Article{gegenconv2024,
  author={Castro-Correa, Jhon A. and Giraldo, Jhony H. and Badiey, Mohsen and Malliaros, Fragkiskos D.},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Gegenbauer Graph Neural Networks for Time-Varying Signal Reconstruction}, 
  year={2024},
  volume={},
  number={},
  pages={1-0},
  keywords={Convolution;Polynomials;Vectors;Task analysis;Signal processing;Signal reconstruction;Matrix decomposition;Gegenbauer polynomials;graph neural networks (GNNs);graph signal processing (GSP);time-varying graph signals},
  doi={10.1109/TNNLS.2024.3381069}}
```

```bibtex
@Inproceedings{timegnn2023,
  author={Castro-Correa, Jhon A. and Giraldo, Jhony H. and Mondal, Anindya and Badiey, Mohsen and Bouwmans, Thierry and Malliaros, Fragkiskos D.},
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Time-Varying Signals Recovery Via Graph Neural Networks}, 
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Time series analysis;Signal processing algorithms;Filtering algorithms;Transformers;Graph neural networks;Spatiotemporal phenomena;Forecasting;Graph neural networks;graph signal processing;time-varying graph signal;recovery of signals},
  doi={10.1109/ICASSP49357.2023.10096168}}
```

### Contact: 

For any query, please contact me at: <jcastro@udel.edu>