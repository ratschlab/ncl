# Neighborhood Contrastive Learning Applied to Online Patient Monitoring
This repository contains the code used for the paper "Neighborhood Contrastive Learning Applied to Online Patient Monitoring" accepted for a short talk at ICML 2021.
<p align="center">
<img src="https://github.com/ratschlab/ncl/blob/main/bin/NCL_figure.png" width="400">
</p>

## Citing Our Work
```
@conference{yecheetal21,
  title = {Neighborhood Contrastive Learning Applied to Online Patient Monitoring},
  author = {Yèche, H. and Dresdner, G. and Locatello, F. and H{\"user}, M. and R{\"a}tsch, G.},
  booktitle = {38th International Conference on Machine Learning},
  month = jul,
  year = {2021},
  doi = {},
  month_numeric = {7}
}
```
## Setup
###  Conda Environment
To run the code you need to set up a conda environment with the `environment.yml` file.

First, ensure you have `conda` installed. Then, if you intend to run the model on **GPU**, do:
```
conda env create -f environment.yml
```
We don't recommend running the code on CPU as it will be very slow to train models. If you still intend to, you will need to modify `tensorflow-gpu` by `tensorflow` in the `environment.yml` file.

Once the command runs without errors, you should have a new environment available, called `ncl`, that you can activate with :
```
conda activate ncl
```
### Data Gathering 
We use two publicly available datasets in our experiments, MIMIC-III benchmark (https://github.com/YerevaNN/mimic3-benchmarks) and Physionet 2019 (https://physionet.org/content/challenge-2019/1.0.0/). In this part, we quickly describe the procedure to obtain them.

#### MIMIC-III Benchmark
This dataset is itself a pre-processed version of MIMIC-III (https://mimic.physionet.org/). We do not provide the MIMIC-III data itself. You must acquire the data yourself from https://mimic.physionet.org/. Specifically, download the CSVs. Once you have stored these files in the folder `path/to/raw/mimic3`, you need to execute the following command to obtain the benchmark version of the data:

```
git clone https://github.com/YerevaNN/mimic3-benchmarks/
cd mimic3-benchmarks/
python -m mimic3benchmark.scripts.extract_subjects path/to/raw/mimic3 path/to/mimic3-benchmark/
python -m mimic3benchmark.scripts.validate_events path/to/mimic3-benchmark/
python -m mimic3benchmark.scripts.extract_episodes_from_subjects path/to/mimic3-benchmark/
python -m mimic3benchmark.scripts.split_train_and_test path/to/mimic3-benchmark/
python -m mimic3benchmark.scripts.create_multitask path/to/mimic3-benchmark/ path/to/mimic3-benchmark/source_data
```
You can find more details about each of this command directly at https://github.com/YerevaNN/mimic3-benchmarks.

Once this is done, the source data should be in `path/to/mimic3-benchmark/source_data`.

#### Physionet 2019
Again, we do not provide the Physionet 2019 data itself. You need to download it from https://physionet.org/content/challenge-2019/1.0.0/. Once this is done data should be in two folders called `trainingSetA` and `trainingSetB`. Ensure to put all patient files in a single folder called `path/to/physionet2019/source_data/`. 

### Pre-processing

At this point you should have a conda environment called `ncl` and two folders containing the sources data: `path/to/mimic3-benchmark/source_data` and `path/to/physionet2019/source_data/`.

In this part we pre-process the data to make it compatible with our pipeline. To do so, we have two scripts `data/preprocess/build_physionet-2019` and `data/preprocess/build_mimic-III-benchmark`. The only thing to do is to insert the previously mentioned paths at the first line of each file. For instance with `data/preprocess/build_physionet-2019`, I will go from :
```
load_path=path/to/physionet2019/source_data/
save_path=./data/preprocess/physionet2019_resources/
resource_path=./data/preprocess/physionet2019_resources/
static_col='Age Gender Unit1 Unit2 HospAdmTime'
python data/preprocess/build_dataset.py --dataset physionet2019 --load_path $load_path --save_path $save_path --resource_path $resource_path --static_columns $static_col
```
to
```
load_path=my/personal/path/to/physionet2019/
save_path=./data/preprocess/physionet2019_resources/
resource_path=./data/preprocess/physionet2019_resources/
static_col='Age Gender Unit1 Unit2 HospAdmTime'
python data/preprocess/build_dataset.py --dataset physionet2019 --load_path $load_path --save_path $save_path --resource_path $resource_path --static_columns $static_col
```

Once this is done, you can simply run:
```
sh data/preprocess/build_physionet-2019
sh data/preprocess/build_mimic-III-benchmark
``` 

This will create `h5` dataset files in either `data/preprocess/physionet2019_resources/` or `/data/preprocess/mimic3_resources/`, called `non_scaled.h5` and `scaled.h5`. `non_scaled.h5` contains the forward imputed version of the data whereas `scaled.h5` is sthe scaled version of it that we used as input to our pipeline. 


## Training
In this part, we detail how to train all the methods we report in our paper. 

If you want to run methods that are not in the paper you can run `python main.py -h` to get help. Otherwise, we've built scripts running each of the methods reported in `bin/run_scripts/`. Here is an example of a script that can be found at `bin/run_scripts/mimic3/train_eval_NCL_w`:

```
python main.py -m train_eval -tc configs/mimic3/NCL/train_unsupervised.gin \
	       -ec configs/mimic3/NCL/eval_unsupervised/mlp_decomp_eval.gin configs/mimic3/NCL/eval_unsupervised/linear_decomp_eval.gin configs/mimic3/NCL/eval_unsupervised/mlp_los_eval.gin configs/mimic3/NCL/eval_unsupervised/linear_los_eval.gin\
	       -l logs_experiment/mimic3/NCL_w \
	       -a 0.3 -w 16 \
	       -tau 0.1 \
	       -mom 0.999 \
               -sdo 0.2 \
  	       -tcos 8 \
	       -gb 0.1 \
               -tch 0.5 \
	       -sd 1111 2222 3333 4444 5555 6666 7777 8888 9999 1234 2345 3456 4567 5678 6789 9876 8765 7654 6543 5432 \
	       -en mlp_decomp linear_decomp mlp_los linear_los \
```
As it name indicates, this script train and evaluates `NCL(n_w)` method over 20 seeds on the MIMIC-III benchmark dataset. One important thing to note is that we base our pipeline on `gin-config` files (https://github.com/google/gin-config). You can run it the following way :

```
sh bin/run_scripts/mimic3/train_eval_NCL_w
```
In addition you can follow the training using `tensorboard` by running the command:
```
tensorboard --logdir ./logs_experiment/mimic3/
```

## Directory organization

#### `data/` 
 
 This folder contains everything relative to data loading and preprocessing. The most important file is `data/loader.py` containing the loader (all on CPU) fo the ICU data. 
 
#### `model/`
 
 This folder contains every relative to the model used to train a representation:
 - `model/architectures/` contains the different building blocks of the representation model, encoder, critic, data augmentations, losses, and neighborhood functions.
 - `model/methods/` contains the file for the different approaches employed.
 - `train.py` is the common run file called by main.

#### `eval/`
This folder contains everything relative to evaluating a representation :
 - `eval/architectures/` contains the different building blocks of the classifier model.
 - `eval/down_stream_tasks/` contains the different classifier used `AddOnClassification` or `AddOnBinnedClassification` for length-of-stay.
 - `evaluate.py` is the common run file called by main.

#### `configs/`
This folder contains all the `gin-config` (https://github.com/google/gin-config) for the different methods.

#### `bin/` 
This folder contains pre-built scripts to run the code. 
