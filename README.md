# Self- and Cross-Excitation in Stack Exchange Question & Answer Communities
### Description
This code repository documents key steps of the user excitation analysis presented in the paper "Self- and Cross-Excitation in Stack Exchange Question & Answer Communities" WWW'19 (see citation below). These key steps comprise pre-processing user activity event streams, performing stationarity analysis, fitting multivariate Hawkes processes (in particular including beta parameter estimation), computing permutation tests and predicting future event times. A Python-based library for illustrating the simulation, plotting and fitting of univariate and multivariate Hawkes processes is included.

The naming of the source script reflects processing pipeline order. Usage instructions for each source script are included as comments in its header.
- 01_extract_activity_event_streams.py assumes the complete [Stack Exchange dataset](https://archive.org/details/stackexchange) (as of June 2017) was downloaded and extracted via data_preparation/s01_extract_log_stackexchange.py of [ActivityDynamics](https://github.com/simonwalk/ActivityDynamics). This script further transforms the data to an event stream format suitable for fitting multivariate Hawkes processes.
- 02_zeilis* feature the application of Zeilis et al.'s algorithms to study stationarity patterns and growth of Stack Exchange instances.
- 03_excitation* include the fitting process of all parameters (incl. beta) of Hawkes processes, and the permutation and prediction experiments presented in the paper. Note in particular that [tick](https://github.com/X-DataInitiative/tick) is included here for its fast C++ implementation of Hawkes process fitting routines.
- hawkes* is a simple (but slower) Python-based illustration of univariate and multivariate Hawkes process simulation, fitting and plotting.

### Requirements
This code leverages both Python and R packages. See requirements.txt for details on the Python packages (or install them using `pip3 install requirements.txt`). Required R packages are [ggpubr](https://cran.r-project.org/package=ggpubr) and [strucchange](https://cran.r-project.org/package=strucchange).

### Citation
If you find this research helpful to your work, please consider citing:
```
@inproceedings{santos2019self,
    author = {Santos, Tiago and Walk, Simon and Kern, Roman and Strohmaier, Markus and Helic, Denis},
    title = {Self- and Cross-Excitation in Stack Exchange Question \& Answer Communities},
    booktitle = {The Web Conference (WWW)},
    year = {2019}
}
```
