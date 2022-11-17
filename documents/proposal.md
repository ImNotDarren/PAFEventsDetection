# Project Proposal

## Question/Need:

This project comes from the [The 4th China Physiological Signal Challenge]([https://physionet.org/content/challenge-2019/1.0.0/](https://physionet.org/content/cpsc2021/1.0.0/))
in 2021.

Atrial fibrillation (AF) is the most frequent arrhythmia, but paroxysmal atrial
fibrillation (PAF) often remains unrecognized. Previous AF detection algorithms
usually focus on the classification of AF rhythm instead of locating the onsets
and ends of AF episodes. My goal is to develop an algorithm that search the AF
episodes in dynamic ECG regords.


## Data Description:

The data I used is recorded from 12-lead Holter or 3-lead wearable ECG monitoring
devices. It provides variable-length ECG records extracted from lead I and lead II
of the long-term dynamic ECGs, each sampled at 200 Hz.

The training set in the 1st stage consists of 730 records, extracted from the Holter
records from 12 AF patients (5PAF patients) and 42 non-AF patients (usually including
other abnoremal and normal rhythms).

The training set in the 2nd stage consists of 706 records from 37 AF patients (18 PAF
patients) and 14 non-AF patients.

## Tools:

* **Pandas** for exploratory data analysis.
* **Python waveform-database package** for reading the data
* **Matplotlib** and **Seaborn** for plotting.
* **Scikit Learn** and **Tensorflow** for modeling.
* **Pickle** for saving models in a pickle file.

## MVP Goal:
My MVP goal is to analyse the data to build some virtualizations and build a baseline
model.
