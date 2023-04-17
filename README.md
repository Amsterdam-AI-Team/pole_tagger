# Pole Tagger: Code to annotate extracted poles

This repository contains a Python implementation of a pole tagger to annotate semantically segmented street lights in point clouds. 

## Preparation
This code has been tested with Python 3.10.9.

1. Clone this repository 

  ```sh
  git clone https://github.com/Amsterdam-AI-Team/pole_tagger.git
  ```

2. Install all python dependencies

  ```sh
  pip install -r requirements.txt
  ```
  
## Data
The data folder contains a csv and images subfolder:

- The csv subfolder contains the csv files with extracted street lights.
- The images subfolder contains (1) images of the extracted street lights to annotate and (2) the images of the different types of street lights.

## Usage


- Start up the tagger:

  ```sh
  python3 pole_tagger.py
  ```
A detailed manual of the tagger including some annotation examples can be found in the 'pole tagger manual.pdf' file. Furthermore, some example images have been added to this repository to get to understand the pole tagger tool.
