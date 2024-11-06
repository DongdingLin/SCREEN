# SCREEN

This repo is the code and synthetic data of the ACM MM 2024 paper "SCREEN: A Benchmark for Situated Conversational Recommendation".

## Dataset

The organized dataset will be uploaded to Google Drive soon and you can download it from the Google Drive link.

## Dataset Construction

### Requirements

The required packages are listed in `requirements.txt`. Suppose you use [Anaconda](https://www.anaconda.com/) to manage the Python dependencies, you can install them by running:

```
conda create -n screen python=3.11
conda activate screen
pip install -r requirements.txt
```

### Step 1: Prepare the Seed Dataset

We will also upload the organized scene snapshot to Google Driver. Please put it in the `scene_info_pool` in the root directory.

### Step 2: Dataset Construction

Please set your openai key and other related parameters, and then run `dialogue_simulation.py` to start constructing data.

```
# set your OpenAI API key
export OPENAI_API_KEY=""

python dialogue_simulation.py
```

If you hope NOT to show the instructions and the synthesized conversations in the console, please set --show_description and --show_message to false.

## Acknowledgement

Our code is partially based on the implementation of ChatArena. We thank the authors for their excellent work.

## Citation

If you use our data or code in your work, please kindly cite our work as:

```
@inproceedings{lin-etal-2024-screen,
    title = "SCREEN: A Benchmark for Situated Conversational Recommendation",
    author = "Lin, Dongding and 
              Wang, Jian and 
              Leong, Chak Tou and
              Li, Wenjie",
    year = {2024},
    isbn = {9798400706868},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3664647.3681651},
    doi = {10.1145/3664647.3681651},
    pages = {9591–9600},
    numpages = {10},
    keywords = {benchmark, role-playing, situated conversational recommendation},
    location = {Melbourne VIC, Australia},
    series = {MM '24}
}
```
