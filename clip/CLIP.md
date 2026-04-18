
# How the CLIP embed pipeline works
The preprocessing from preprocess.py and rearrange.py creates directories with frames and landmarks for every sub-dataset within the dataset. Rearrange.py creates a .JSON file with the structure of the dataset, frames etc.

## CLIP embedder
The CLIP embedder uses this .JSON file produced by rearrange.py directly. It is located in preprocessing/dataset_json files. For raw videos without any preprocessing, a custom rearrangement has been made in the preprocessing directory.

To run the CLIP emebedder, first set up a config and run a CLI command; 
```bash
asd
```

## The file structure
embed.py is the entry point when running from CLI. It parses the config, the CLI commands, handles files and sets up output path, saves embeddings and manages the catalogue.csv file. It is basically an orchastrator. 

embed.py handles the loop, and uses CLIP_embedder.py which only purpose is to load the model and embed the frames. They are returned to embed.py in .npz format, and embed.py store them in the outoput dir, along with a run-config.json and catalogue.csv. 

catalogue.csv contains the path to each embedding file, which video and dataset it belongs to etc. catalogue.csv is later used as input to the trainer, which loads the embeddings with a DataLoader. Thus, catalogue.csv is important to correctly configure and store, as it acts as the access point to the embeddings. Run-condig.JSON is mianly used to confirm what type of embeddings are in the directory. 

## The config file


## Checkpoints, crashes and resuming



## Using PEFT
Potential future experiment using PEFT works a bit differently. That requires the CLIP embeddings NOT to be pre-computed, and thus this embeddings pipeline and CLIP model is part of the overarching pipeline. The CLIP model needs to be saved and stored as the fine-tuning processing goes on, and the embeddings need a more robust management system and be exported directly to the model.
