
# Preprocessing and rearrangement pipeline
This .md file briefly covers how the preprocessing and rearrangement works. 

## Preprocessing & rearrangement
After dataset is donwloaded and put into RGHB, run preprocess.py with the correct config. That creates directories in each of the dataset's sub-dataset directories with frames and landmarks.

## Rearrange
After we have created the directories with frames and landmarks, it is time to prepare the data for CLIP embeddings, or being run into the models. rearrange.py will create a .JSON file that is stored in preprocessing/dataset_json (e.g. dlib, mtcnn depending on the preprocessing method). The .JSON file is structure like this: 

{ dataset: {
  label_cat: {                                                                   
    split: {                                                                  
      comp (FF++ only): {
        video_id: {                                                                   
          label, frames: [...] 
        }}}}}    

After this step, the JSON files are being used to create the CLIP embeddings (for this experiment). They could also be used as input into the DeepfakeBench models, or custom models that don't need CLIP embeddings.
