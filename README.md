# Wavy: Generating Images from Speech

Recent advancements in generative artificial in- telligence (AI) have motivated the development of multimodal architectures capable of handling diverse data types simultaneously. Our project contributes to this landscape by introducing a novel Unified Encoding Framework, integrating HuBERT, a speech encoding model, with a pre- trained Vision Transformer designed for image processing. This framework aims to encode both speech and image data into a unified representa- tion, capturing linguistic distinctions in speech and contextual understanding in images. Our methodology incorporates specialized projectors for aligning speech and image embeddings, along- side mathematical operations for effective model training. In evaluation, our Speech-CLIP encoder model demonstrates commendable efficacy in gen- erating images from speech prompts, although slightly falling short of the text diffusion model’s performance. Through this work, we aim to as- sess the feasibility of introducing a model that can create images based on audio signals, enabling users to express artistic ideas directly through spoken language and expand the scope of image generation through multimodal approaches.

# Installation
```
cd AudioToken
pip install -r requirements.txt
```
And initialize an Accelerate environment with:
```
accelerate config
```
Download Audio Encoder pre-trained model 
```
Download `.pt` file from https://onedrive.live.com/?authkey=%21APLo1x9WFLcaKBI&id=6B83B49411CA81A7%2125955&cid=6B83B49411CA81A7&parId=root&parQt=sharedby&o=OneUp

And place it in a folder as, -> `model/BEATs/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt`

```

# Inference

After you've trained a model with the above command, you can simply generate images using the following script:
```angular2html
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export DATA_DIR="./vggsound/"
export OUTPUT_DIR="output/"

accelerate launch inference.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --data_dir=$DATA_DIR \
  --output_dir=$OUTPUT_DIR 
```
