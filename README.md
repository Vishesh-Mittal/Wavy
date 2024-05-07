# Wavy: Generating Images from Speech

Recent advancements in generative artificial in- telligence (AI) have motivated the development of multimodal architectures capable of handling diverse data types simultaneously. Our project contributes to this landscape by introducing a novel Unified Encoding Framework, integrating HuBERT, a speech encoding model, with a pre- trained Vision Transformer designed for image processing. This framework aims to encode both speech and image data into a unified representa- tion, capturing linguistic distinctions in speech and contextual understanding in images. Our methodology incorporates specialized projectors for aligning speech and image embeddings, along- side mathematical operations for effective model training. In evaluation, our Speech-CLIP encoder model demonstrates commendable efficacy in gen- erating images from speech prompts, although slightly falling short of the text diffusion model’s performance. Through this work, we aim to as- sess the feasibility of introducing a model that can create images based on audio signals, enabling users to express artistic ideas directly through spoken language and expand the scope of image generation through multimodal approaches.

# SpeechCLIP Model Traning and Inference

### Detailed Code Explanation

#### Audio Data Loading and Preprocessing
The notebook begins with loading audio data using the `soundfile` library. This is crucial as the first step involves accessing the raw audio files, which are then processed to fit the input requirements of the model.

```python
x, _ = sf.read(f'data/wavs/{wav2cap[0].values[0]}', samplerate=None)
```

This line reads an audio file from the specified path, with `sf.read` returning the audio data `x` and its sample rate. The `samplerate=None` argument indicates that the file's original sample rate is preserved.

#### Feature Extraction with Pretrained Model
The notebook uses a pretrained model from Hugging Face's `transformers` library, specifically the "facebook/hubert-large-ls960-ft". This model is designed to process audio inputs and extract features that can be used for further speech recognition tasks.

```python
processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
```
This line initializes a processor for the Hubert model, which is used to convert raw audio files into a format suitable for the model.

```python
processed = processor(raw, return_tensors="pt").input_values[0]
```
Here, the raw audio data is processed to produce a tensor of input values compatible with PyTorch (`return_tensors="pt"`). This tensor is then ready for model input.

#### Handling and Preparing Data
The notebook includes code for organizing and preparing datasets, which is essential for training the model efficiently.

```python
samples_500 = list(set(wav2cap[1]))[:500]
```
This line demonstrates how to select a subset of data (500 samples) from a larger dataset, ensuring diversity by converting the list to a set and back, which removes duplicates.

#### Dataset and DataLoader Implementation
The implementation of a custom dataset and DataLoader in PyTorch is crucial for batch processing of data during model training. The notebook likely includes classes or functions for this purpose, ensuring efficient data handling.

```python
dataset = ImageTextDataset(wav2cap_dict, processor)
```
This might be an example line where a custom dataset class `ImageTextDataset` is instantiated with audio data and the processor. It suggests the integration of image data with text, possibly for a multimodal model.




# StableDiffusion Model (Generating Images)

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
