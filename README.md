# Long-Exposure Fusion

## Introduction

**Long-Exposure Fusion** is a full Python pipeline that generates long-exposure style images from videos or burst photo sequences. Images are generated using our own variant of [Exposure Fusion](https://ieeexplore.ieee.org/document/4392748) that we call Hybrid Weight Map Fusion.

![Partial fusion of cloudy sky](assets/fusedClouds.png)

### Motivation

Long exposure images are notoriously hard to capture on non-professional equipment.

![Blurry handheld long-exposure shot](assets/clumsyFiltered.jpg)

Our method allows for the creation of sythetised long-exposure style images using videos or burst image sequences as input.

Furthermore, long-exposure shots are often plagued by unintentional motion-blurring of subjects.

![Long-exposure of train](assets/longExposureTrain.jpg)

Our method is able to selectively apply motion blur to only certain objects in the scene.  
This is achieved by segmenting each input frame using [SAM 2](https://github.com/facebookresearch/sam2). These segmentations are then used to define hybrid weight maps which determine how each pixel of each frame is scaled before being blended with the full sequence. This process is illustrated in the figure below with the weight maps illustrated in grayscale.

![Long-Exposure Fusion illustration](assets/longExposureFusion.svg)

While some [paid proprietary methods](https://apps.apple.com/us/app/spectre-camera/id1450074595) already automate the creation of long-exposure style images from videos. Our method aims to allow for better control of how inputs are fused using the aforementioned segmented hybrid weight maps.

### Example

Long-Exposure Fusion is designed to process videos of scenes with slight changes over time, such as this one below:

![Video of sunset in city](assets/cars.gif)

Without the use of weight maps, a simple mean of all input images yields a dull result with washed out colors and odd trails left by the cars.

![Constant blend of sunset scene](assets/constantCars.png)

Our method is able to treat different objects in the image differently.  
In this example, we've chosen different weight maps for the sky, the city, and the cars such that:
- the sky is fused while giving more impact to brighter pixels.
- the city is captured from a reference frame.
- the cars are eliminated from the scene.

![Long-Exposure Fusion of sunset scene](assets/hiddenCars.png)

## Unfinished

This repository is unfinished and some documentation is missing.
Please note that you may encounter a few issues while running the project on your own.

## Installation

### Disclaimer

This repository was not tested for production and you may run into problems before you get everything working.
Don't hesitate to open a Github issue if you do.

The steps below describe how to install this project on a Linux system with python 3.11.13 and cuda 12.9 and were tested using WSL.
Try using a different environment at your own risk.

### Source

First, clone this repository using:
```bash
git clone https://github.com/ThomasMPont/long-exposure-fusion.git
cd long-exposure-fusion
```

Then ensure you have python 3.11.13 installed:
```bash
python3.11 --version
```
If you don't, you'll want to run something like:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev
```

Create a virtual environment using:
```bash
python3.11 -m venv venv
source venv/bin/activate
```
All of the steps from now on should be run inside this venv.

Install the basic modules using:
```bash
pip install -r requirements.txt
pip install torch torchvision
```

### Dependencies

This repository depends on forks of [LightGlue](https://github.com/cvg/LightGlue) [Practical-RIFE](https://github.com/ThomasMPont/Practical-RIFE), and [SAM2](https://github.com/ThomasMPont/sam2-sequential).
Download them using:
```bash
git submodule update --init
```

#### LightGlue

Install LightGlue from the `LightGlue` submodule:
```bash
pip install -e ./external/LightGlue
```

#### Practical-RIFE

Download [this model](https://drive.google.com/file/d/1ZKjcbmt1hypiFprJPIKW0Tt0lr_2i7bg/view?usp=sharing) and 
`train_log/` into `external/Practical-RIFE/`.  
You may also use any other of the [pretrained Practical-RIFE models](https://github.com/ThomasMPont/Practical-RIFE#trained-model).

#### SAM2

SAM2 requires compiling a custom CUDA kernel with the nvcc compiler. If it isn't already available on your machine, please install [the CUDA toolkits](https://developer.nvidia.com/cuda-toolkit-archive) with a version that matches your PyTorch CUDA version.

Install SAM2 from the `sam2-sequential` submodule:
```bash
pip install -e ./external/sam2-sequential
```

Download [this model](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt) and paste the `.pt`file into `external/sam2-sequential/checkpoints/`.  
You may also use any other of the [pretrained SAM2 models](https://github.com/ThomasMPont/sam2-sequential?tab=readme-ov-file#download-checkpoints).
If you choose to do so, you'll have to edit the model path in `src/pipeline/segment_picker.py`.

#### FFMPEG

To decode video inputs, you'll need to install FFMPEG:
```bash
sudo apt update
sudo apt install ffmpeg
```

## Demo

Now that you've installed everything, let's try running the main script on an example:
```bash
source venv/bin/activate
python long_exposure_fusion.py demo/lake.mp4 -m demo/lake.yaml -o demo/output --align --interpolate 2
```
In case you've missed installs or encounter errors, intermediate results you've achieved are cached under `.cache/`.
At any time, you may add the parameter `--clear-cache` to this script to start from scratch.

After running the command above, you should see:
- FFMPEG decode the input video `demo/lake.mp4`
- LightGlue align and crop all frames to reference frame 0
- RIFE interpolate frames to double the framerate
- The Segment Picker UI open

In the Segment Picker UI:
- Select object 0 using the object buttons on the right. We'll only be using a single object in this demo.
We want this object to segment the lake.
- Left click on the lake to add a positive point to our object.
- Right click next to the lake to add a negative point to our object.
- Add more points if necessary to have SAM2 mask only the lake out.
If you mess up, press 'c' to clear all points in the frame.
- Press 'space' to start mask propagation through the video.
This might take a while, stay hydrated!
- If the mask is lost during propagation, add more points on intermediate frames.
You can traverse the sequence using the scroll wheel.
- Wait for the masks to propagate through to the final frame of the sequence.
- Close the UI.

Finally, you should see two fused images be created using the maps defined in `demo/lake.yaml`.

Once the script is finished, go check out your results in `demo/output`.  
`first.png`, `last.png`, and `reference.png` only hold frames of the input.  
`constant.png` is a simple single weight map fusion of the video.
`partial.png` uses the mask you've defined to selectively blur out the lake without affecting other elements.
Note that some artifacts arise from the large change in view angle throughout the image sequence.
Perhaps you'll find a better weight map that solves this issue.
The next section details how to design and apply your own weight maps.

## Usage

### Parameters

**Required:**
- `input`: Path to input video file or directory containing image sequence
- `-m | --maps <file>`: `.yaml` file defining weight maps for fusion (see Weight Maps section below)

**Optional:**
- `-o | --output <dir>`: Output directory for fused images (output will be cached no matter what)
- `--align`: Align frames before fusion (recommended for moving cameras)
- `--interpolate <multi>`: Interpolate frames to increase framerate (use sparingly)
- `--pyramid`: Process weight maps using pyramid decomposition (as done in the original [Exposure Fusion](https://ieeexplore.ieee.org/document/4392748))
- `--clear-cache`: Clear intermediate results and start from scratch

### Segment Picker

<!-- TODO -->
(UNFINISHED)

### Weight Maps

You've reached the core of the project: **weight maps**.
This section lists which weight maps are available for image fusion and how you can combine them into complex hybrid weight maps.

<!-- TODO -->
(UNFINISHED)

## Pipeline

The following section describes the steps of our Long-Exposure Fusion pipeline.

The pipeline is run in `long_exposure_fusion.py`.

### Decode

If the input is in video form, we first decode it into a sequence of frames using [FFmpeg](https://ffmpeg.org/).

This is implemented in `src/pipeline/decode_video.py`.

### Align

Any input that was not captured by a completely static camera should be aligned before any further processing.

If `--align` is set, we use [LightGlue](https://github.com/cvg/LightGlue) with [DISK](https://arxiv.org/abs/2006.13566) as feature extraciton method to compute a transformation that maps each frame onto the reference frame.
Then, frames with a large enough overlap with the reference frame are cropped onto the largest common rectangle between the aligned frames using [lir](https://github.com/OpenStitching/lir).

This is implemented in `src/pipeline/align_images.py`.

### Interpolate

If the framerate of our input sequence is too low, objects might jump too far between frames. This may lead to visible artifacts in our final fusion.

If `--interpolate <multi>` is set, we use [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) to interpolate intermediate frames, which augments the framerate of our input by a factor of `<multi>`.

This is a costly step and should only be done when necessary.  
It is implemented in `src/pipeline/interpolate_images.py`.

### Segment

In order to treat the various sections of our scene differently during fusion, we use [SAM 2](https://github.com/facebookresearch/sam2) to define and propagate segmentation masks across our image sequence.

The Segment Picker interface allows us to define masks for any number of objects that are then propagated through the image sequence. These masks will be used to restrict the application of the weight maps we define for our fusion step.

This is implemented in `src/pipeline/segment_picker.py`.

### Fusion

The fusion step intelligently combines the input images using the maps defined in `--maps <yaml_file>`.
The all important `masked` weight map applies a given list of weight maps to the distinct sections defined in the segmentation step.

The results of fusion are then copied into the directory provided in `--output <output_dir>`

This is implemented in `src/pipeline/fuse_images.py`.
