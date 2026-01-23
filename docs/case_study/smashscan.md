# SmashScan Case Study

**Repository**: https://github.com/jpnaterer/smashscan
**Author**: jpnaterer
**Blog**: [Medium Series](https://blog.goodaudience.com/smashscan-using-neural-networks-to-analyze-super-smash-bros-melee-a7d0ab5c0755)
**Status**: Inactive (2018)

## Overview

SmashScan uses neural networks for automated video analysis of Melee tournament footage. It detects stages, characters, and UI elements from recorded streams to enable automated metadata extraction for the SmashVods database.

## Goals

1. Extract gameplay information from tournament videos
2. Identify legal stages despite stream overlays
3. Segment video timestamps where gameplay occurs
4. Integrate with SmashVods for automated indexing

## Approaches Tried

### Attempt 1: Template Matching
- Matched "Go!" sprite at match start
- Used OpenCV template matching
- **Failed**: Scale variations and overlay occlusion

### Attempt 2: Feature Matching (ORB)
- OpenCV Feature Matching with ORB detector
- Scale-invariant detection
- **Failed**: Gameplay footage lacks distinctive features

### Attempt 3: Neural Network (Success)
- DarkFlow (YOLO variant)
- Trained on manually annotated bounding boxes
- **Succeeded**: Reliable detection despite overlays

## Architecture

### DarkFlow/YOLO

```
Tournament Video Frame
    │
    ▼
┌─────────────────────────────┐
│  YOLO Convolutional Network │
│  - Real-time detection      │
│  - Bounding box localization│
└─────────────────┬───────────┘
                  │
                  ▼
┌─────────────────────────────┐
│  Detected Objects           │
│  - Stage identification     │
│  - UI element locations     │
└─────────────────────────────┘
```

## Training Data

**Collection**:
- Downloaded videos via pytube
- 360p at 30fps for manageable size
- 100+ videos, ~300 matches
- All 6 legal tournament stages

**Annotation**:
- matplotlib Rectangle Selector widget
- Manual bounding box drawing
- Handled varied stream overlays

**Challenges**:
- Different broadcaster overlays
- Varying capture card quality
- Different stage perspectives
- Stream compression artifacts

## Detection Capabilities

**Implemented**:
- Stage detection (all 6 legal stages)
- Percent counter segmentation

**Planned (incomplete)**:
- OCR for percent displays
- Stock icon detection
- Match timer
- Character identification
- Support for other Smash games

## Technical Stack

- Python 3.6+
- TensorFlow/TensorFlow-GPU
- OpenCV
- DarkFlow
- youtube-dl / pytube
- numpy

## Processing Pipeline

```
YouTube Tournament Videos
    ↓ pytube download
Raw Video Files
    ↓ Frame extraction
Training Images
    ↓ Manual annotation
Labeled Dataset
    ↓ DarkFlow training
Stage Detection Model
    ↓ Video processing
SmashVods Metadata
```

## Results

**Stage Detection**: Functional, handles overlays
**Percent Detection**: Working via template matching
**Production Status**: Not integrated into SmashVods

## Related Projects

| Project | Approach |
|---------|----------|
| Visor (2017) | Haskell stage classification |
| SSBM_Image_Processing | SVM stage classification |
| ssbm_fox_detector | Faster R-CNN character detection |
| DeepLeague | Similar approach for LoL |

## Relevance to ExPhil

**Not directly applicable** - SmashScan is for video analysis, not gameplay AI.

**However**:
- Demonstrates computer vision approaches to Melee
- Stage/character detection could augment replay analysis
- Shows what's possible with annotated tournament data

## References

- [Repository](https://github.com/jpnaterer/smashscan)
- [Medium Blog](https://blog.goodaudience.com/smashscan-using-neural-networks-to-analyze-super-smash-bros-melee-a7d0ab5c0755)
- [YOLO Paper](https://arxiv.org/abs/1506.02640)
- [DarkFlow](https://github.com/thtrieu/darkflow)
