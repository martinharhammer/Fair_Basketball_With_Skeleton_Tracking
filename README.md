# Fair Basketball With Skeleton Tracking
![viz1](https://github.com/user-attachments/assets/c767902e-7a83-4688-abe2-a12e216668da)
## Overview
This project was carried out as part of my bachelor thesis.  The goal was to apply computer vision techniques to basketball game recordings in order to re-evaluate scoring fairness by incorporating player height into the scoring process.

The emphasis of the work lies on the application of computer vision methods rather than on designing the fairest possible scoring system, since official player heights are already known and could be used in simpler ways. Instead, the project demonstrates how skeleton tracking and geometric reasoning can be leveraged to estimate player heights directly from video footage and how this estimation can be used to weight scoring events.

The main goals were:
- Detection of scoring events  
- Estimation of player height relative to a reference point (the basket) using skeleton tracking  
- Re-computation of scores by applying an easiness factor derived from player height to produce a "fairer" score 

## Approach
The system is divided into two parts: **precompute** and **game logic**.  
The precompute stage extracts all necessary detections from the raw video. The game logic stage then combines these detections, identifies shooters, estimates player heights, and re-evaluates scores. This division ensures that expensive inference tasks are only run once, while the analysis can be repeated quickly.

### Precompute Stage
The precompute stage applies several detection models to the video frames.

Ball tracking is implemented with a minimal version of the [WASB-SBDT approach](https://github.com/nttcom/WASB-SBDT). To adapt the model to the basketball used in FIBA competitions, the Molten BG5000, additional fine-tuning was conducted. The distinctive white panel of the BG5000 proved challenging for the base model, frequently resulting in misdetections. Finetuning the model on this ball substantially improved detection robustness. Inference is performed using HRNet, applied over sliding windows of three consecutive frames. HRNet generates heatmaps in which local maxima correspond to candidate ball positions. By enforcing temporal consistency across consecutive frames, outlier detections that deviate significantly from the trajectory are suppressed, resulting in stable and reliable ball tracking.

In addition to the ball, the hoop and court keypoints are detected with YOLO-based models. Player skeletons are estimated using OpenPose in the Pose25 format. Because OpenPose is both slow and produces large outputs, it is only run in a backtracking window around scoring events, not on the full video. All detections are written to JSONL files so that later analysis can be repeated without rerunning the heavy inference stage.

### Game Logic Stage
In the **game logic stage**, the detection of scoring events is implemented using a finite-state machine. A valid scoring sequence requires the ball to appear in a zone above the hoop, trigger a change in the hoop bounding box that indicates the ball passing through the net, and subsequently appear in a zone below the hoop.

Once a scoring event is detected, the responsible shooter is identified by backtracking the ball trajectory and comparing its coordinates to the wrist positions of player skeletons obtained from OpenPose. If the ball passes within a defined distance of a player’s wrist, that skeleton is assigned as the shooter.

![viz_scoring_gif](https://github.com/user-attachments/assets/6723b138-4813-4872-a18d-65adf020912d)

The type of score is then determined through a set of heuristics. Three-point shots are classified when the number of frames between the identified shooter frame and the final scoring trigger exceeds a threshold, reflecting longer-distance attempts. Free throws (one point) are identified when the shooter’s ankle keypoints lie close to the free-throw line, which is localized using court keypoints. All other cases are categorized as two-point field goals.

Player height estimation is performed using court homography. A homography of the court is first computed following approaches from prior basketball analysis work, and by applying the inverse homography the hoop position is projected onto the ground plane, producing the _hoop shadow point_. With this reference, the vertical distance from the floor to the hoop is set to 3.05 m. The shooter’s height is then estimated by comparing their OpenPose skeleton dimensions to this calibrated reference, providing a relative height estimate directly from the video.

![viz_height_gif](https://github.com/user-attachments/assets/1b2de49f-b09f-45ec-8d8a-65c0fcf662a6)

Finally, the estimated player heights are used to compute a new **fair score**. Each scoring event is reweighted with an _easiness factor_ that favors shorter players and slightly penalizes taller ones. Heights are first clamped to a reasonable range (160–225 cm), and the median height of all players serves as the neutral reference point. At this median height, the factor is 1.0. Players shorter than the median receive a boost of up to +50%, while taller players can receive a reduction of up to −50%. Events with missing height default to a neutral factor. The weighted points are then aggregated per team to produce an adjusted final fair score.

## Results
The system successfully recognized 80 out of 88 scoring events in the analyzed video, which demonstrates that the scoring event detection pipeline is generally reliable.  

Height estimation was reasonably accurate for the majority of scoring events, but several cases showed noticeable underestimation. To mitigate these cases, a clamping strategy was introduced to restrict estimated heights to a plausible range. The inaccuracies are likely attributable to inconsistencies in the court keypoint detections, which affect the computation of the hoop shadow point and thereby the reference height. In addition, the current approach does not explicitly compensate for perspective changes or for variations in player positioning on the court. These factors contribute cumulatively to the observed errors. 

While the approach works as a proof of concept, there is still room to make it more reliable. More stable court detections, handling perspective change, and taking the shooting player’s pose into account would help reduce errors and make the fair score calculation more consistent.

That being said, the final fair score in the analyzed game was **92–73 in favor of China**. The original score was **109–50**, so the adjusted fair score reduces the margin but does not change the overall outcome, with China remaining the winner.

## Usage

1. Create and activate environment First create the environment from the provided YAML file and activate it:
```
bash conda env create -f environment.yml
conda activate env
```
2. Build OpenPose (once)
```
bash scripts/build_openpose.sh
```
3. Run the game logic
This uses the precomputed outputs already included in the repository
```
python -m game_logic
```
4. (Optional) Run precompute on a new video
To process your own game video:
Upload Video to path
Upload Models to path
- Works best with the Molten BG5000 ball.
- Cut out replays to avoid double counting.
- Processing can take several hours depending on video length.
- Specify metadata (which team starts where, halftime timestamp) in the config.
Then run:
```
python -m precompute.runner
python -m game_logic
```

