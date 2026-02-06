# Metrics Plan & Evaluation Strategy  
**Project: find_my_human_go2 – Low-Light Human Tracking**  
**Week 1 – Phase 3**  
**Last updated:** February 07, 2026  

## 1. Goal of Evaluation
Quantitatively show that our depth-guided + IR-based tracking performs better in low-light conditions than simple RGB-only baselines, especially for:
- Detection accuracy
- Tracking continuity (no ID switches, fast recovery after frame exit)
- Re-identification (correctly recognizing the same person after re-entering frame)
- Rejecting strangers/plants/noise

All tests use real Realsense D435i data (IR + depth) in different lighting levels.

## 2. Lighting Categories (to be noted during recording)
- **Bright**: Normal office/daylight (>200 lux)
- **Dim**: Evening/low lamps (20–80 lux)
- **Very low light**: Almost complete dark, only Realsense IR illuminator active (<10 lux)

Colleague: Please use phone lux meter app to note approximate lux for each bag.

## 3. Key Metrics

### Detection Metrics (per frame)
- **mAP@0.5** (mean Average Precision at IoU=0.5) – overall detection quality
- **Precision / Recall** – how many detections are correct vs. missed
- **FPS** – frames per second (system speed)

### Tracking Metrics (across sequence)
- **MOTA** (Multi-Object Tracking Accuracy) – balances misses, false positives, ID switches
- **IDF1** (ID F1 Score) – measures ID consistency over time
- **ID Switches** – how many times the tracker loses/changes the target ID
- **Track Continuity** – longest consecutive frames with same ID for target

### Re-Identification Metrics (after target exits and re-enters frame)
- **Recovery Success Rate** – % of exits where correct ID is reassigned within N frames
- **False Positive Rate on Strangers** – % of stranger detections incorrectly matched to target
- **Re-acquisition Time** – average frames needed to recover ID after exit

## 4. Baselines for Comparison
We compare our full system vs. simplified versions (ablation study):

| Baseline Name              | Description                                      | Expected Weakness in Low Light |
|----------------------------|--------------------------------------------------|--------------------------------|
| RGB-only YOLO              | Use color image instead of IR, no depth filter   | Noisy/misses in dark           |
| No Depth Filter            | Remove std dev / mean depth check                | More false positives (plants)  |
| No Ratio Check             | Remove shoulder-head ratio for re-ID             | More ID switches               |
| Our Full Method            | IR input + depth filter + ratio re-ID + gesture lock | Baseline for gains             |

## 5. Scenarios to Record (per lighting level)
For each lighting category, record 3–5 short bags (30–120 sec each):

1. **Lock & Follow** – Person raises hand to lock → walks around → exits frame → re-enters
2. **Stranger Test** – Target locked → stranger walks in/out → ensure no ID switch
3. **Occlusion Test** – Target walks behind object → reappears
4. **Fast Motion** – Target moves quickly across frame
5. **Crowd** – 2–4 people → lock one → track only target

## 6. Ground Truth for Metrics
- Manual labeling (simple): Watch video, note target ID changes, exits/re-entries, stranger detections.
- Tools: Use CVAT or LabelStudio (offline) on extracted frames/videos later if needed.
- For now: We can start with qualitative (visual inspection) + basic counts.

## 7. Next Steps After Recording
- Colleague sends bags + lux notes + short description
- I run bag_analyzer.py to extract frames
- Compute metrics → create tables/graphs in analysis/results
- Compare full method vs. baselines

Questions for colleague:
- Can you record lux levels for each bag?
- Any preferred scenarios to prioritize?