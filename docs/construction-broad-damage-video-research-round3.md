# Broad construction damage videos: verified availability and gap

20 October 2026. Bounded primary-source search while local training ran; no NN, full archives, arbitrary video downloads or active configuration changes. The target was genuine recorded rust, exposed reinforcement, rockpocket/honeycomb, moisture or efflorescence footage, preferably with publisher frame-localized ground truth. **No new broad-category video was promoted as a validated local detection case.** Real photos and predictions drawn by another model cannot establish that result.

## Existing Kansas source: real raw footage, image GT and model outputs are separate

The [author's curated Zenodo record19268973](https://zenodo.org/records/19268973) declares CC BY4.0 and separates manually labeled inspection photographs, predicted photographs and raw bridge videos. I verified the publisher API and both relevant ZIP central directories using byte ranges:

| Archive | Published bytes | Actual directory inspection |
|---|---:|---|
| `01_curated_dataset_images_annotations.zip` | 1,591,971,740 | 1,326 image/TXT entries; generic `bd_img_*.jpg/.txt` names |
| `02_curated_inference_results_images.zip` | 94,203,843 | Publisher identifies these as inference outputs, not manual GT |
| `03_curated_dataset_bridge_videos.zip` | 18,463,440,612 | 39 raw MP4 entries; no frame-GT sidecar or image-to-video/timestamp crosswalk found |

The named defect vocabulary includes crack, alligator crack, efflorescence, surface corrosion/washout, spalling, wetspot, rust and exposed reinforcement. This vocabulary does not prove every raw video contains each condition. The smallest raw video is `bd_vid_0031.mp4` at151,145,752bytes, already acquired earlier; remaining members are larger. No new class-specific pixels were verified in this metadata pass, so no additional random raw file was downloaded or presented as damage evidence.

The [older author record17477702](https://zenodo.org/records/17477702) is also CC BY4.0. Its three video ZIPs total roughly25.3GB and are explicitly **inference videos**. Overlaid rust/rebar/efflorescence polygons there are predictions from the author's YOLO model, not independent frame annotations and not our local model output. They cannot become ground truth by copying them. The [USDOT dataset record](https://rosap.ntl.bts.gov/view/dot/88076) links both releases and confirms that distinction between visual prediction outputs and the curated supporting release.

The [author's training/inference repository](https://github.com/natnaeltaye/concrete-defect-yolov11) documents generated polygon/confidence/count exports and video tracking/stride filtering. It references an on-machine `best.pt` path and YOLO11x segmentation training; this search did not verify an immediately downloadable small trained checkpoint or manual temporal labels. A repository screenshot of predictions would not close that gap.

Saved factual artifacts: `datasets/construction/broad-video-research/zenodo-19268973-record.json` and `zenodo-19268973-zip-member-inventory.json`. They contain original public metadata, member names, sizes and CRCs. No raw-video payload or inference-video archive was acquired in this pass.

## Other checked primary sources

| Primary source | Relevant actual material | Why it does not yet provide the requested video evidence |
|---|---|---|
| [GYU-DET author data descriptor](https://www.nature.com/articles/s41597-025-05395-w) | Real annotated bridge **images**; six classes including honeycomb, seepage and exposed rebar | The released descriptor establishes still-image boxes, not recorded sequences with temporal/frame GT. No still-to-video conversion performed. |
| [Hydraulic-concrete dataset authors](https://pmc.ncbi.nlm.nih.gov/articles/PMC12900026/) | Crack, spalling, efflorescence, corrosion, exposed rebar and exposed aggregate images | Authors explicitly describe discrete photography rather than video acquisition. Does not validate moving-camera behavior. |
| [IBM CiF author project](https://ibm.github.io/cracks-in-the-foundation-dataset/) | Real infrastructure image segmentation; rust, precipitation/efflorescence and other conditions; CDLA Permissive2.0 stated by project | Photo collection, not a verified timestamped recorded-video release in this search. No large acquisition attempted. |
| [Chinese national-center bridge apparent-disease catalog](https://nbsdc.cn/general/dataDetail?id=60a621eafd80d22c90145b41&type=1) | Search-visible catalog describes a71.88GB bridge image/video archive including exposed reinforcement/roughness | Direct rendered metadata did not establish a usable download, explicit license or timestamped label format. Remains an unverified lead, not an acquired dataset. |

These are candidates for future source-specific work, not newly implemented category capabilities. No contact messages, account registrations or requests to third parties were sent.

## Current usable conclusion

Genuine public raw multi-condition bridge footage exists under a usable attribution license, but **the verified Kansas raw-video directory does not supply a per-frame manual localization benchmark**. Separately published predictions are not that benchmark. The inspected alternative broader datasets mainly provide photographs. This bounded search did not verify a small ready-to-run recorded rust/rebar/efflorescence/honeycomb/moisture sequence with independent publisher temporal GT.

The local broad DACL model therefore retains its measured **photo-only**, weak-localization scope. Actual crack/spalling camera sequences and their publisher boxes are being evaluated separately. A future raw-video review could expose new real conditions, but it must first inspect actual pixels and establish independent annotations/scope; source-level class names alone do not justify a successful detection claim. Nothing was added to the UI as an inspection demonstration from this research, and no photos were encoded into a slideshow.
