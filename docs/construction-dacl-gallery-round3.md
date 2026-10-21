# Real construction conditions: original photo, annotation and local model

20 October 2026. These are 14 real DACL10k inspection photographs, not videos. Each image has separate **RAW CAMERA PHOTO / PUBLISHER ANNOTATION / ACTUAL LOCAL MODEL MASK** panels. Orange in the annotation panel is the publisher label; orange in the model panel comes from actual local DACL512 inference. Their shapes can differ or be absent. No damage was generated or added to the source photos.

For each label, the example is the first GT-positive photo in the previously frozen55-photo inventory. It was not selected for a favorable model result. Failures and partial masks remain visible. Pixel precision/recall/IoU below concern that named class on that single original-resolution photo. Region matches use one-to-one connected components at IoU0.50, not native instance labels. The model uses fixed probability>0.5 and at least four native-grid pixels per component.

The public-validation photos may overlap checkpoint training/selection; bridge-disjoint evaluation is not established. These examples are not construction acceptance, calibrated severity, hidden-defect detection or video-validation evidence. Full55-photo results and source/model licensing are in [the measured report](<C:/Users/IAM/Documents/hackathon/opencv/project1/docs/construction-category-expansion-round3.md>). Data: original-author CC BY-NC4.0; checkpoint license separately undeclared.

## Crack

Original `dacl10k_v2_validation_0055`. Pixel precision **0.0%**, recall **0.0%**, IoU **0.0%**; derived-region matches **0/1** at IoU0.50; predicted regions **2**. **Less than half the labeled pixels were covered.**

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0055.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0055.json>)

![Crack: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/crack-raw-annotation-model.jpg>)

## Alligator crack

Original `dacl10k_v2_validation_0231`. Pixel precision **43.9%**, recall **99.8%**, IoU **43.9%**; derived-region matches **0/2** at IoU0.50; predicted regions **1**. At least half the labeled pixels were covered.

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0231.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0231.json>)

![Alligator crack: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/alligator_crack-raw-annotation-model.jpg>)

## Wetspot

Original `dacl10k_v2_validation_0046`. Pixel precision **96.5%**, recall **27.7%**, IoU **27.4%**; derived-region matches **0/1** at IoU0.50; predicted regions **1**. **Less than half the labeled pixels were covered.**

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0046.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0046.json>)

![Wetspot: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/wetspot-raw-annotation-model.jpg>)

## Efflorescence

Original `dacl10k_v2_validation_0046`. Pixel precision **91.2%**, recall **18.6%**, IoU **18.3%**; derived-region matches **0/5** at IoU0.50; predicted regions **6**. **Less than half the labeled pixels were covered.**

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0046.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0046.json>)

![Efflorescence: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/efflorescence-raw-annotation-model.jpg>)

## Rust

Original `dacl10k_v2_validation_0047`. Pixel precision **65.1%**, recall **78.0%**, IoU **55.0%**; derived-region matches **3/7** at IoU0.50; predicted regions **8**. At least half the labeled pixels were covered.

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0047.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0047.json>)

![Rust: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/rust-raw-annotation-model.jpg>)

## Rockpocket

Original `dacl10k_v2_validation_0035`. Pixel precision **25.0%**, recall **57.9%**, IoU **21.2%**; derived-region matches **1/3** at IoU0.50; predicted regions **7**. At least half the labeled pixels were covered.

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0035.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0035.json>)

![Rockpocket: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/rockpocket-raw-annotation-model.jpg>)

## Inspector chalk boundary after sounding

Original `dacl10k_v2_validation_0047`. Pixel precision **87.2%**, recall **46.9%**, IoU **43.9%**; derived-region matches **0/1** at IoU0.50; predicted regions **3**. **Less than half the labeled pixels were covered.**

The target is an inspector's visible chalk/crayon outline. The model does not see or prove an internal void.

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0047.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0047.json>)

![Inspector chalk boundary after sounding: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/hollowareas-raw-annotation-model.jpg>)

## Cavity

Original `dacl10k_v2_validation_0035`. Pixel precision **3.3%**, recall **99.6%**, IoU **3.3%**; derived-region matches **1/4** at IoU0.50; predicted regions **20**. At least half the labeled pixels were covered.

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0035.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0035.json>)

![Cavity: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/cavity-raw-annotation-model.jpg>)

## Spalling

Original `dacl10k_v2_validation_0035`. Pixel precision **60.3%**, recall **20.5%**, IoU **18.1%**; derived-region matches **0/4** at IoU0.50; predicted regions **6**. **Less than half the labeled pixels were covered.**

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0035.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0035.json>)

![Spalling: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/spalling-raw-annotation-model.jpg>)

## Graffiti

Original `dacl10k_v2_validation_0121`. Pixel precision **69.3%**, recall **38.5%**, IoU **32.9%**; derived-region matches **0/2** at IoU0.50; predicted regions **1**. **Less than half the labeled pixels were covered.**

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0121.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0121.json>)

![Graffiti: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/graffiti-raw-annotation-model.jpg>)

## Weathering

Original `dacl10k_v2_validation_0035`. Pixel precision **2.9%**, recall **9.5%**, IoU **2.3%**; derived-region matches **0/1** at IoU0.50; predicted regions **11**. **Less than half the labeled pixels were covered.**

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0035.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0035.json>)

![Weathering: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/weathering-raw-annotation-model.jpg>)

## Residual formwork

Original `dacl10k_v2_validation_0098`. Pixel precision **54.3%**, recall **51.7%**, IoU **36.0%**; derived-region matches **0/1** at IoU0.50; predicted regions **3**. At least half the labeled pixels were covered.

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0098.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0098.json>)

![Residual formwork: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/residual_formwork-raw-annotation-model.jpg>)

## Exposed rebar

Original `dacl10k_v2_validation_0060`. Pixel precision **66.3%**, recall **87.9%**, IoU **60.8%**; derived-region matches **1/1** at IoU0.50; predicted regions **2**. At least half the labeled pixels were covered.

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0060.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0060.json>)

![Exposed rebar: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/exposed_rebar-raw-annotation-model.jpg>)

## Concrete surface loss / erosion appearance

Original `dacl10k_v2_validation_0055`. Pixel precision **87.6%**, recall **26.5%**, IoU **25.6%**; derived-region matches **0/3** at IoU0.50; predicted regions **4**. **Less than half the labeled pixels were covered.**

The label groups several surface-loss appearances. It does not determine the physical cause or prove chemical corrosion.

[Original photo](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/images/damage_diagnostic/dacl10k_v2_validation_0055.jpg>) · [Publisher polygons](<C:/Users/IAM/Documents/hackathon/opencv/project1/datasets/construction/dacl10k/annotations/damage_diagnostic/dacl10k_v2_validation_0055.json>)

![Concrete surface loss / erosion appearance: raw camera photo, publisher annotation, actual local model mask](<C:/Users/IAM/Documents/hackathon/opencv/project1/output/evaluation/construction-dacl19-round3/512/washout-raw-annotation-model.jpg>)
