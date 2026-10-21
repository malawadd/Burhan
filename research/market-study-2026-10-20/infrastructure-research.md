# Infrastructure, environment, energy and agriculture market screen

Research date: 20 October 2026. Scope: global opportunity search; country-specific evidence is identified and never extrapolated into global market size. Constraints: no factory, customer, specialist hardware, private footage or supplied data; six days to the stated 26 October deadline. This is desk research, not customer validation. The earlier suggested concepts were not used as a shortlist.

## Decision before concepts

Four recurring needs were screened: pavement maintenance evidence, sewer inspection throughput, solar maintenance triage, and crop-health diagnosis. All have established workflows and suppliers. None supports a claim that an inexpensive camera plus AI is an unserved category. Solar provides the cleanest small public dataset; roads provide the strongest multinational benchmark. Both are conditional hackathon options, not validated businesses. Sewer and generic crop diagnosis should be rejected under present constraints.

No global TAM is calculated. Installed infrastructure, crop losses and equipment downtime are not software revenue. A credible serviceable market would require counts of reachable buyers with compatible data, purchasing authority and incremental unmet need; those are unknown.

## 1. Pavement maintenance: demand exists, generic detection is crowded

**Need and evidence.** Road managers need consistent condition records to select maintenance work. FHWA documents both automated surveys and manual/windshield methods, but its cited 44 automated versus 21 windshield agencies derive from older surveys, so they are historical workflow evidence, not a current adoption estimate. A US government SBIR record identifies a $49,677 RoadBotics award in 2020: evidence that an institution funded development, not a present customer subscription price. [FHWA guide](https://www.fhwa.dot.gov/pavement/management/qm/data_qm_guide.pdf), [SBIR award](https://www.sbir.gov/awards/187826).

**Buyer, user, payer.** Municipal public-works directors and engineering consultants buy; pavement analysts and maintenance supervisors use; municipal operating/capital budgets or consultant project budgets pay. A plausible initial segment is consultants producing evidence for small municipal clients, because one consulting relationship can cover multiple road networks. This channel advantage is an untested hypothesis.

**Competition and substitutes.**

| Supplier | Verified overlap | Public price status |
|---|---|---|
| vialytics | Smartphone road images, GPS/time, AI condition assessment, assets, priorities and work orders; claims 1,000+ communities in seven countries | Demo/sales path; no numeric current price found |
| RoadBotics | Published RoadWay subscription options include condition ratings, CSV/shapefile exports, potholes and crack classes | Available brochure is from 2021; no reliable current numeric price |
| Pavemetrics LCMS-2 | Specialist 2D+3D pavement measurement, cracks and severity, potholes, rutting and standards-related output | Hardware/service quotation; no public numeric price found |

Sources: [vialytics](https://www.vialytics.com/), [RoadBotics brochure](https://www.roadbotics.com/wp-content/uploads/2021/03/subscription-options.pdf), [Pavemetrics](https://www.pavemetrics.com/capabilities/lcms-2/cracking/). Vendor capabilities and adoption counts are supplier statements, not independent comparative tests. Manual inspections, citizen reports, spreadsheets, GIS records and specialist survey vans are alternatives. A simple detector must beat an incumbent workflow's total effort, not merely produce boxes.

**Verified organizations to investigate, not leads.** Edison, New Jersey; Metuchen, New Jersey; and Oberasbach, Germany appear as named vialytics reference users. They demonstrate buyer existence and are already served. Colliers Engineering & Design, Verdantas and BETA Group appear as vendor partners and could inform a consultant segment. No interest, budget or contact access has been established. [vialytics references and partners](https://www.vialytics.com/).

**Data and six-day feasibility.** The RDD2022 authors describe 47,420 images from six countries and more than 55,000 damage instances in four principal classes. The originating project states image licensing as CC BY-SA 4.0; its code license is separately MIT. Preserve country, attribution and license metadata; the exact downloaded archive still needs checking. The Figshare landing page did not load in this research session. The dataset measures visual detection, not road repair effectiveness, real-world severity or verified before/after outcomes. [Authors' paper](https://arxiv.org/abs/2209.08538), [originating project and license](https://github.com/sekilab/RoadDamageDetector).

**Conditional opportunity hypothesis:** an evidence-review queue for consultants that rejects unusable imagery, classifies only supported visible distress and produces a human-approved evidence packet. OpenCV 5 would perform blur/exposure checks, image transformations, region analysis, inference and duplicate checks; AWS would execute queued batch jobs, store source/derived artifacts and record reviewer decisions. The next action is a documented retake request or review task, not automatic repair dispatch. No feature here is proven absent from incumbents.

A six-day demo could use a small licensed RDD subset, source-attributed original images, clearly labeled synthetic blur/exposure perturbations and a held-out country split. It cannot truthfully demonstrate GPS localization or recurring defects without corresponding ground truth. Shadows, road markings, wet pavement, patches and perspective are major failures. Kill if country-held-out performance collapses, reviewers save less than 30% time, or two plausible buyers report their existing software already handles the workflow adequately. Time-saving thresholds are proposed validation criteria, not findings.

## 2. Sewer inspection: strong workflow evidence, poor fit without access

EPA identifies CCTV as a long-established and widely used condition-assessment method; it also describes analytic software and integration with maintenance/asset systems. This establishes an existing budgeted workstream more directly than generic water-loss totals. [EPA smart sewer technologies](https://www.epa.gov/npdes/smart-sewer-technologies).

Water utilities and sewer-network owners are economic buyers; licensed inspectors, CCTV contractors and engineering firms are users; inspection contracts and asset-management budgets pay. Primary alternatives include manual coded CCTV review, acoustic screening, flow monitoring, smoke/dye testing and physical investigation. Visual footage cannot see through water or determine surrounding soil condition. [EPA condition-assessment report](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P1008H44.TXT).

| Supplier | Verified overlap | Public price status |
|---|---|---|
| SewerAI | Inspection management, AI coding, risk/rehab planning; an assistant already produces reports/maps from inspection and GIS data | General plans quote-based; separate limited-time starter offer explicitly $25,000/year for three users and one unlimited AutoCode truck seat |
| Pallon | Cloud sewer/manhole defect assessment, reporting and sharing; claims 250+ customers | Free trial; numeric production price not found |
| VAPAR | Automated footage coding, condition assessment, cloud data and contractor/owner interface; flags obscuring water | Quote based on anticipated metres surveyed |

Sources: [SewerAI pricing](https://www.sewerai.com/pricing), [starter offer](https://go.sewerai.com/starter-bundle), [existing assistant](https://www.sewerai.com/products/command-center), [Pallon](https://www.pallon.com/), [VAPAR FAQ](https://www.vapar.co/faqs). The promotional price is neither a general market average nor guaranteed future availability.

**Prospect research targets:** Gelsenkanal, StEB Cologne and SH Power are named Pallon references. These are verified category buyers already using a supplier, not evidence of an open purchasing opportunity. Municipal procurement, coding standards, GIS/CMMS compatibility, qualified review and footage security would all require discovery. VAPAR's own scope illustrates practical friction: circular pipes up to 1,200 mm, PTZ footage, metadata/chainage overlays and limited optimization for brick. [Pallon references](https://www.pallon.com/), [VAPAR constraints](https://www.vapar.co/faqs).

Sewer-ML supplies 1,300,201 images from 75,618 videos across three Danish utilities, nine years and 17 defect classes. The university provides a dataset request form and pretrained-model link. These are image-level benchmark resources, not immediate access to a customer's complete video/asset workflow. The original Bitbucket repository and form could not be inspected successfully; GitHub mirrors report CC BY-NC-SA 4.0, so a commercial-use license has **not been independently cleared here**. Do not equate an MIT code license with data permission. [University project](https://vap.aau.dk/sewer-ml/), [CVPR paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Haurum_Sewer-ML_A_Multi-Label_Sewer_Defect_Classification_Dataset_and_Benchmark_CVPR_2021_paper.pdf).

**Decision: reject now.** A niche around footage-quality assurance was considered, but incumbents already cover QA or water obstruction and the team lacks footage, standards expertise and cleared deployment data. An OpenCV/AWS review demo is technically possible; the business wedge and deployment permission are weaker than the attractive problem statement suggests.

## 3. Solar O&M: accessible benchmark, established end-to-end competition

DOE describes routine monitoring, inspection, cleaning and corrective maintenance as operating requirements. It also separates preventive, administrative and replacement costs. Raptor Maps' 2026 report page says its analysis covers 373 GWdc; this is a vendor data footprint, not the addressable market for a new inspection product. Its 2025 report describes 193 GWdc analyzed, with 67 GWdc during 2024. These figures demonstrate industrial activity but are subject to vendor selection and commercial incentives. [DOE O&M guidance](https://www.energy.gov/cmei/femp/life-cycle-photovoltaic-systems-operate-and-maintain-existing-photovoltaic-system), [Raptor report descriptions](https://raptormaps.com/resources/reports).

Asset managers, independent power producers, O&M contractors and EPC warranty teams buy; thermographers and technicians use; asset operating/warranty budgets pay. SCADA/inverter telemetry, electrical tests, manual thermal analysis and contracted drone surveys compete with vision software. A camera finding does not establish root cause or recoverable energy loss without context.

| Supplier | Verified overlap | Public price status |
|---|---|---|
| Raptor Maps | Thermal/visual analytics, equipment-localized digital twins, field workflows and automated investigations | Contact/demo; no current public numeric price found |
| Sitemark | Module thermography, connectors, vegetation, soiling, trackers, damage and severity/estimated loss prioritization | Sales-led; older official brochure says per-MW pricing, no verified current number |
| Aerodyne / vertikalitiSOLAR | Drone thermal inspection, module tracking, shading analysis and cloud management | Contact sales; no public numeric price found |

Sources: [Raptor inspections](https://raptormaps.com/products/solar-inspections-analytics), [Sitemark inspections](https://www.sitemark.com/solutions/solar-inspections/), [Aerodyne infrastructure](https://aerodyne.group/infra.html), [older Sitemark brochure](https://f.hubspotusercontent30.net/hubfs/6021737/Marketing%20Materials/Corporate%20Brochure/Corporate%20Overview.pdf). Do not use advertised savings as verified causal ROI.

Sonnedix, EnBW and Omexom are named Sitemark users/references. They are potential interview targets, with existing supplier relationships; there is no evidence they want another tool. Buying friction includes thermal capture specifications, repeatability, equipment identity, warranty evidence and integration with field work. [Sitemark customer references](https://www.sitemark.com/solutions/solar-inspections/).

Raptor's public repository contains 20,000 infrared module crops at only 24×40 pixels and 12 classes, including 10,000 normal images. The repository includes an MIT license. This is an unusually feasible laptop benchmark, but it does not provide full-site localization, calibrated temperature, financial outcomes or chronological repair verification. [Dataset](https://github.com/RaptorMaps/InfraredSolarModules), [license](https://github.com/RaptorMaps/InfraredSolarModules/blob/master/LICENSE).

**Conditional opportunity hypothesis:** an auditable second-review workbench for small thermal inspection contractors, with an abstention queue, anomaly crops and reviewer corrections exported into an existing report. OpenCV 5 can implement intensity/shape descriptors, preprocessing and model inference; AWS can host queued inference, originals and audit records. The genuine next action is requesting another measurement or human review. A full solar digital twin would be implausible in six days.

Kill if a simple normal-versus-anomaly model cannot achieve proposed 90% precision at useful recall on held-out data, if module/site leakage cannot be assessed, or if contractor interviews show report integration consumes more time than review saves. Benchmarks on tiny crops cannot validate radiometric diagnosis. This is a reserve concept, with high overlap and no demonstrated defensibility.

## 4. Crop-health diagnosis: reject generic consumer app

FAO's 2021 estimate that pests destroy up to 40% of crops establishes broad damage context only. It does not tell us which visible symptoms are preventable, who pays for software or how much they would pay. [FAO](https://www.fao.org/newsroom/detail/Climate-change-fans-spread-of-pests-and-threatens-plants-and-crops-new-FAO-study/ar).

Plantix already offers free photo diagnosis and treatment suggestions; its B2B page claims eight million yearly active users and 150 million images. Kindwise crop.health offers API diagnosis, with public credit pricing from €0.05 to €0.01 by volume and 100 trial credits. Cropin sells enterprise monitoring and disease warnings; it names PepsiCo, McCain and East-West Seed among customers. Those organizations verify buyer categories, not demand for this team. [Plantix](https://www.plantix.net/en/), [B2B scale](https://www.plantix.net/en/plantix-intelligence/), [Kindwise pricing](https://www.kindwise.com/pricing), [Cropin customers](https://www.cropin.com/customers/).

Farmers use diagnostics, but agribusiness, insurers, extension programs or input distributors may pay. That payer/user split needs validation. Agronomists, lab testing, weather models and field scouting are substitutes. A leaf image cannot reliably distinguish every nutrient, pest and pathogen cause.

PlantVillage has over 54,000 controlled leaf images. Its original study obtained about 99% internal test accuracy but only 31.4% on external images, demonstrating domain-shift risk rather than current performance ceilings. Original image permissions must be inspected before use; no mirror's license should substitute. [Dataset authors](https://github.com/spMohanty/PlantVillage-Dataset), [original experiment](https://arxiv.org/abs/1604.03169).

**Decision: reject.** Without field access, an agronomist and a specific crop/buyer workflow, a six-day classifier would offer weak differentiation against free diagnosis and cheap APIs. Neither a chatbot nor AWS hosting fixes the commercial gap.

## Economics and validation boundaries

All unit economics below are assumptions to test, not market prices: at $30/hour review labor, saving four hours yields $120 per job. A $30/job price captures 25% of that gross benefit before onboarding, support, error correction and cloud costs. If only 30 minutes is saved, gross benefit falls to $15 and the example price fails. For 20 jobs/month the first assumption yields $600 monthly revenue per contractor; this is a scenario, not a forecast. Do not price based on total energy/crop/infrastructure losses.

Before building commercially: interview at least five relevant buyers, observe two actual workflows and compare against their incumbent and manual process. Ask for a bounded retrospective evaluation with representative licensed images and measured time, not hypothetical enthusiasm. Require a named payer, an integration path and a reason an existing tool is insufficient. No outreach was conducted or authorized. Hackathon success can be demonstrated in six days; purchase intent and field reliability cannot.

## Evidence ledger

All sources accessed 20 October 2026; undated live vendor pages may change. Inline links above are the corresponding URLs.

| Source/title | Publication date | Claim supported | Limitation |
|---|---|---|---|
| [FHWA, Practical Guide](https://www.fhwa.dot.gov/pavement/management/qm/data_qm_guide.pdf) | 2013 | Mixed manual/automated workflows | Historical agency survey |
| [SBIR road assessment award](https://www.sbir.gov/awards/187826) | Award year 2020 | $49,677 funded project | R&D award, not recurring spend |
| [vialytics homepage](https://www.vialytics.com/) | Undated | Product overlap, named users, supplier adoption claims | Vendor self-report |
| [RoadBotics subscription options](https://www.roadbotics.com/wp-content/uploads/2021/03/subscription-options.pdf) | March 2021 | Existing distress/workflow features | Current packaging unknown |
| [Pavemetrics cracking](https://www.pavemetrics.com/capabilities/lcms-2/cracking/) | Undated | 2D+3D specialist alternative | Vendor claims |
| [Arya et al., RDD2022](https://arxiv.org/abs/2209.08538) | 2022 preprint; 2024 journal | Six-country dataset and classes | Benchmark, not deployed ROI |
| [Sekilab RoadDamageDetector](https://github.com/sekilab/RoadDamageDetector) | Live repository | Data/code license distinction | Verify exact download |
| [EPA Smart Sewer Technologies](https://www.epa.gov/npdes/smart-sewer-technologies) | Undated live page | CCTV and analytics workflow | No startup buying intention |
| [EPA Condition Assessment](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P1008H44.TXT) | 2010 | Visual limits and alternatives | Historical technical review |
| [SewerAI pricing](https://www.sewerai.com/pricing), [starter offer](https://go.sewerai.com/starter-bundle) | Undated | Pricing structure and $25K/year promotion | Limited offer; vendor claims |
| [SewerAI Command Center](https://www.sewerai.com/products/command-center) | Undated | Existing AI reporting assistant | No independent performance test |
| [Pallon homepage](https://www.pallon.com/) | Undated | Competing service and named utilities | Vendor references |
| [VAPAR FAQ](https://www.vapar.co/faqs) | Undated | Per-metre pricing, capture constraints | Supplier-specific scope |
| [Aalborg Sewer-ML](https://vap.aau.dk/sewer-ml/) | 2021 | Dataset provenance, size, standards | Form access; license not cleared |
| [DOE PV O&M](https://www.energy.gov/cmei/femp/life-cycle-photovoltaic-systems-operate-and-maintain-existing-photovoltaic-system) | Undated live page | Maintenance tasks and procurement workflow | Primarily US federal context |
| [Raptor Maps reports](https://raptormaps.com/resources/reports) | 2026 / 2025 editions | Analyzed fleet footprint | Commercially selected sample |
| [Raptor](https://raptormaps.com/products/solar-inspections-analytics) / [Sitemark](https://www.sitemark.com/solutions/solar-inspections/) / [Aerodyne](https://aerodyne.group/infra.html) | Undated | Competitive coverage | Prices mostly unpublished |
| [Raptor dataset](https://github.com/RaptorMaps/InfraredSolarModules) / [LICENSE](https://github.com/RaptorMaps/InfraredSolarModules/blob/master/LICENSE) | 2020 | 20K tiny IR images, MIT repository | No outcome or site context |
| [FAO plant-pest study](https://www.fao.org/newsroom/detail/Climate-change-fans-spread-of-pests-and-threatens-plants-and-crops-new-FAO-study/ar) | 2 June 2021 | Broad crop damage context | Not serviceable market |
| [Plantix](https://www.plantix.net/en/), [B2B](https://www.plantix.net/en/plantix-intelligence/) | Undated | Free consumer alternative and scale claims | Supplier self-report |
| [Kindwise Pricing](https://www.kindwise.com/pricing) | Undated | Current published API credit tiers | No calls purchased |
| [Cropin Customers](https://www.cropin.com/customers/) | Undated | Named enterprise buyer categories | Supplier case studies |
| [Mohanty plant disease study](https://arxiv.org/abs/1604.03169) | 2016 | Internal/external generalization gap | Historical model, not modern ceiling |

