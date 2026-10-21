# Saudi construction inspection: code research and video evidence plan

Research date: **20 October 2026**. Scope: **both low-rise residential and commercial buildings**, with differences marked. This study covers **89 inspection families across 16 lifecycle stages**. It supplies a traceable checklist and evidence requirements. It does **not** establish that 89 conditions can be detected by the installed models, or that a building complies with Saudi law.

The full stage-by-stage list is in [saudi-construction-checklist.md](saudi-construction-checklist.md), backed by [the machine-readable catalog](../edge_runtime/data/construction-inspections.json). All rows initially remain `unassessed`; `research_only` describes the checklist's automation state. A model observation is evidence for review, never an occupancy approval or a code pass.

## What was verified

The [official 2024 SBC index](https://sbc.gov.sa/en/BC/Pages/BuildingCode/BC.aspx?year=2024) identifies the relevant code families. The full official Arabic **SBC 201-AR-2024** was downloaded successfully: 593 PDF pages, SHA-256 `237070fe2784a3e7c4663740718cfec841fa6675eb558d8a9fd2c84dc105a8fc`. Its actual inspection provisions were read; the Arabic extraction occasionally reverses numbers and renders ligatures imperfectly, so section references were checked in context. The source PDF and UTF-8 extraction are retained under `output/research/` as research material.

For SBC 303, 304, 401, 701 and 801, the downloaded public files are **tables of contents**, not the complete clause books. Their headings establish topic and section locations, not acceptance limits or universal applicability. The SBC 501 official product description establishes mechanical scope; its complete clauses were not acquired. The catalog records these differences per source and does not invent numerical tolerances.

The [2019 MOMAH technical inspector guide](https://momah.gov.sa/sites/default/files/2020-03/%D8%AF%D9%84%D9%8A%D9%84%20%D8%A7%D9%84%D9%85%D9%81%D8%AA%D8%B4%20%D8%A7%D9%84%D9%81%D9%86%D9%8A%20%D9%84%D9%84%D9%85%D8%A8%D8%A7%D9%86%D9%8A.pdf) was read as historical workflow guidance. It distinguishes initial, during-construction and final inspection and includes site photographs, records, measurements and tests. It explicitly refers to the 2018 code; it is not treated as an updated 2024 clause book.

The staged checklist is an **engineering evidence taxonomy inferred from these verified topics**. Subdividing a topic into rebar cover, lap details, delivery traceability or other review tasks does not assert a new Saudi statutory requirement. Applicability depends on occupancy, structure, location, system, permit, controlled code text and the responsible authority.

## Low-rise residential and commercial scope

The product's research plan covers **both**, rather than choosing one building type. Every checklist row now contains separate `building_applicability.low_rise_residential` and `building_applicability.commercial` notes. The notes route evidence collection; they do not declare a requirement applicable, waive it, or determine compliance. All **89** rows remain **unassessed / research_only**.

The [verified official SBC1101–1102 scope description](https://sbc.gov.sa/ar/BC/Pages/BuildingCode/BCDetails.aspx?codeId=1101&year=2024) concerns qualifying **one- or two-family dwellings no more than three stories**, with integrated disciplines. “Low-rise residential” alone is therefore insufficient to select that route: apartments, other residential occupancies and mixed-use buildings do not automatically qualify. Full scope conditions, exceptions and the permitted project's authority decision still need checking.

For commercial work, determine the actual occupancy, height, structural system and installed services, then route through **SBC201 and the applicable discipline families**. A shop, office, restaurant or other commercial use is not one interchangeable code condition. Mixed-use buildings may require different treatment for different portions. The [official suite index](https://sbc.gov.sa/en/BC/Pages/BuildingCode/BC.aspx?year=2024) establishes family identities; it does not supply a project-specific applicability ruling.

| Inspection area | Low-rise residential | Commercial | What is shared / what differs |
|---|---|---|---|
| Permit and design basis | Confirm eligibility for the residential code route; do not classify by height alone | Confirm actual commercial/mixed occupancy and discipline route | Common need for genuine approved documents; applicable criteria differ with scope |
| Site establishment and temporary works | Actual work hazards, public interfaces and support conditions still require review | Actual workforce, plant, access and neighboring/public use affect the site plan | Same physical hazard/evidence questions; no residential exemption or commercial universal rule is inferred |
| Soil, earthworks and foundations | Actual ground and approved dwelling design govern; piles/special ground are conditional | Actual ground, loads and approved commercial foundation system govern | Shared measurements/tests; building label does not establish soil or bearing acceptance |
| Concrete, rebar and structural materials | Inspect the system actually specified under the confirmed residential/discipline route | Inspect the commercial structural design and applicable material-code route | Shared physical quality tasks; detailing, loading and special inspections are project-dependent |
| Fire separation, escape and protection | Determine the dwelling's applicable provisions and any rated assemblies; no blanket waiver | Determine occupancy/height/hazard-specific fire strategy and relevant SBC201/SBC801 provisions | System triggers, ratings, alarms and escape criteria can differ; no unverified trigger thresholds are supplied |
| Accessibility | Distinguish private dwelling space from shared/common or public-facing areas; confirm actual scope | Review relevant public/customer/staff routes and facilities for actual use | Do not assume either a universal residential exemption or identical commercial obligations |
| Energy and insulation | Confirm SBC602 and/or the qualifying integrated residential provisions and referenced requirements | SBC601 is the nonresidential energy-family reference; mixed-use portions need separate review | Different family routing is verified; numerical thermal/controls/commissioning rules were not fully verified |
| Electrical, mechanical and plumbing | Inspect installed dwelling systems under the applicable residential/referenced discipline route | Actual activity, demand, equipment and hazards govern the discipline design | Shared before-cover and real-test evidence; commercial use does not imply one standard system size/layout |
| Gas, lifts, smoke control and specialist systems | Conditional on installed/required systems and confirmed scope | Conditional on actual commercial use, building configuration and approved systems | Presence and test scope must be established; no system is assumed merely from either label |
| Waterproofing, envelope and finishes | Inspect the actual dwelling assemblies and specified acceptance criteria | Inspect actual commercial assemblies; location/use can change fire, access or durability criteria | Shared need for genuine inspection, measurements and leak/integrity evidence; no visual-only acceptance |
| Commissioning and special inspections | Actual required dwelling systems and structural/material triggers determine scope | Actual commercial systems, interfaces and structural/material triggers determine scope | No generic public video establishes the entire test plan for either type |
| Handover, occupancy and later changes | Applicable residential permit/report process and actual authority-issued approval | Applicable commercial engineering reports, discipline approvals and municipal acceptance | Both require real project evidence; repairs/change of use need fresh scope review |

These distinctions are **evidence-planning interpretations of verified official scope/topic descriptions**. Exact acceptance clauses were not fully verified for every dwelling or commercial scenario. Project/authority check is required before applying a criterion. The [MOMAH permit service](https://momah.gov.sa/ar/e-services/issuance-building-license), [occupancy service](https://momah.gov.sa/ar/e-services/issuing-occupancy-certificate) and [published implementing regulations](https://www.uqn.gov.sa/details?p=18175) support the shared requirement for competent project documentation and approval; a camera finding replaces none of it.

The generated catalog adds `applicability_basis` with one of `common_lifecycle`, `system_or_design_dependent` or `occupancy_height_or_scope_dependent`. It is a planning classification, **not a boolean applicability or approval result**. The UI can display both text notes without discarding either building scope. The detailed checklist now has separate residential and commercial columns in every stage.

## Edition and regulatory status

Use the official 2024 suite as the research baseline. The official SBC `GETCODE2025` brochure is search-indexed with ministerial decision **181, dated 31 December 2024**, and a **30 June 2025** transition date. Its live URL returned HTTP 404 during this work; the date is therefore an indexed official statement, **not independently confirmed current project applicability**. Record the permitted project's governing edition and authority instructions before configuring acceptance criteria. [Official brochure reference](https://sbc.gov.sa/ar/ContactUS/Documents/GETCODE2025.pdf)

The [Official Gazette implementing regulations](https://www.uqn.gov.sa/details?p=18175) connect permits to approved documents and qualified practitioners, inspections to construction stages, and occupancy to final approvals and as-built records. Relevant articles reviewed include 13–17 and 21–22. This research did not exhaustively audit subsequent amendments.

An unresolved procedural difference must remain visible: that published regulation's **Article 21** allows transition between construction stages without prior approval, while **SBC 201-AR-2024 §110.6** describes approval before proceeding past the inspection stage. The application must not decide the legal hierarchy or automatically authorize/stop construction from these passages. The project's competent reviewer and authority must resolve the procedure and record the applicable instructions. [SBC 201 official Arabic text](https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20201/SBC201_AR2024.pdf)

The [building permit service](https://momah.gov.sa/ar/e-services/issuance-building-license) lists a survey decision, design and supervising offices, contractor, soil report and other applicable prerequisites. The [occupancy service](https://momah.gov.sa/ar/e-services/issuing-occupancy-certificate) includes an engineering report and municipal approval. Neither service can be replaced by a computer-vision result.

## Verified inspection-stage anchors

The following SBC 201 section identifiers were checked against the full 2024 Arabic document. The labels below are abbreviated topic identifiers; the controlled text determines timing and exceptions.

| Section | Topic |
|---|---|
| 110.2 | Initial inspection |
| 110.3.1 | Footings and foundations |
| 110.3.2 | Concrete slab and underfloor work |
| 110.3.3 | Reserved numbering; no invented inspection stage |
| 110.3.4 | Frame |
| 110.3.5 | Lath/gypsum assemblies, with stated exceptions |
| 110.3.6 | Fire/smoke penetrations and joints |
| 110.3.7 | Energy efficiency |
| 110.3.8 | Additional inspections |
| 110.3.9 | Plumbing, mechanical, gas and electrical rough-in |
| 110.3.10 | Special inspections; chapter 17 |
| 110.3.11 | Final inspection |
| 111 | Occupancy certificate |

Source: [SBC 201-AR-2024, printed pages 20–23](https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20201/SBC201_AR2024.pdf). This mapping is deliberately concise; the catalog's expanded tasks are original review planning, not a reproduction of the code.

## Coverage across the building lifecycle

The code families below are topical routing references. They are not a determination that every listed volume or every example applies to each project.

| Stage | Inspection scope | Relevant family routing | What video can contribute | Evidence video cannot establish alone |
|---|---|---|---|---|
| Preconstruction | Permit, classification, survey, ground report, coordinated design, material approvals | 201, 301, 302, 303; applicable discipline volumes | Site condition record | Document validity, calculations, authority approval, bearing capacity |
| Site safety throughout work | PPE, public separation, openings, access, temporary services, fire arrangements | 201/801 construction safeguards; electrical/plumbing where applicable | Visible equipment and hazard candidates | PPE suitability, anchor strength, electrical safety, equipment certification |
| Earthworks | Excavation, support, founding level, water, engineered fill, buried routes | 303; 201; 701 for plumbing trenches | Exposed conditions and before-cover sequence | Soil properties, compaction, shoring capacity, survey coordinates |
| Foundations | Footings, piles, below-ground moisture protection, slab interfaces, anchors | 303, 304; 201 | Accessible surface and installation evidence | Pile integrity, bearing acceptance, concealed defects, leak resistance |
| Reinforcement | Identity, layout, cover, laps, coupling, condition, prestressing | 304; 201 | Close visible bar arrangement before pouring | Grade, exact uncalibrated spacing, development adequacy, tendon force |
| Formwork | Geometry, support, cleanliness, striking/reshoring | 304; 302 | Visible preparation and gross issues | Stability, loads, exact alignment, release strength |
| Concreting | Mix traceability, fresh tests, placement, curing, hardened acceptance, surface defects | 304; 201 | Placement sequence and visible crack/defect candidates | Compressive strength, internal voids, crack depth, chemical properties |
| Structural frame | Floor geometry, steel, masonry, precast, visible damage | 201; 305–309; 312 as applicable | Visible component and damage record | Weld integrity, bolt pretension, load capacity, structural significance |
| Envelope and waterproofing | Roof, wet areas, cladding, glazing, insulation, drainage | 201; 303; 601/602; 701 | Exposed installation continuity and visible damage | Thermal resistance, watertightness, anchor capacity, hidden membrane continuity |
| MEP rough-in | Electrical, earthing, water, drainage, HVAC, gas, telecom | 401, 501, 701, 1201; telecom guidance | Before-cover routes and visible arrangement | Ground resistance, pressure/leak tests, airflows, gas integrity, circuit protection |
| Fire and life safety | Firestops, rated doors, exits, suppression, alarms, smoke control | 201, 801 | Obstruction and visible installation evidence | Listing compatibility, rated performance, hydraulic capacity, integrated function |
| Accessibility | Routes, ramps, parking, doors, sanitary layouts | 201; 701 | Route documentation and candidate obstruction | Precise slopes, operating forces, complete dimensional compliance |
| Finishes | Rated partitions, surface workmanship, material identity, stairs/guards | 201; approved specifications | Visible surface defects and missing finishes | Fire classification, hidden fastening, measured flatness, guard strength |
| Testing and commissioning | Electrical, plumbing, balancing, controls, integrated fire, lifts, special inspections | Applicable discipline code; 201 chapter 17 | Witness record tied to actual test | Calibrated measurements, test validity, complete system acceptance |
| Handover and occupancy | Finding closure, as-builts, site completion, occupancy approval | 201; municipal process | Final condition and corrective-work record | Issued certificate, approved documents, authority signoff |
| Operation and changes | Periodic verification, repairs, alterations, specialist occupancies | 901, 310; 902/1001/702/1401 when applicable | Visible change and defect history | Continued fitness, structural/fire reassessment, legal change of use |

The [official family index](https://sbc.gov.sa/en/BC/Pages/BuildingCode/BC.aspx?year=2024) also includes specialist subjects, including GFRP concrete and innovative residential systems. **SBC 1101–1102 is not a universal all-buildings shortcut**: the [official residential-code description](https://sbc.gov.sa/ar/BC/Pages/BuildingCode/BCDetails.aspx?codeId=1101&year=2024) limits its stated scope to qualifying one- or two-family dwellings up to three stories. Specialist occupancies, local planning conditions and authority directions require project-specific extensions to this checklist.

## Model scope and evidence gaps

The runtime now offers visible PPE observations (`ppe_hardhat`, `ppe_vest`, `person`, `site_equipment`) and a surface crack candidate (`surface_crack`). The [actual model/data verification report](construction-vision.md) documents executable benchmarks and significant real-video misses. These narrow observations do not complete the broader inspection families, so the legal-research catalog remains `research_only` and unassessed. The [data-gap study](construction-data-gap-study.md) maps every potentially visual family to actual public data/model availability and missing evidence.

The relevant catalog rows are `ppe-head-body` and `surface-defects`. A helmet or vest label is not proof that equipment is correctly certified or suitable. Explicit model classes such as `NO-Hardhat` may raise a review observation; failure to see a helmet does not itself establish a violation. A surface-crack label does not establish crack width, depth, activity, cause, concrete strength or structural acceptability. No crack finding is automatically mapped to honeycombing, spalling or exposed steel unless separately trained and evaluated classes exist.

Road pavement detection is not evidence of building inspection performance. Factory surface inspection is not evidence of concrete inspection performance. Photo compatibility checks are not video recall measurements, and geographically different public footage is not Saudi site qualification. A clean output means **no model finding**, not all checklist families passed.

## Real video application and data requirements

Use actual camera, drone or phone footage with original frame timing and known provenance. Do not generate a clip from repeated still images and call it real site video. Public images can support a clearly labelled image diagnostic, but not temporal performance or missed-event claims. Synthetic boxes, staged graphics and simulated scene footage do not count as evidence for this request.

The working capture/evidence pipeline should preserve:

1. Source URL or local source identifier, owner, license/permission, download date and SHA-256; actual camera timestamp when available.
2. Original video duration, frame count, resolution, frame rate, codec and orientation; inference resize and sampled frame indices separately.
3. Project, stage, physical location/element and inspection time. An element that will be hidden needs evidence before concealment and links to relevant test records.
4. Original and annotated video or source-time references; model name/hash, thresholds, per-frame findings and confidence; recording and storage policy.
5. Independent labels from competent reviewers, with class definitions, visibility/occlusion and uncertain labels. Associate dimensional claims with calibrated scale and measurement uncertainty.
6. Inspection status that distinguishes `unassessed`, observed candidate, reviewer-confirmed finding, test evidence required and authority approval. Never convert unsupported stages to pass.

The following data still has to exist for credible coverage beyond initial visible classes:

| Family | Required real data for model development and evaluation | Separate truth needed |
|---|---|---|
| PPE/site observations | Diverse continuous site videos, multiple workers, occlusion, day/night, dusty and distant views, explicit negatives | Expert event labels and task/PPE context |
| Concrete defects | Actual pre/post-pour and inspection videos with cracks, honeycombing, spalling and sound surfaces | Close-up reference images, calibrated dimensions and engineer defect labels |
| Rebar/formwork | Continuous close inspections before pouring, multiple angles, bars/cover/splices/forms with real errors and correct cases | Survey, bar schedule, measured spacing/cover and signed inspection labels |
| Waterproofing/envelope | Real installation and flood/leak test videos, defective and correct joints/terminations | Specified test results, material system identity and inspector labels |
| MEP/firestop | Real before-cover walkthroughs, product identities, installation variations and failures | Approved design/listed-system match, measurements and qualified review |
| Access/egress/finishes | Complete routes and room walkthroughs with scaled views | Calibrated slopes, clearances, forces and approved design context |
| Structural/soil/material acceptance | Video can document taking a sample or witnessing a test | Traceable laboratory/field tests and engineering assessment; no RGB-only substitute |
| Commissioning/occupancy | Continuous actual witnessed functional-test footage linked to test cases | Calibrated logs, signed reports and authority-issued documents |

No public dataset discovered by this code-source research establishes every Saudi inspection family, all construction stages and all numerical code acceptance rules. Collecting that complete evidence requires project access, calibrated instruments, laboratories, drawings, qualified review and authority records. Those inputs are currently missing; the checklist exposes the gaps rather than filling them with fabricated observations.

## Validation design for adding each capability

Before enabling a new capability, freeze a versioned real-data manifest and labels. Split by **project/site and original video**, not neighboring frames. Reserve Saudi site footage that was not used to train or choose thresholds, and retain difficult negative clips. Unknown training overlap means a public-data result is a compatibility diagnostic.

Measure event-level precision/recall, missed-event count and false alerts per hour, with class support and uncertainty intervals; add per-frame detection metrics where appropriate. Record motion blur, distance, darkness, occlusion and minimum visible feature size. For dimensions, evaluate calibrated measurement error separately from classification. For code-linked acceptance, evaluate against the actual approved project criteria and qualified review rather than treating a class label as ground truth for law.

Benchmark actual decode + preprocessing + inference + postprocessing + event publication on the laptop, alongside frame drops and source-to-result latency. Test sustained continuous footage rather than inference on one still. A 15 FPS factory benchmark does not transfer to a construction model or to multiple simultaneous streams. Accuracy and speed need separate evidence for each enabled mode and resolution.

This study adds no generated images, no fabricated inspection findings, no model weights and no invented field-test results. Actual installed models, footage provenance, annotations, performance and known failures belong in the implementation's accompanying model/data verification report.

## Source register and reproduction

All source URLs and access dates are included in the JSON catalog, including whether a source is a full text, contents page, product description or indexed-only statement. Successful PDF hashes/page counts are recorded in `output/research/official-source-downloads.json`; the full SBC 201 hash is recorded above and in the catalog. Failed downloads are recorded rather than treated as complete sources.

```powershell
.venv\Scripts\python.exe scripts\build_construction_catalog.py
```

The command rebuilds the JSON and detailed Markdown checklist. It performs no network requests, model inference or compliance decisions. Any later implementation mapping must be added explicitly to the generator so rebuilding cannot silently erase provenance or inflate coverage.

Additional primary scope references: [SBC 303 contents](https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20303/SBC303_CR_241224-FA.pdf), [SBC 304 contents](https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20304/SBC304_CR_241224-FA.pdf), [SBC 401 contents](https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20401/SBC401_CR_241224-FA.pdf), [SBC 501 description](https://sbc.gov.sa/en/BC/Pages/BuildingCode/BCDetails.aspx?codeId=501&year=2024), [SBC 701 contents](https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20701/SBC701-CR_241224-FA.pdf), [SBC 801 contents](https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20801/SBC801_CR_241224-FA.pdf), and [official telecom-guide announcement](https://www.spa.gov.sa/N2164456).
