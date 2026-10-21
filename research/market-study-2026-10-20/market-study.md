# Global market study for the OpenCV hackathon

20 October 2026 | Market needs, competition, buyers and project selection

## Decision after research

**Build one narrowly scoped factory manufacturing fault inspection and review prototype, subject to a first-day technical test.** The strongest alternative for a team without equipment or partners is freight document submission readiness. Road inspection review is a third option; medical image privacy QA is a more demanding fourth reserve. These choices follow the market screen below, rather than the earlier brainstorm.

This is a desk-based opportunity study. It establishes real workflows, existing suppliers, relevant organizations and possible evaluation data. It does not establish purchase intent, an uncontested competitive gap, or product-market fit. No customers were interviewed or contacted. All four proposed gaps remain hypotheses.

| Priority | Research-derived project | Why retain it | Main unresolved issue |
|---|---|---|---|
| 1 | Factory manufacturing fault inspection and review | Required industrial theme; operator evidence of review work; real licensed PCB images | Can a narrow tool improve review beyond existing AOI software? |
| 2 | Freight document submission readiness | Documents support billing; demonstrable with public capture data and fictional forms | Capture and guidance already exist; packet-level advantage is unproven |
| 3 | Road inspection evidence review | Identifiable municipal and consultant buyers; multinational public benchmark | Mature competition; image dataset does not validate field workflow |
| 4 reserve | Medical image residual identifier QA | Specific research-data workflow; paired public evaluation data | DICOM expertise, false negatives and institutional validation |

The factory concept means **visible manufacturing faults in products**, initially one PCB family. It does not predict internal machine failures. If the intended requirement is specifically machinery faults, a motion-anomaly simulation is possible, but it has weaker validation without actual machine footage.

**Commercial conclusion:** none yet deserves an investment claim based on this study alone. **Hackathon conclusion:** the factory and freight projects offer the best routes to measurable, reproducible demonstrations with the resources described. A polished dashboard is insufficient; the work must show useful decisions, uncertainty handling and comparisons against simple baselines.

Research notes and source ledgers accompany this report in the evidence notebook. Scores later in the report are explicit analyst judgments, not measured market statistics or predictions of winning.

<!--pagebreak-->

## Scope and evidence method

The user specified global scope and no available resources or partner access. Planning assumes an existing computer and internet connection. Team size, coding skills, GPU access, AWS account and available spend are unknown. No proposal depends on buying equipment, obtaining a factory visit, or collecting private customer data before the deadline.

Three research agents independently examined industrial operations; infrastructure, energy and agriculture; and frontline/document workflows. The lead researcher checked the competition and an additional food-service market, cross-checked important sources and corrected a stale pricing reference. The scan covered 14 distinct problem areas. It is a targeted screen of computer-vision opportunities, not an exhaustive survey of the global economy.

Evidence was considered in this order: an operator's documented workflow; a public agency or original research source; a vendor's named customer account; a vendor's feature or pricing page. Supplier adoption and savings statements are labelled as supplier claims. Historic evidence is retained only with its date and context. No general industry loss estimate is converted into software revenue.

**Global scope does not mean worldwide validation.** Examples from the US, Europe and multinational operators establish specific workflows. They do not prove that every country's buyers have the same budget, procurement process or unmet need. Named organizations are interview targets or existing category adopters, never implied customers of this team.

### Competition verification

The supplied brief defines the required OpenCV 5 analysis, meaningful AWS execution, technical report, code, evaluation, diagram, working demonstration and five-minute video. Live [Devpost overview](https://opencv26.devpost.com/) and [rules](https://opencv26.devpost.com/rules) were checked. COOL and agentic methods are optional for the main awards. Use the supplied judging weights when preparing the submission; do not add an agent solely for its label.

There is a timing discrepancy: the overview narrative says 26 October at 11:59 p.m. Pacific, while the platform header displays 11:45 p.m. PDT. Plan to submit before the earlier displayed time. The [project gallery](https://opencv26.devpost.com/project-gallery) is unpublished, so this study cannot assess current entrants or claim an uncontested competition niche.

The study provides architectures and evaluation plans; no application was built, benchmarked or deployed during research. Online descriptions verify availability, not successful download or runtime compatibility. First-day checks are therefore part of the recommendation.

<!--pagebreak-->

## Market screen before idea selection

| Need examined | Evidence and existing alternatives | Disposition under current constraints |
|---|---|---|
| Manufacturing defects and review | Operators describe AOI and technician disposition; Cognex, Instrumental and Tulip overlap | Retain narrowly; real public PCB data |
| Machine motion faults | Industrial cycle abnormalities and SICK jam sensors | Conditional simulation; no real footage |
| Leaks and internal machine faults | Recognized maintenance need; acoustic, thermal and condition sensors | Reject ordinary RGB-only claim |
| Shipping carton damage | DHL describes inspection and rehandling | Defer; representative licensed grading data not established |
| Pavement inspection | Municipal workflows; vialytics, RoadBotics, Pavemetrics | Retain review workflow conditionally |
| Sewer inspection | Utilities already procure CCTV analytics; SewerAI, Pallon, VAPAR | Reject now; data/access and standards barriers |
| Solar maintenance | DOE O&M guidance; Raptor Maps, Sitemark, Aerodyne | Reserve below shortlist; tiny crop data lack site context |
| Crop diagnosis | Established free Plantix and paid diagnostic APIs | Reject generic app; no field partner |
| Freight document intake | Carrier document/billing workflows; capture SDK purchases | Retain packet readiness, not generic scanning |
| Medical image privacy QA | TCIA curation and DICOM pixel-cleaning workflow | Fourth reserve; constrained benchmark demo |
| Assistive capture | Authentic VizWiz data; free Seeing AI and Be My Eyes | Reject generic tool without user co-design |
| Warehouse safety | Recognized hazards; Intenseye, Protex, Voxel | Reject without site geometry/footage/users |
| Returned-item grading | DHL reverse-logistics operations; established returns platforms | Defer; no disposition ground truth |
| Commercial kitchen waste | Ingka adoption; Winnow, Leanpath, Orbisk | Reject now; mature solutions and no kitchen data |

Detailed sources and limitations for the first thirteen areas are in the three evidence memos. The additional food-service check found customer-side adoption: [Ingka reported](https://www.ingka.com/newsroom/one-million-bags-of-leftovers-and-how-were-putting-them-to-good-use/) a 60% reduction in production food waste per cover since FY17 and Waste Watcher in 402 stores by FY25. This is an operational program outcome, not the causal effect of a new classifier. [Leanpath](https://www.leanpath.com/products/food-waste-tracking/) already offers mobile camera capture; [Orbisk](https://orbisk.com/product/solutions/) offers ingredient tracking and action guidance. Camera-only capture is therefore not demonstrated whitespace.

The recurring lesson is that a serious problem can have poor fit for this team. Accessible images alone also do not establish an economically valuable product. Shortlisting requires both an operational workflow and a credible evaluation route.

<!--pagebreak-->

## Factory manufacturing fault inspection and review

**Problem evidence.** [MacroFab's July 2024 account](https://nws.macrofab.com/blog/aoi-inspection-process) describes technicians deciding whether AOI flags are real defects or false alarms. It also describes sophisticated 3D and multiview equipment. [Flex's manufacturing account](https://flex.com/downloads/driving-innovation-efficiency-and-quality-through-advanced-manufacturing-solutions) explains downstream rework from misplaced PCBA components. These support a quality workflow, not the proposition that a webcam replaces industrial inspection.

**Buyer and proposed job.** Target an independent assembler with one recurring review task. The quality manager owns the problem; an AOI technician or quality engineer uses the system; plant operations or manufacturing engineering pays. The proposed tool checks selected visible faults, presents a reference comparison, and records a human decision with evidence.

**Competition and possible gap.** Cognex supplies industrial vision; Instrumental already links visual defects, unit records and failure analysis; Tulip supports operator workflows and vision. A narrowly configured single-station task may be easier to adopt, but this has not been proven. Defect grouping, a heatmap or an agent are not sufficient differentiation. The commercial test is lower review effort at an acceptable defect-escape rate.

**MVP.** Upload images from one PCB subset. OpenCV 5 checks image quality, aligns supported views, compares approved references or runs a compatible small model, and extracts candidate regions. A workflow retries failed alignment, examines a crop and either creates a review case or abstains. A reviewer accepts or rejects the finding. AWS runs the inspection and persists cases and evidence. No automatic machine control is required.

**Data and demo.** [Amazon VisA](https://github.com/amazon-science/spot-diff) provides real industrial object images, including PCB subsets, with image/pixel labels under CC BY 4.0; the [AWS registry](https://registry.opendata.aws/visa/) confirms the license. The demo is an image replay, not a live production line. Normal board variation may defeat reference differencing; no detector has yet been tested.

**Success and stop rule.** Report precision, recall across all defective inputs, recall on accepted inputs, false alerts per 100 normal samples, coverage, registration failures, localization and latency. List abstained defective and normal counts separately; abstention is not an automated correct detection. Proposed day-one gate: at least 80% overall recall, at most 10 false alerts per 100 normal images and at least 90% coverage. Use validation data to select thresholds and one bounded model fallback, then one untouched final test. These are screening targets, not industrial acceptance standards. If the detector fails, narrow the claim or choose freight. Hidden joints and electrical function are outside scope.

**Customer validation needed.** Inspect ten actual defect-review cases, measure time spent and obtain authorized images. Reject the commercial premise if existing AOI software already handles the task or saved review time cannot cover integration and support.

<!--pagebreak-->

## Freight document submission readiness

**Problem evidence.** [Lynden's own tools](https://www.lynden.com/online-tools/) link invoicing to bills of lading and proof-of-delivery receipts. [ArcBest's Scanbot case study](https://scanbot.io/success-stories/arcbest/) includes a named IT director's report that improved images make information available earlier for billing. That is vendor-hosted testimony about an incumbent deployment, not an expected saving from this proposal. Lynden's supplier invoice rules are additional context about legibility and completeness; they do not measure driver document rejection rates.

**Buyer and proposed job.** Target a smaller freight operator or logistics-software vendor with recurring document exceptions. Billing/operations owns the economic problem, a mobile-product owner approves integration, and drivers or clerks use it. The job is to submit one complete, readable packet rather than discover missing evidence later.

**Competition and possible gap.** Scanbot already provides quality analysis, spatial heatmaps, cropping and guided capture. Regula checks blur, glare, boundaries and occlusion. Scandit supports driver proof-of-delivery workflows. The only plausible narrower hypothesis is template-aware packet completeness: a sharp image can still omit a required field or page. Existing SDKs may implement that too; no absence has been demonstrated.

**MVP.** Support two original fictional document templates and a configurable required-page list. OpenCV 5 detects and rectifies pages, locates reference regions, measures local legibility and detects duplicate images. The workflow requests only the missing page or unusable region, then rechecks the packet. AWS processes uploads and retains packet state and a decision trace. An upload/replay interface works without camera access. Do not claim signature authenticity or freight-law compliance.

**Data and demo.** [SmartDoc15 Challenge 1](https://github.com/jchazalon/smartdoc15-ch1-dataset) has 150 document videos, roughly 24,000 frames, corner ground truth and CC BY 4.0 terms. It supports geometry evaluation, not freight semantics. Create fictional forms for missing-field and duplicate-page tests, with clearly labelled generated distortions. The separate SmartDoc-QA dataset has different restrictions and must not be substituted silently.

**Success and stop rule.** Split by original document/video, never adjacent frames. Compare whole-image sharpness plus automatic crop against region/packet checks. Measure false acceptance of incomplete packets, unnecessary prompts, successful completion, processing latency and extraction error where OCR is used. Test held-out form layouts. Stop if the extra checks merely repeat scanner features, increase driver effort, or rely on a proprietary workflow unavailable to the team.

**Commercial validation needed.** Obtain an anonymized rejection-reason tally and observe five end-to-end submissions. The proposed product has value only if the avoided exceptions exceed integration, support and user interruption costs. Current public evidence does not supply that rework rate.

<!--pagebreak-->

## Road inspection evidence review

**Problem evidence.** [FHWA guidance](https://www.fhwa.dot.gov/pavement/management/qm/data_qm_guide.pdf) documents established condition-data collection and quality-management workflows; it is historical US guidance, not a current global adoption estimate. [vialytics](https://www.vialytics.com/) currently advertises use by over 1,000 communities in seven countries and names municipal users. This is supplier evidence that buyers exist, not independent proof of its results or unmet demand.

**Buyer and proposed job.** A pavement engineering consultancy is a plausible initial buyer because it may serve multiple municipal clients. Its analyst uses the tool; a project manager pays from a survey project budget. The job is to turn an existing image batch into a reviewable evidence packet, not to promise an autonomous city maintenance platform.

**Competition and possible gap.** vialytics already captures smartphone images with GPS/time, assesses damage and supports work orders. RoadBotics offers condition and export workflows; Pavemetrics uses specialist 2D/3D survey systems. The proposed gap is a configurable quality/review step for a consultant's existing process. Whether this saves effort beyond current tools is unknown. A work-order dashboard by itself is not innovation.

**MVP.** Ingest a small supported pavement-image batch. OpenCV 5 rejects blur/exposure failures, runs a compatible detector for supported distress classes, extracts evidence regions and checks duplicates where inputs permit. An uncertain result triggers another analysis or a human review request. AWS executes batch processing and stores approved findings and their source images. Export a simple evidence report. Do not invent GPS, depth, severity, duplicate-event identity or repair outcomes absent ground truth.

**Data and demo.** [RDD2022's authors](https://arxiv.org/abs/2209.08538) describe 47,420 images from six countries and four main damage classes. The [originating repository](https://github.com/sekilab/RoadDamageDetector) specifies CC BY-SA 4.0 for images separately from MIT code. Verify the precise archive and its annotations on day one, retain attribution and apply applicable share-alike terms. Availability and license metadata were checked online; a download was not tested.

**Success and stop rule.** Hold out a country or route rather than randomly mixing near-duplicates. Report precision/recall by country, class and image quality; false alerts from shadows/markings; abstention; and actual batch time. A proposed human-study gate is at least 30% less review time at matched missed-defect performance. Until qualified reviewers participate, that benefit remains untested. Classical morphology alone is unlikely to generalize across varied pavements; model licensing and ONNX compatibility must pass the first-day check.

**Commercial validation needed.** Observe two consultant reporting workflows and identify an export/integration need that their existing platform leaves unresolved. Stop if current tools cover the job or route-held-out performance degrades severely. This ranks below factory and freight because field validation and procurement remain harder.

<!--pagebreak-->

## Medical image residual identifier QA

**Status: fourth reserve.** This is a technically interesting research-data tool, with lower commercial readiness for a team lacking an institutional partner.

**Problem evidence.** TCIA documents de-identification effort and evaluation data. The current [DICOM Clean Pixel Data option](https://dicom.nema.org/medical/dicom/current/output/chtml/part15/sect_E.3.html) addresses identifying text stored in image pixels, acknowledges possible human review and distinguishes changing pixel values from hiding them with an overlay. This establishes a specific workflow; it does not establish demand for another vendor.

**Buyer and proposed job.** Imaging-research repositories, clinical-trial imaging cores and data-curation teams are plausible buyers. A curator or privacy reviewer uses it; research operations funds it. TCIA is a relevant operator and benchmark producer, not an interested customer. The job is an independent review after another tool has removed identifiers.

**Competition and possible gap.** Sante DICOM Editor already provides anonymization and pixel annotation removal. PixelMed and pydicom/deid are available alternatives; [Microsoft Presidio](https://microsoft.github.io/presidio/samples/) also provides DICOM redaction examples. A generic anonymizer is not a useful differentiation claim. A narrower hypothesis is residual-identifier QA with documented abstention and reviewer decisions. Its incremental value must be demonstrated against the existing toolchain.

**MVP.** Decode a deliberately limited supported DICOM subset using pydicom. OpenCV 5 performs pixel normalization, candidate-region processing and before/after differencing; a compatible OCR component supplies text candidates. The workflow flags uncertain residual text, asks for review and rechecks an approved pixel edit. AWS runs a batch job and stores an audit record using public synthetic-identifier examples. Separate metadata checks from pixel checks. Do not call the output certified anonymous or compliant.

**Data and demo.** The [TCIA Pseudo-PHI collection](https://www.cancerimagingarchive.net/collection/pseudo-phi-dicom-data/) provides 1,693 evaluation images and 1,693 curated counterparts from 21 subjects, with synthetic identifiers, CC BY 4.0 terms and TCIA usage conditions. Access uses TCIA Data Retriever. Use a small supported subset and preserve the required DOI citation. Confirm manifest access, current usage conditions and transfer syntax support before selection. Public-data feasibility does not establish permission or controls for processing real patient data.

**Success and stop rule.** Measure identifier-region recall, files with any missed identifier, false positives on legitimate annotations, anatomy affected by edits and reviewer time. Never combine metadata success with pixel success into one misleading score. A cosmetic overlay is a failed cleaning test. Low false-positive text detection is not evidence of safe identifier removal; an unreadable identifier can still leak information. Stop if the parser cannot handle the declared subset, identifier recall is inadequate or the review step adds no useful coverage.

**Commercial validation needed.** An institutional curator must identify residual errors that current tools miss and approve a bounded retrospective comparison. Procurement, validation responsibility and deployment requirements are substantial. This is a reserve research prototype, not a rapid healthcare business launch.

<!--pagebreak-->

## Competitors and observed commercial terms

All prices are observations of public pages on 20 October 2026, not quotations for an equivalent configuration. Feature availability and total deployment costs differ. Missing public pricing is not evidence that a competitor is expensive.

| Market | Supplier or substitute | Verified overlap | Price evidence and limitation |
|---|---|---|---|
| Factory | [Cognex](https://support.cognex.com/en/products/in-sight-d900) | Industrial deep-learning vision | Configured price not verified |
| Factory | [Instrumental](https://instrumental.com/how-it-works) | Visual defects, unit records, failure analysis | Numeric price not verified |
| Factory | [Tulip](https://tulip.co/plans/) | Apps, review workflows, vision add-on | Essentials $100/interface/month, annual billing, 10-interface minimum; add-ons extra |
| Factory | [LandingAI](https://landing.ai/landinglens/) | Model-development and deployment alternative | Applicable current LandingLens price unverified; old price URL redirects to different APIs |
| Freight | [Scanbot](https://scanbot.io/contact-sales/) | Capture, quality, user guidance | Generally starts $2,500/year for simple cases; custom scope; unlimited scans |
| Freight | [Regula](https://docs.regulaforensics.com/develop/doc-reader-sdk/overview/image-quality/) | Focus, glare, bounds, occlusion | Numeric price not verified; mainly identity documents |
| Freight | [Scandit](https://www.scandit.com/industries/post-and-parcel/driver-operations/) | Driver proof of delivery and scanning | Sales-led; numeric price not verified |
| Roads | [vialytics](https://www.vialytics.com/) | Capture, assessment, priorities, work orders | Sales-led; numeric price not verified |
| Roads | [RoadBotics](https://www.roadbotics.com/wp-content/uploads/2021/03/subscription-options.pdf) | Condition data and exports | 2021 brochure is historical; current price unknown |
| Roads | [Pavemetrics](https://www.pavemetrics.com/capabilities/lcms-2/cracking/) | Specialist 2D/3D pavement measurement | Configured price not verified |
| Medical QA | [Santesoft](https://www.santesoft.com/win/sante-dicom-editor/sante-dicom-editor.html) | DICOM editing and anonymization | $480/user or $2,500/site perpetual; not equivalent QA scope |
| Medical QA | [PixelMed](https://www.dclunie.com/pixelmed/software/javadoc/com/pixelmed/display/DicomImageBlackout.html) and [pydicom/deid](https://pydicom.github.io/deid/getting-started/dicom-pixels/) | Pixel blackout and cleaning workflows | Software alternatives; deployment/review still costs time |

**Implication:** there is no evidence for a broad feature monopoly. A sustainable advantage would need to arise from an unusually easy integration, a narrowly useful workflow, access to representative data, measured results, or distribution. This team currently has none of those advantages validated.

<!--pagebreak-->

## Potential customers and buying access

These are verified examples of relevant organizations, not leads, partners or expressions of interest. Several already use competing products. They help identify real roles and workflows, but are not automatically realistic first customers.

| Organization and geography | Evidence of relevance | Role to learn from and principal barrier |
|---|---|---|
| CircuitHub, US | [High-mix assembly lines](https://docs.circuithub.com/assembly/assembly-lines) | Quality/manufacturing engineer; existing automation |
| MacroFab, US | [AOI and technician review](https://nws.macrofab.com/blog/aoi-inspection-process) | AOI lead; sophisticated existing equipment |
| Flex, multinational | [PCBA manufacturing workflow](https://flex.com/downloads/driving-innovation-efficiency-and-quality-through-advanced-manufacturing-solutions) | Site quality/NPI lead; IP and enterprise qualification |
| Jabil, multinational | [Electronics and test capabilities](https://www.jabil.com/capability/electronics.html) | Quality engineering; toolchain and validation |
| Lynden, US/international | [Freight and document tools](https://www.lynden.com/online-tools/) | Billing/mobile-product owner; incumbent scanning |
| ArcBest, US | [Document retrieval workflow](https://support.arcb.com/en/articles/1509576-document-retrieval) | Billing/product manager; existing Scanbot deployment |
| Edison, New Jersey, US | Named [vialytics user](https://www.vialytics.com/) | Public-works analyst; already served |
| Metuchen, New Jersey, US | Named [vialytics user](https://www.vialytics.com/) | Public-works director; incumbent and procurement |
| Oberasbach, Germany | Named [vialytics reference](https://www.vialytics.com/) | Municipal engineering; incumbent and local requirements |
| Colliers Engineering and Design, US | Listed [vialytics partner](https://www.vialytics.com/) | Pavement consultant; commercial partnership conflicts |
| TCIA, US research archive | [Image curation workflow](https://www.cancerimagingarchive.net/about-the-cancer-imaging-archive-tcia/) | Data curator; institutional validation and deployment |

**Initial customer hypotheses:** independent high-mix assemblers with recurring manual review; smaller freight software vendors with packet exceptions; pavement consultants with a specific reporting bottleneck; research imaging cores with documented residual-identifier misses. Their reachability, prevalence and willingness to pay remain unknown.

For discovery, ask each role for the last five or ten relevant cases, the tool they used, time spent, exception frequency and escalation route. Ask who may authorize sample data and who owns the budget. A friendly reaction to a demo is insufficient. The evidence required is an observed task and an agreement on a measurable pilot outcome.

The study did not authorize outreach; no messages were sent. Selecting interview targets does not imply permission to contact them.

<!--pagebreak-->

## Data rights and evaluation readiness

| Candidate | Public evaluation route | Important boundary |
|---|---|---|
| Factory | VisA PCB images, CC BY 4.0 | Visible product defects; no machine-failure or throughput labels |
| Freight | SmartDoc15 Challenge 1, CC BY 4.0; original fictional forms | Geometry is real; packet semantics remain generated |
| Roads | RDD2022; authors specify CC BY-SA 4.0 images, MIT code | Check exact archive; no assumed GPS, severity or repair outcomes |
| Medical QA | TCIA Pseudo-PHI, CC BY 4.0 plus usage terms | Synthetic identifiers; limited supported DICOM subset |
| Solar reserve | Raptor Maps infrared crops; repository MIT license | 24 by 40 pixel crops; no radiometry or site outcome proof |
| Machine motion reserve | Self-authored simulation | No industrial validation; IPAD permission unresolved |

Dataset sources are linked in the project sections and the evidence notebook. [Raptor's repository](https://github.com/RaptorMaps/InfraredSolarModules) is a useful solar benchmark, but crop classification is insufficient for field maintenance triage. [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) explicitly restricts commercial use; a public download does not mean unrestricted competition reuse. Do not substitute a mirror's code license for image rights.

Before building, download only the required subset, preserve its original terms and attribution, record hashes and create fixed train/validation/test manifests. Licenses for model weights are separate from dataset licenses. No pretrained model, ONNX conversion or AWS container compatibility was tested in this study.

### Evaluation rules shared by all candidates

Use an untouched final test after selecting models and thresholds on validation data. Do not repeatedly choose fallbacks against the final test. Split by original object, document, video, route, institution or source group as appropriate. Avoid random adjacent-frame splits. Report sample sizes, failure categories and uncertainty; a few handpicked successes are not an evaluation.

Compare against a simple baseline and the incumbent workflow when available. Use paired task results when claiming time saved. Record abstentions and accepted coverage so that rejecting difficult inputs cannot artificially inflate accuracy. Keep synthetic stress tests separate from real held-out data. Report CPU, memory, batch size and p50/p95 latency measured in the actual environment.

Agentic evidence should show an input changing a subsequent tool call: poor alignment triggers registration retry; a missing page triggers a targeted request; uncertainty triggers human review. Compare the adaptive workflow with a fixed pass to establish whether it helps. A chatbot explaining a fixed result is insufficient under the supplied brief.

<!--pagebreak-->

## Economics and market size discipline

No defensible serviceable market size is available from this scan. A future bottom-up estimate should count reachable organizations with the specific workflow, supported data, an identified payer and a plausible purchase path. Multiply that qualified count by a tested price. Do not multiply global defects, infrastructure losses or vision-impairment prevalence by an invented capture percentage.

The scenarios below expose sensitivity. **Every operational volume, wage, time saving and proposed price is an assumption, not a finding.** Time redeployed is not automatically cash saved. Cloud, integration, labeling, support, sales and errors reduce gross benefit.

| Workflow | Illustrative monthly assumptions | Gross labor value | Downside case |
|---|---|---|---|
| Factory review | 300 flags/day x 22 days x 15 seconds saved; $30/hour | 27.5 hours, $825 | 100 flags/day and 5 seconds saved gives about $92 |
| Freight packets | 10,000 packets x 5% actionable rework x 3 minutes; $20/hour | 25 hours, $500 | 1% rework gives $100 before any new prompting cost |
| Road review | 20 jobs x 4 hours saved; $30/hour | 80 hours, $2,400 | 30 minutes/job gives $300 |
| Medical QA | 1,000 cases x 1 minute saved; $30/hour | 16.7 hours, $500 | 10 seconds/case gives about $83 |

Hypothetical prices to test, not recommendations: factory $200/month; freight $100-150/month; road $30/job; medical QA $100/month. At these prices each downside case fails or becomes difficult after support and integration. Public competitor prices are context, not evidence that buyers accept these figures.

### Prototype cloud allowance

[AWS's T3 example table](https://aws.amazon.com/ec2/instance-types/t3/) displays t3.medium at $0.0418/hour. For 168 hours this is about $7.02 compute-only; [public IPv4](https://aws.amazon.com/vpc/pricing/) adds $0.84 at $0.005/hour. Region, OS, storage, requests, EBS, logs, transfer, CPU credits, optional inference and taxes can change the bill. A $20-40 prototype allowance is an assumption, not a cap or a production cost estimate. Larger models may require a different instance. Confirm pricing and memory/CPU requirements before deployment; do not assume grants or free-tier eligibility.

The research itself created no AWS resources and incurred no paid API purchases.

<!--pagebreak-->

## Comparative assessment and selection

Scores are analyst judgments from 1 weak to 5 strong. The weighted total is the sum of score divided by five times the criterion weight. Weights reflect this user's lack of access and the deadline; they are not the organizer's judging rubric. No score measures validated purchase intent.

| Candidate | Pain 20 | Buyer 10 | Gap 20 | Data 20 | Build 20 | Fit 10 | Total |
|---|---|---|---|---|---|---|---|
| Factory fault review | 4 | 4 | 2 | 5 | 4 | 5 | 78 |
| Freight readiness | 4 | 4 | 2 | 4 | 5 | 4 | 76 |
| Road evidence review | 4 | 4 | 1 | 4 | 3 | 5 | 66 |
| Medical residual QA | 4 | 3 | 2 | 4 | 2 | 4 | 62 |
| Solar second review | 4 | 4 | 1 | 3 | 3 | 4 | 60 |

Pain scores reflect documented workflows, not prevalence. Buyer scores reflect identifiable roles, not access. Gap scores are low because feature overlap is substantial and no interviews establish an unmet niche. Data scores account for rights and task relevance; freight's semantic tests are fictional, while medical data use synthetic identifiers. Build scores penalize untested models, unfamiliar file formats and missing context. Fit reflects the likely depth of OpenCV work and a demonstrable decision loop.

The two-point difference between factory and freight is not decisive. Reducing factory build feasibility from four to three yields 74, below freight. A single day of technical evidence can therefore reverse the selection. The factory requirement also makes an industrial option necessary in the final set; it should not be used to pretend the commercial evidence is stronger than it is.

**Recommendation 1:** start with factory product-fault inspection and review for one PCB family. Its real licensed images and documented operator task justify a first-day trial. If machinery failure is required literally, show a separately labelled motion simulation and narrow every claim to observable motion; do not present product defects as bearing-failure prediction.

**Recommendation 2:** use freight packet readiness as the implementation fallback or choose it first if the team prefers deterministic geometry and workflow checks. Its weakness is competitive differentiation, not demo access.

**Recommendation 3:** retain road evidence review only if a small compatible licensed detector and exact data archive work promptly. Do not attempt a citywide digital twin.

**Fourth reserve:** select medical residual QA only with DICOM competence and a restricted evaluation scope. Otherwise keep it in the research portfolio rather than consume the build window.

<!--pagebreak-->

## Build and validation plan

Build one candidate, not the whole shortlist. This is a proposed schedule, not completed work.

| Date | Work and decision evidence |
|---|---|
| 20 October | Download a small licensed subset, preserve terms, pin OpenCV 5, test one baseline and declared input format. Stop/pivot if data or runtime fails. |
| 21 October | Implement image-quality checks, the narrow core workload, abstention and held-out evaluation. Record failures before adding UI. |
| 22 October | Add the review interface and one adaptive tool sequence. Log the visual measurement, decision, next call and resulting state. |
| 23 October | Run the OpenCV service on AWS. Save evidence and decisions; test upload, permission and retention behavior. |
| 24 October | Compare with the fixed baseline; measure latency, false acceptance, coverage and task outcomes. Freeze scope. |
| 25 October | Finish report, architecture, pinned setup, limitations and video. Rehearse a five-minute demonstration. |
| 26 October | Reproduce from clean instructions, test the judge endpoint and submit before the earlier displayed cutoff. |

### Proposed minimum architecture

Browser upload or public-data replay -> private S3 source objects -> EC2 container running OpenCV 5 and task checks -> workflow decision -> crop/retry/request/human review -> persisted evidence and reviewer outcome -> browser report.

This provides meaningful AWS processing rather than a cosmetic cloud database. [S3 presigned URLs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html) support time-limited uploads/downloads. Keep sample data and results traceable to input/model/configuration versions. Demonstrate failed upload and failed analysis recovery. Limit accepted file types, sizes and processing time as part of the product's reliability.

Use an explicit, bounded orchestration policy for the MVP. If pursuing the Agentic Vision award, include the perception-to-next-action trace and qualify the actual implementation against the organizers' rules; a state machine is not automatic proof of award eligibility. Use COOL only if the core workload actually runs on the documented Arm path and there is time for a reproducible comparison.

### Commercial validation after the prototype

Before asserting a business case, interview five relevant budget owners, observe two actual workflows and obtain one authorized retrospective evaluation. Ask what existing tool fails, how often, what the exception costs, and who would approve a paid pilot. Require an agreed success measure and a route to usable data. Kill or revise the premise if current tools solve it, data cannot be shared, or measured benefit cannot cover switching and support costs.

The next justified commitment is a bounded technical test and customer discovery. The desk evidence supports investigating these projects; it does not justify promising sales, savings, accuracy or a competition win.
