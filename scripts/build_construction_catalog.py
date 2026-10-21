"""Rebuild the research catalog and its Markdown checklist; never evaluates compliance."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ACCESSED = "2026-10-20"
INDEX = "https://sbc.gov.sa/en/BC/Pages/BuildingCode/BC.aspx?year=2024"
GENERAL = "https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20201/SBC201_AR2024.pdf"

SOURCES = [
    {"id": "sbc-2024-index", "title": "Official Saudi Building Code 2024 suite", "url": INDEX, "verification": "Official index opened; code-family identity verified."},
    {"id": "sbc-residential-scope", "title": "Official SBC 1101-1102 residential scope description", "url": "https://sbc.gov.sa/ar/BC/Pages/BuildingCode/BCDetails.aspx?codeId=1101&year=2024", "verification": "Official description verified: qualifying one- or two-family dwellings, no more than three stories; integrated discipline coverage. This is a scope description, not a full applicability-clause review."},
    {"id": "sbc201-2024", "title": "SBC 201-AR-2024", "url": GENERAL, "verification": "Official full 593-page PDF downloaded and text extracted; sections 110, 111 and chapter 17 inspected.", "sha256": "237070fe2784a3e7c4663740718cfec841fa6675eb558d8a9fd2c84dc105a8fc"},
    {"id": "sbc303-2024-toc", "title": "SBC 303-CR-2024 official contents", "url": "https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20303/SBC303_CR_241224-FA.pdf", "verification": "4-page contents downloaded; topic/section headings only, no numerical acceptance criteria verified."},
    {"id": "sbc304-2024-toc", "title": "SBC 304-CR-2024 official contents", "url": "https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20304/SBC304_CR_241224-FA.pdf", "verification": "6-page contents downloaded; chapter 26 topic headings only."},
    {"id": "sbc401-2024-toc", "title": "SBC 401-CR-2024 official contents", "url": "https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20401/SBC401_CR_241224-FA.pdf", "verification": "30-page contents downloaded; chapter 61 inspection/testing/reporting headings verified."},
    {"id": "sbc501-2024", "title": "SBC 501 official 2024 code description", "url": "https://sbc.gov.sa/en/BC/Pages/BuildingCode/BCDetails.aspx?codeId=501&year=2024", "verification": "Official description opened; full clause text not obtained."},
    {"id": "sbc701-2024-toc", "title": "SBC 701-CR-2024 official contents", "url": "https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20701/SBC701-CR_241224-FA.pdf", "verification": "6-page contents downloaded; sections 112, 312 and system headings verified."},
    {"id": "sbc801-2024-toc", "title": "SBC 801-CR-2024 official contents", "url": "https://sbc.gov.sa/ar/BC/Documents/tableofcontent2024/SBC%20801/SBC801_CR_241224-FA.pdf", "verification": "12-page contents downloaded; fire-system and construction-safety topic headings only."},
    {"id": "implementing-regulations", "title": "Official Gazette: implementing regulations for applying the Saudi Building Code", "url": "https://www.uqn.gov.sa/details?p=18175", "verification": "Official publication opened; articles 13-17 and 21-22 inspected. Later amendments were not exhaustively audited."},
    {"id": "inspector-guide-2019", "title": "MOMAH technical building inspector guide (2019; 2018-code context)", "url": "https://momah.gov.sa/sites/default/files/2020-03/%D8%AF%D9%84%D9%8A%D9%84%20%D8%A7%D9%84%D9%85%D9%81%D8%AA%D8%B4%20%D8%A7%D9%84%D9%81%D9%86%D9%8A%20%D9%84%D9%84%D9%85%D8%A8%D8%A7%D9%86%D9%8A.pdf", "verification": "15-page official guide opened; initial/during/final inspection and annex checklist read; historical guidance, not a substitute for 2024 code."},
    {"id": "permit-service", "title": "MOMAH issuing a building permit", "url": "https://momah.gov.sa/ar/e-services/issuance-building-license", "verification": "Current public service requirements opened."},
    {"id": "occupancy-service", "title": "MOMAH issuing an occupancy certificate", "url": "https://momah.gov.sa/ar/e-services/issuing-occupancy-certificate", "verification": "Current public service steps opened."},
    {"id": "sbc-edition-brochure", "title": "SBC GETCODE2025 service brochure", "url": "https://sbc.gov.sa/ar/ContactUS/Documents/GETCODE2025.pdf", "verification": "Search-indexed official text identifies ministerial decision 181 dated 2024-12-31 and transition date 2025-06-30; live PDF returned 404, so current legal effect is not independently confirmed."},
    {"id": "telecom-guide-announcement", "title": "SPA: official announcement of building telecom requirements", "url": "https://www.spa.gov.sa/N2164456", "verification": "Official announcement opened; telecom infrastructure planning scope verified, detailed guide not acquired."},
]
for source in SOURCES:
    source["accessed_on"] = ACCESSED
SOURCE_BY_ID = {s["id"]: s for s in SOURCES}

STAGES = [
    ("preconstruction", "Design, approvals and preconstruction"),
    ("site_safety", "Site establishment and safety throughout construction"),
    ("earthworks", "Excavation, ground treatment and earthworks"),
    ("foundations", "Foundations and below-ground work"),
    ("reinforcement", "Reinforcement and embedded items before concealment"),
    ("formwork", "Formwork, falsework and temporary support"),
    ("concreting", "Concrete production, placement and curing"),
    ("structure", "Structural frame, masonry, steel and precast"),
    ("envelope", "Envelope, roof, insulation and waterproofing"),
    ("mep_roughin", "Electrical, mechanical, plumbing, gas and telecom rough-in"),
    ("fire_life_safety", "Fire, smoke and means of escape"),
    ("accessibility", "Accessibility and external access"),
    ("finishes", "Partitions, finishes and fit-out"),
    ("testing_commissioning", "Testing, commissioning and integrated verification"),
    ("handover", "Completion, occupancy and handover"),
    ("operation_changes", "Operation, maintenance and later alterations"),
]

INSPECTIONS = []


def building_applicability(stage, slug):
    """Evidence-planning interpretation, never a legal applicability decision."""
    # These notes are grounded in the verified scope/family descriptions and
    # inspected SBC201 headings. Exact project acceptance rules are not invented.
    common = {
        "preconstruction": ("Common preconstruction review; establish the dwelling category and applicable residential/discipline route.", "Common preconstruction review; establish the actual commercial occupancy, use and applicable discipline route."),
        "site_safety": ("Common site-safety task; a small residential site is not assumed free of the stated work hazard.", "Common site-safety task; assess actual workforce, plant, neighboring uses and public interfaces."),
        "earthworks": ("Applies where this ground operation is specified; use the actual dwelling site investigation and ground conditions.", "Applies where this ground operation is specified; use the actual commercial site investigation, loads and neighboring structures."),
        "foundations": ("Applies to the specified dwelling foundation system; low height alone does not establish bearing or foundation acceptance.", "Applies to the specified commercial foundation system; design loads and ground conditions govern the evidence."),
        "reinforcement": ("Common reinforcement task where reinforced concrete is used; inspect against this dwelling's approved structural details.", "Common reinforcement task where reinforced concrete is used; inspect against the approved commercial structural details and loading basis."),
        "formwork": ("Common task where formwork/falsework is used; residential use does not establish temporary-support or release acceptance.", "Common task where formwork/falsework is used; loading, spans and sequence follow the approved project method."),
        "concreting": ("Common task where concrete is used; a dwelling's mix, exposure, placement and acceptance criteria still require project evidence.", "Common task where concrete is used; the approved mix, exposure, placement and acceptance criteria remain project-specific."),
        "structure": ("Applies to the structural system actually used in the dwelling; do not assume a particular material or simplified acceptance.", "Applies to the structural system actually used commercially; occupancy and design loads may change the required detailing and inspections."),
        "envelope": ("Common envelope task for the assembly actually installed; use the dwelling's approved system and relevant residential criteria.", "Common envelope task for the assembly actually installed; commercial facade, roof and opening systems may require different design/test evidence."),
        "mep_roughin": ("Inspect the dwelling system actually installed under its confirmed residential/discipline route.", "Inspect the commercial system actually installed; actual use, equipment, demand and hazards determine design and testing."),
        "fire_life_safety": ("Occupancy/scope dependent: confirm the applicable dwelling fire, separation and escape provisions; do not assume a blanket exemption.", "Occupancy/height/use dependent: confirm the approved commercial fire strategy and relevant SBC201/SBC801 provisions."),
        "accessibility": ("Scope dependent: distinguish a private dwelling from shared/common or public-facing areas; no blanket residential exemption is inferred.", "Occupancy/use dependent: review the applicable accessible public, customer and staff routes/facilities against the approved design."),
        "finishes": ("Inspect the dwelling's specified finish/assembly; appearance, dimensional acceptance and any fire classification are separate checks.", "Inspect the commercial specified finish/assembly; occupancy and location may alter fire, accessibility and durability criteria."),
        "testing_commissioning": ("Perform the tests required for this dwelling's actual installed systems and approved criteria.", "Perform the tests required for the actual commercial systems and approved occupancy-specific criteria; do not assume one universal test scope."),
        "handover": ("Common completion review under the applicable residential permit and authority process; an issued approval is still required where applicable.", "Common completion review under the commercial permit and authority process; engineering reports and discipline approvals follow the actual project."),
        "operation_changes": ("Use the actual residential building history, proposed use/change and applicable existing-building route.", "Use the actual commercial occupancy history, proposed use/change and applicable existing-building route."),
    }
    residential, commercial = common[stage]
    basis = "common_lifecycle"
    if stage in {"earthworks", "foundations", "reinforcement", "formwork", "concreting", "structure", "envelope", "mep_roughin", "testing_commissioning"}:
        basis = "system_or_design_dependent"
    if stage in {"fire_life_safety", "accessibility", "operation_changes"}:
        basis = "occupancy_height_or_scope_dependent"
    overrides = {
        "permit-design": ("Confirm whether this one-/two-family dwelling qualifies for SBC1101-1102; low-rise apartments or mixed use must not be assigned that route automatically.", "Confirm the actual commercial/mixed occupancy and SBC201 discipline route; the dwelling-only scope description does not cover a building merely because it is low-rise."),
        "energy-fire-access-design": ("Review the applicable residential energy route (SBC602 and/or integrated SBC1101-1102 provisions as confirmed), plus dwelling-specific fire/accessibility scope.", "Review SBC601 for applicable nonresidential portions and the actual fire/accessibility strategy; mixed-use/residential portions require their own applicability review."),
        "inspection-plan": ("Determine this dwelling's required inspection/test scope with its authority; qualifying residential scope is not permission to omit before-cover evidence.", "Determine inspection/test and any special-inspection scope for the commercial occupancy, structural system and authority directions."),
        "special-ground": ("Conditional on actual expansive/collapsible/sabkha ground; residential use neither establishes nor excludes that soil condition.", "Conditional on actual expansive/collapsible/sabkha ground and approved treatment; commercial use alone is not a trigger."),
        "piles-deep": ("Only where deep foundations are specified by the dwelling's geotechnical/structural design; no assumption that every low-rise dwelling uses piles.", "Only where the commercial foundation design specifies deep foundations; type, loads and specified tests govern."),
        "posttension": ("Only where prestressing is actually specified; obtain the dwelling's approved specialist details and calibrated records.", "Only where prestressing is specified; the commercial structural design and specialist method govern, not the commercial label alone."),
        "steel-connections": ("Only where structural steel is used; qualifying residential scope does not make connection inspection unnecessary.", "Only where structural steel is used; check project-specific SBC306/related inspection requirements and the approved connection design."),
        "masonry-work": ("Inspect actual load-bearing/non-load-bearing dwelling masonry under the confirmed design route; do not assume every wall has the same structural role.", "Inspect actual masonry role, loads, fire/location and approved SBC305/related requirements; commercial use does not define one masonry detail."),
        "precast-installation": ("Only where precast elements are used; verify the dwelling's approved bearings/connections and relevant referenced requirements.", "Only where precast is used; verify approved commercial bearings/connections, installation sequence and applicable SBC312 requirements."),
        "thermal-insulation": ("Residential energy applicability must be confirmed, including SBC602 or the qualifying integrated residential provisions; use certified properties and actual installation evidence.", "SBC601 is the nonresidential energy-family reference; confirm mixed-use portions, the approved commercial envelope criteria and actual assembly properties."),
        "energy-controls": ("Commission only the controls/equipment required by this dwelling's confirmed residential energy provisions and design.", "Commission applicable nonresidential energy controls/equipment under the approved SBC601 route; exact trigger and scope need project review."),
        "electrical-routing": ("Review the installed dwelling circuits and protection using the qualifying residential route and referenced electrical requirements.", "Review the actual commercial circuit use, demand, protection and any required essential supplies under the applicable SBC401/design route."),
        "earthing-bonding": ("Actual dwelling earthing/bonding and calibrated verification remain necessary where required; residential scope does not prove continuity.", "Actual commercial earthing/bonding design and calibrated verification govern; image appearance cannot establish electrical protection."),
        "plumbing-distribution": ("Review actual dwelling fixture demand and installed water services under the applicable residential/plumbing route.", "Review actual commercial demand and activity-specific services under SBC701/approved design; a shop, office or food use cannot share assumed criteria automatically."),
        "drainage-vents": ("Review the installed dwelling drainage/vent layout and applicable residential/plumbing criteria.", "Review actual fixture loads, use and approved drainage/vent layout; commercial activity can change design inputs and specialist provisions."),
        "potable-backflow": ("Assess actual dwelling water interfaces and required protection; test results and disinfection evidence cannot be inferred from appearance.", "Assess actual commercial water hazards/interfaces and specified protection; activity-specific equipment may change the backflow review."),
        "hvac-ductwork": ("Review the dwelling ventilation/duct system actually specified; a qualifying residential route may organize the requirements differently.", "Review actual commercial ventilation, exhaust, fire interfaces and access under SBC501/approved design; requirements depend on use."),
        "hvac-plant": ("Review the actual dwelling HVAC/condensate system and manufacturer/approved installation criteria.", "Review the actual commercial plant and operating use; plant size, location and fire/service interfaces require project-specific checks."),
        "fuel-gas": ("Conditional on a gas installation; determine the dwelling route and applicable SBC1201/referenced gas requirements with the authority.", "Conditional on installed gas and actual commercial activity; confirm applicable SBC1201 design, ventilation and specialist tests."),
        "firestop-penetrations": ("Applies where the dwelling design requires rated/separating assemblies; a visible seal is not a verified system or blanket firestop acceptance.", "Applies at required commercial fire/smoke assemblies; occupancy, shafts, services and the tested-system detail determine the inspection."),
        "fire-doors": ("Only where rated openings/doors or dampers are required by the applicable dwelling design; do not import a commercial door schedule automatically.", "Confirm occupancy/location-specific rated opening and damper requirements from the approved fire strategy; verify actual labels and operation."),
        "egress-routes": ("Check the applicable dwelling escape/exit provisions and actual configuration; do not assume low height removes escape requirements.", "Check the actual commercial occupancy/load, route arrangement and approved egress strategy; exact dimensions/travel limits are not verified here."),
        "suppression-installation": ("Determine whether and what suppression is required for this dwelling under its confirmed scope; no blanket residential exemption or requirement is asserted.", "Determine suppression scope from commercial occupancy, height, hazards and approved strategy; not every commercial project has identical systems."),
        "alarm-emergency-light": ("Determine applicable dwelling smoke/alarm, emergency-lighting and signage provisions; requirements are not assumed equal to commercial premises.", "Determine alarm/detection, emergency-lighting and exit-signage scope for the actual commercial occupancy and approved fire strategy."),
        "smoke-compartment": ("Conditional on required smoke-control/compartment systems; do not assume such a system exists in every low-rise dwelling.", "Conditional on occupancy/height/configuration and approved smoke strategy; verify applicable specialist testing rather than assuming a universal system."),
        "fire-integrated": ("Test the actual fire-system interfaces required for this dwelling; do not invent uninstalled commercial subsystems.", "Test the actual approved commercial cause-and-effect interfaces; integrated scope follows installed/required systems and authority acceptance."),
        "gypsum-rated": ("The rated/shear assembly and stated inspection exceptions must be checked for the dwelling; ordinary finish appearance is not assembly verification.", "Check actual commercial rated/shear assemblies and applicable exceptions; occupancy-specific rating and approved construction determine scope."),
        "finish-material": ("Confirm the dwelling location's applicable finish/material criteria; do not infer a fire classification from visual similarity.", "Confirm commercial occupancy/location-specific finish/material criteria from the approved design; product identity and certification are required."),
        "stairs-guards": ("Check actual dwelling stair/guard/glazing geometry under the confirmed residential route; exact residential criteria are not verified here.", "Check commercial egress/accessibility/guard/glazing interfaces under the approved design; do not reuse dwelling dimensions as commercial criteria."),
        "elevators": ("Only if vertical transport is installed or otherwise required by the applicable design/scope; no automatic installation obligation is inferred from low-rise use.", "Review installed or required vertical transport with the actual commercial access/occupancy strategy and specialist authority; exact trigger not verified here."),
        "special-inspections": ("Determine whether the specific dwelling structure/material/system triggers special inspections; not a blanket exemption for residential buildings.", "Determine special-inspection scope from actual commercial systems/materials and applicable SBC201 chapter17 provisions; not universally every listed test."),
        "occupancy-approval": ("Use the applicable residential permit/report channel and actual municipal occupancy approval; model results cannot issue or replace a certificate.", "The verified nonresidential occupancy service refers to an engineering office/report and municipal approval; confirm the actual commercial project's complete approvals."),
        "alteration-rehabilitation": ("Review the actual dwelling alteration, repair or change of use and relevant SBC901/SBC310 scope; original residential status does not settle a new use.", "Review the actual commercial repair/alteration/change of occupancy and relevant existing-building route; approved reassessment may differ from the original use."),
        "special-occupancies": ("Conditional on actual pools, sewage, heritage, green or other specialist features; none is assumed present in every dwelling.", "Conditional on actual commercial occupancy, hazards and specialist features; identify additional authority/code requirements rather than treating commercial as one category."),
    }
    if slug in overrides:
        residential, commercial = overrides[slug]
        basis = "occupancy_height_or_scope_dependent" if slug in {
            "permit-design", "energy-fire-access-design", "inspection-plan", "thermal-insulation", "energy-controls",
            "firestop-penetrations", "fire-doors", "egress-routes", "suppression-installation", "alarm-emergency-light",
            "smoke-compartment", "fire-integrated", "gypsum-rated", "finish-material", "stairs-guards", "elevators",
            "special-inspections", "occupancy-approval", "alteration-rehabilitation", "special-occupancies"
        } else "system_or_design_dependent"
    suffix = " Exact applicability/acceptance clauses were not fully verified; project/authority check required."
    return {"low_rise_residential": residential + suffix, "commercial": commercial + suffix}, basis


def add(stage, slug, title, evidence_type, codes, evidence, visual=None, capability=None):
    references = []
    for source_id, section in codes:
        source = SOURCE_BY_ID[source_id]
        references.append({"source_id": source_id, "title": source["title"], "url": source["url"], "section": section,
                           "reference_scope": "verified_section_heading_or_general_topic_mapping; not a verified acceptance rule"})
    applicability, basis = building_applicability(stage, slug)
    INSPECTIONS.append({
        "id": slug, "stage": stage, "title": title, "code_references": references,
        "building_applicability": applicability, "applicability_basis": basis,
        "evidence_type": evidence_type, "capability_id": capability,
        "required_evidence": evidence.split("; "), "automation_status": "research_only",
        "planned_capability_ids": {"ppe-head-body": ["ppe_hardhat", "ppe_vest"],
                                   "surface-defects": ["surface_crack"]}.get(slug, []),
        "video_observability": visual or ("candidate_visual_observation_only" if evidence_type == "video_visual" else "video_alone_insufficient"),
        "validation_status": "not_validated_for_saudi_code_acceptance",
        "default_assessment": "unassessed", "acceptance_criteria": "Project-specific approved design, applicable controlled code and qualified reviewer required.",
    })


# This is an original engineering evidence plan grouped around verified code topics.
# Topic references do not assert that every example is a universal statutory requirement.
def build():
    g = "sbc201-2024"; i = "sbc-2024-index"; c = "sbc304-2024-toc"; s = "sbc303-2024-toc"
    e = "sbc401-2024-toc"; p = "sbc701-2024-toc"; m = "sbc501-2024"; f = "sbc801-2024-toc"
    r = "implementing-regulations"; guide = "inspector-guide-2019"
    rows = [
        ("preconstruction", "permit-design", "Permit, approved design and occupancy classification", "design_review", [(r,"Articles 13-14"),(g,"105; 107")], "Authority-approved permit; signed coordinated drawings; project occupancy and construction classification"),
        ("preconstruction", "survey-boundaries", "Plot boundaries, levels and setbacks", "survey", [(guide,"5.1"),(g,"110.2")], "Licensed survey and benchmarks; approved site plan; calibrated site measurements"),
        ("preconstruction", "geotechnical-design", "Ground investigation, bearing conditions and foundation design", "lab_test", [(s,None)], "Borehole and laboratory reports; geotechnical interpretation; signed foundation design"),
        ("preconstruction", "inspection-plan", "Inspection and test plan, competent parties and hold points", "document_review", [(g,"110; Chapter 17"),(r,"Articles 21-22")], "Approved inspection and test plan; authority interpretation of stage approvals; qualifications and responsibilities"),
        ("preconstruction", "material-submittals", "Materials, product approvals and supplier traceability", "document_review", [(guide,"Annex 1"),(g,"Chapter 17")], "Approved submittals; product certificates; batch traceability; required material tests"),
        ("preconstruction", "energy-fire-access-design", "Coordinated energy, fire and accessibility design", "design_review", [(g,"Chapters 7-11; 13"),(i,None)], "Signed discipline designs; energy calculations; fire strategy; accessibility assessment"),
        ("site_safety", "ppe-head-body", "Visible helmet and high-visibility garment observations", "video_visual", [(guide,"5.2; inspection visit form"),(g,"Chapter 33")], "Dated site video with visible workers; competent safety review; PPE specification and site risk assessment"),
        ("site_safety", "ppe-other", "Foot, eye, hand and respiratory protection suitability", "manual_inspection", [(guide,"5.2"),(g,"Chapter 33")], "Close inspection of equipment condition and specification; task risk assessment; fit testing where applicable"),
        ("site_safety", "site-boundary-barriers", "Site fencing, warning signs and public separation", "video_visual", [(g,"Chapter 33"),(guide,"inspection visit form")], "Surveyed site boundary; multi-view site video; pedestrian and traffic management plan"),
        ("site_safety", "height-openings", "Open edges, floor openings and fall-protection arrangements", "manual_measurement", [(g,"Chapter 33")], "Close video before concealment; guardrail and anchorage measurements; competent temporary-works inspection"),
        ("site_safety", "scaffolds-access", "Scaffolds, ladders, access platforms and lifting arrangements", "qualified_inspection", [(g,"Chapter 33")], "Competent inspection; load and anchorage design; equipment certifications; access-route video"),
        ("site_safety", "housekeeping-traffic", "Housekeeping, material storage and equipment/pedestrian separation", "video_visual", [(guide,"5.1; 5.2"),(g,"Chapter 33")], "Timestamped site walkthrough; traffic layout; qualified review of hazard observations"),
        ("site_safety", "temporary-services", "Temporary electrical supply, water and worker welfare", "functional_test", [(guide,"5.1"),(e,None),(p,"311")], "Temporary installation inspection; protection and grounding tests; welfare and utility plan"),
        ("site_safety", "construction-fire", "Construction-phase fire prevention and emergency access", "qualified_inspection", [(f,"Chapter 33"),(g,"Chapter 33")], "Site fire plan; actual equipment inspection; clear emergency routes; authority conditions"),
        ("earthworks", "excavation-condition", "Excavation geometry, support and adjacent-property protection", "survey", [(g,"Chapter 33"),(s,None)], "Surveyed excavation profile; shoring design; competent inspection; adjacent-building condition records"),
        ("earthworks", "founding-level", "Founding level and exposed bearing stratum", "qualified_inspection", [(g,"110.3.1"),(s,None)], "Geotechnical acceptance of exposed stratum; calibrated levels; approved foundation drawings"),
        ("earthworks", "groundwater", "Groundwater, drainage and dewatering performance", "sensor_measurement", [(s,None)], "Water-level monitoring; approved dewatering design; discharge records; inspection of surrounding movement"),
        ("earthworks", "fill-compaction", "Fill material, layer thickness and compaction", "lab_test", [(s,None),(guide,"5.2")], "Approved fill source; layer measurements; field density and moisture tests; geotechnical acceptance"),
        ("earthworks", "special-ground", "Expansive, collapsible or sabkha ground treatment where present", "lab_test", [(s,"Chapters 9-11 (contents verified)")], "Site-specific soil classification; treatment design; field and laboratory verification"),
        ("earthworks", "buried-services", "Buried services, protection and backfill before covering", "survey", [(g,"110.3.2; 110.3.9"),(p,"306")], "Before-cover walkthrough; surveyed routes and levels; pressure or drainage tests; approved utility drawings"),
        ("foundations", "footing-prepour", "Footing and foundation inspection before concrete placement", "qualified_inspection", [(g,"110.3.1"),(s,None)], "Pre-pour inspection signed by competent engineer; approved dimensions; reinforcement and bearing checks"),
        ("foundations", "piles-deep", "Deep foundations and pile integrity where used", "lab_test", [(s,None),(g,"Chapter 17")], "Installation logs; specified integrity and load tests; geotechnical and structural acceptance"),
        ("foundations", "foundation-moisture", "Below-ground damp-proofing, waterproofing and drainage", "functional_test", [(s,"Chapter 13 (contents verified)")], "Before-cover membrane video; material records; measured laps and terminations; specified leak or integrity tests"),
        ("foundations", "slab-underfloor", "Slab and underfloor reinforcement, service penetrations and layers", "qualified_inspection", [(g,"110.3.2")], "Pre-cover inspection; approved slab and service drawings; layer thickness and level measurements"),
        ("foundations", "anchors-embedments", "Foundation anchors, starter bars and embedded fixtures", "manual_measurement", [(c,"26.7; 26.8 (contents verified)")], "Approved anchor details; calibrated positions and projection measurements; material and test certificates"),
        ("reinforcement", "rebar-grade", "Reinforcement grade, diameter and batch identification", "document_review", [(c,"26.6 (contents verified)")], "Mill certificates and bar marks; diameter measurements; material tests as specified"),
        ("reinforcement", "rebar-layout", "Reinforcement count, spacing and arrangement", "manual_measurement", [(c,"26.6 (contents verified)"),(g,"110.3.1; 110.3.4")], "Calibrated pre-pour measurements; signed bar schedule; multi-view evidence with scale"),
        ("reinforcement", "rebar-cover", "Concrete cover, spacers and chairs", "manual_measurement", [(c,"26.6 (contents verified)")], "Measured cover before placement; approved cover design; spacer suitability; cover-meter checks where specified"),
        ("reinforcement", "rebar-laps", "Splices, development, couplers and anchorage", "manual_measurement", [(c,"25.4; 25.5; 26.6 (contents verified)")], "Measured lap and anchorage details; approved drawings; coupler certificates and tests"),
        ("reinforcement", "rebar-condition", "Visible reinforcement contamination, damage and corrosion indications", "video_visual", [(c,"26.6 (contents verified)")], "Close pre-pour images or video; engineer assessment of section loss and acceptable cleaning; material records"),
        ("reinforcement", "posttension", "Prestressing tendons, anchorages and grouting where used", "functional_test", [(c,"26.10 (contents verified)")], "Specialist installation records; calibrated stressing logs; grout tests; approved tendon geometry"),
        ("formwork", "formwork-geometry", "Form dimensions, alignment, levels and openings", "survey", [(c,"26.11 (contents verified)")], "Surveyed formwork geometry; approved drawings; signed pre-pour inspection"),
        ("formwork", "formwork-support", "Shoring, bracing and temporary support stability", "qualified_inspection", [(c,"26.11 (contents verified)")], "Temporary-works design; competent support inspection; load and ground-bearing checks"),
        ("formwork", "formwork-cleanliness", "Form cleanliness, joints and visible preparation", "video_visual", [(c,"26.11 (contents verified)")], "Close pre-pour walkthrough; approved release-agent details; engineer approval"),
        ("formwork", "striking-reshoring", "Form striking and reshoring readiness", "lab_test", [(c,"26.11; 26.12 (contents verified)")], "Required in-place strength evidence; approved striking sequence; engineer release"),
        ("concreting", "concrete-delivery", "Concrete mix, delivery identification and placement records", "document_review", [(c,"26.4; 26.5 (contents verified)")], "Approved mix; batch and delivery tickets; placement location and time traceability"),
        ("concreting", "fresh-concrete", "Fresh-concrete consistency, temperature and air tests as specified", "lab_test", [(c,"26.4; 26.5 (contents verified)")], "Calibrated fresh-concrete tests; sampling records; project acceptance limits"),
        ("concreting", "placement-compaction", "Placement sequence and visible vibration/segregation indications", "video_visual", [(c,"26.5 (contents verified)")], "Timestamped actual placement video; engineer review; approved method and workability tests"),
        ("concreting", "curing-environment", "Curing coverage, duration and environmental protection", "sensor_measurement", [(c,"26.5 (contents verified)")], "Curing logs; time and temperature measurements; material specifications; site observations"),
        ("concreting", "concrete-strength", "Concrete hardened strength and acceptance", "lab_test", [(c,"26.12 (contents verified)")], "Traceable specimens; accredited strength tests; approved acceptance review; further testing if directed"),
        ("concreting", "surface-defects", "Visible concrete cracks, honeycombing, spalling and exposed steel indications", "video_visual", [(c,"26.12; 26.13 (contents verified)"),(i,None)], "Close video with lighting and scale; mapped defect locations; calibrated crack width and depth investigations; engineer assessment"),
        ("structure", "frame-geometry", "Frame plumb, levels, dimensions and floor-by-floor continuity", "survey", [(g,"110.3.4")], "Calibrated as-built survey; approved structural drawings; engineer inspection for each relevant floor"),
        ("structure", "steel-connections", "Structural steel connections, bolt installation and welds", "qualified_inspection", [(g,"1705.2"),(i,None)], "Fabricator traceability; connection drawings; bolt checks; required weld inspection and nondestructive tests"),
        ("structure", "masonry-work", "Masonry units, mortar, ties, reinforcement and grout", "qualified_inspection", [(i,None),(g,"Chapter 17")], "Approved SBC 305 design; material tests; measured tie and reinforcement layout; grout records"),
        ("structure", "precast-installation", "Precast component bearing, lifting, connections and grouting", "qualified_inspection", [(i,None),(c,"26.9 (contents verified)")], "SBC 312 project details; component certificates; surveyed bearings; connection and grout inspection"),
        ("structure", "movement-damage", "Visible structural movement, impact damage and deformation indications", "video_visual", [(g,"Chapter 17"),(i,None)], "Dated multi-view record; measured movement; structural engineer investigation; repair approval"),
        ("envelope", "roof-membrane", "Roof waterproofing, terminations and penetrations", "functional_test", [(g,"Chapter 15")], "Before-cover close video; approved membrane system; measured details; specified water or integrity test"),
        ("envelope", "wetarea-waterproof", "Wet-area membrane continuity and leak resistance", "functional_test", [(g,"110.3.8"),(p,None)], "Before-cover inspection; approved waterproofing detail; measured falls; witnessed flood or leak test"),
        ("envelope", "facade-anchors", "Facade and cladding anchors, fixings and support", "qualified_inspection", [(g,"Chapter 14; Chapter 17")], "Approved facade design; product certificates; anchor installation checks and specified tests"),
        ("envelope", "glazing-openings", "Windows, glazing, doors and visible sealing details", "manual_measurement", [(g,"Chapters 14; 24")], "Safety-glazing identification; approved opening schedule; measured installation; specified water/air tests"),
        ("envelope", "thermal-insulation", "Insulation continuity and thermal performance", "design_review", [(g,"110.3.7"),(i,None)], "Before-cover video; certified thermal properties; thickness measurements; SBC 601 or 602 energy design and checks"),
        ("envelope", "roof-drainage", "Roof falls, primary drains and overflow paths", "survey", [(g,"Chapter 15"),(p,"Chapter 11 (contents verified)")], "Calibrated falls and outlet measurements; drainage design; witnessed flow testing"),
        ("mep_roughin", "electrical-routing", "Electrical cable routing, protection, supports and enclosures", "qualified_inspection", [(g,"110.3.9"),(e,None)], "Approved electrical design; before-cover walkthrough; measured clearances; circuit and product records"),
        ("mep_roughin", "earthing-bonding", "Earthing and protective bonding continuity", "functional_test", [(e,"61-4.2; 61-4.3 (contents verified)")], "Calibrated continuity and earth tests; approved earthing design; accessible joint inspection"),
        ("mep_roughin", "plumbing-distribution", "Water distribution materials, supports, joints and protection", "qualified_inspection", [(p,"Chapters 3; 6 (contents verified)"),(g,"110.3.9")], "Approved plumbing design; before-cover inspection; product records; pressure test"),
        ("mep_roughin", "drainage-vents", "Drainage, traps, gradients, vents and cleanout access", "manual_measurement", [(p,"Chapters 7; 9; 10 (contents verified)")], "Calibrated slope and size checks; approved layout; witnessed leakage or flow tests"),
        ("mep_roughin", "potable-backflow", "Potable water separation and backflow prevention", "functional_test", [(p,"608; 610 (contents verified)")], "Approved backflow device records; installation and functional tests; disinfection and water-quality records"),
        ("mep_roughin", "hvac-ductwork", "Ductwork, supports, insulation and access", "qualified_inspection", [(m,None),(g,"110.3.9")], "Approved mechanical design; before-cover survey; product records; leakage tests where specified"),
        ("mep_roughin", "hvac-plant", "HVAC equipment, condensate and refrigerant installation", "functional_test", [(m,None),(p,"314 (contents verified)")], "Approved schedules; clearance measurements; leak and condensate tests; manufacturer startup records"),
        ("mep_roughin", "fuel-gas", "Fuel gas routing, valves, ventilation and leak resistance where installed", "functional_test", [(i,None),(g,"110.3.9")], "Applicable SBC 1201 design; licensed specialist inspection; calibrated pressure and leak tests"),
        ("mep_roughin", "telecom-infrastructure", "Telecom pathways, spaces, cabling and interfaces", "document_review", [("telecom-guide-announcement",None)], "Applicable official telecom guide; coordinated drawings; pathway inspections; cable test records"),
        ("fire_life_safety", "firestop-penetrations", "Fire and smoke barrier penetrations and joints before covering", "qualified_inspection", [(g,"110.3.6; Chapter 7")], "Before-cover close video; listed tested system matching actual substrate and services; measured installation; qualified signoff"),
        ("fire_life_safety", "fire-doors", "Fire doors, dampers and rated opening protection", "functional_test", [(g,"Chapter 7"),(f,None)], "Product labels and listings; installation measurements; actual operation tests; approved fire strategy"),
        ("fire_life_safety", "egress-routes", "Exit routes, stairs, exit doors and obstruction indications", "manual_measurement", [(g,"Chapter 10")], "Complete route survey; measured widths and travel distances; occupancy calculations; door operation tests"),
        ("fire_life_safety", "suppression-installation", "Sprinklers, standpipes, pumps and suppression equipment", "functional_test", [(f,None),(g,"Chapter 9")], "Approved hydraulic design; coverage and clearance measurements; witnessed pressure and flow tests"),
        ("fire_life_safety", "alarm-emergency-light", "Alarm devices, emergency lighting and exit signage", "functional_test", [(f,None),(g,"Chapters 9; 10")], "Approved cause-and-effect schedule; detector tests; audibility/visibility and emergency-duration measurements"),
        ("fire_life_safety", "smoke-compartment", "Smoke control, compartmentation and smoke-management performance", "functional_test", [(g,"909; Chapter 17"),(f,None)], "Approved fire strategy; pressure and airflow measurements; witnessed integrated smoke-control testing"),
        ("accessibility", "accessible-route", "Accessible paths, ramps, curb transitions and landings", "manual_measurement", [(g,"Chapter 11")], "Complete calibrated route survey; measured slopes and clear widths; applicable approved accessibility design"),
        ("accessibility", "accessible-parking", "Accessible parking layout and connection to entrances", "survey", [(g,"Chapter 11")], "Surveyed dimensions and gradients; approved parking plan; signage and route inspection"),
        ("accessibility", "accessible-doors", "Accessible door clearances, hardware and operating forces", "manual_measurement", [(g,"Chapter 11")], "Calibrated opening and force measurements; approved door schedule; functional inspection"),
        ("accessibility", "accessible-sanitary", "Accessible sanitary fixtures and maneuvering spaces", "manual_measurement", [(g,"Chapter 11"),(p,"404 (contents verified)")], "Calibrated fixture and space survey; approved layout; physical operation review"),
        ("finishes", "gypsum-rated", "Gypsum/lath installation in rated or shear assemblies before finishing", "qualified_inspection", [(g,"110.3.5")], "Before-finish close evidence; approved assembly; measured fastening and joint details; product records"),
        ("finishes", "finish-surface", "Visible finish cracks, damaged tiles, uneven or incomplete surfaces", "video_visual", [(g,"Chapters 8; 12")], "Actual walkthrough with scale and lighting; approved finish specification; calibrated flatness checks; reviewer classification"),
        ("finishes", "finish-material", "Interior finish fire performance and material identity", "document_review", [(g,"Chapter 8")], "Traceable finish product certificates; applicable fire classification; approved room finish schedule"),
        ("finishes", "stairs-guards", "Stairs, handrails, guards and glazing at occupied spaces", "manual_measurement", [(g,"Chapters 10; 24")], "Calibrated heights and openings; approved geometry; anchorage and loading evidence where specified"),
        ("testing_commissioning", "electrical-verification", "Electrical initial verification and protection operation", "functional_test", [(e,"61-4.2; 61-4.3; 61-4.4 (contents verified)")], "Calibrated electrical test results; protection-operation tests; signed initial-verification report"),
        ("testing_commissioning", "plumbing-verification", "Water and drainage pressure, leakage and sanitary verification", "functional_test", [(p,"112; 312 (contents verified)")], "Witnessed calibrated pressure and leakage records; drainage tests; approved test procedures"),
        ("testing_commissioning", "ventilation-balancing", "Airflow, ventilation and system balancing", "sensor_measurement", [(m,None),(g,"110.3.7; 110.3.9")], "Calibrated air and water balancing reports; approved design duties; actual operating measurements"),
        ("testing_commissioning", "energy-controls", "Energy controls, metering and equipment performance", "functional_test", [(g,"110.3.7"),(i,None)], "Applicable SBC 601 or 602 commissioning requirements; controls sequences; witnessed functional records"),
        ("testing_commissioning", "fire-integrated", "Integrated fire alarm, suppression, smoke and shutdown tests", "functional_test", [(f,None),(g,"Chapters 9; 17")], "Approved cause-and-effect test script; witnessed integrated tests; competent fire reviewer and authority records"),
        ("testing_commissioning", "elevators", "Elevators and vertical transport acceptance where installed", "qualified_inspection", [(g,"Chapter 30")], "Specialist inspection and applicable certification; load, emergency and safety-function tests"),
        ("testing_commissioning", "special-inspections", "Project-specific special inspections and structural observations", "qualified_inspection", [(g,"110.3.10; Chapter 17")], "Approved special-inspection statement; required site and laboratory reports; closure of discrepancies"),
        ("handover", "final-inspection", "Final inspection and closure of outstanding findings", "qualified_inspection", [(g,"110.3.11"),("occupancy-service",None)], "Final inspections; documented corrective work; authorized engineering report; authority acceptance"),
        ("handover", "asbuilt-records", "As-built drawings, warranties, operation and maintenance records", "document_review", [(r,"Article 17"),(guide,"5.3")], "Signed as-built drawings; accepted tests; product and maintenance documentation; training records"),
        ("handover", "external-final", "Final site drainage, public interfaces, parking and entrances", "survey", [(guide,"5.3"),(g,"110.3.11")], "Surveyed completed site; actual final walkthrough; approved site plan; accessibility and drainage evidence"),
        ("handover", "occupancy-approval", "Municipal occupancy certificate and required discipline approvals", "authority_approval", [(g,"111"),(r,"Articles 16-18"),("occupancy-service",None)], "Issued authority certificate; final inspection and test acceptance; actual permit and approval records"),
        ("operation_changes", "maintenance-monitoring", "Periodic inspection, maintenance and defect follow-up", "qualified_inspection", [(e,"61-5 (contents verified)"),(i,None)], "Applicable periodic inspection plan; maintenance logs; measured defect changes; competent review"),
        ("operation_changes", "alteration-rehabilitation", "Alteration, repair, change of use and existing-building assessment", "design_review", [(i,None),(r,"Articles 5; 7")], "Applicable SBC 901 or 310 scope; structural and fire assessment; authority-approved alteration documents"),
        ("operation_changes", "special-occupancies", "Project-specific systems and occupancy conditions", "qualified_inspection", [(i,None)], "Applicability review for pools SBC 1401, private sewage SBC 702, heritage SBC 902, green SBC 1001 and special occupancies; approved specialist test and inspection plan"),
    ]
    for row in rows:
        add(*row)


if __name__ == "__main__":
    build()
    stages = [{"id": key, "label": label, "order": number} for number, (key, label) in enumerate(STAGES, 1)]
    catalog = {"schema_version": 1, "researched_on": ACCESSED,
               "scope": "Saudi building lifecycle research checklist for BOTH low-rise residential and commercial buildings, with per-topic differences marked. Broad stage coverage is not exhaustive clause coverage or automatic detection. Every item starts unassessed; no model observation establishes code compliance. Project applicability, edition and acceptance criteria require a competent Saudi reviewer and relevant authority.",
               "building_scopes": {
                   "low_rise_residential": "First confirm eligibility for SBC1101-1102: the verified official description covers qualifying one-/two-family dwellings no more than three stories. Low-rise apartments, other residential occupancies and mixed use do not automatically qualify. Apply the confirmed residential/discipline route and project authority instructions.",
                   "commercial": "Use the actual commercial occupancy, height, construction and systems to select SBC201 and applicable discipline families. Commercial is not one uniform occupancy. Mixed-use residential portions need separate scope review.",
                   "interpretation_status": "Engineering evidence-planning interpretation of verified official scope/topic descriptions; full project applicability clauses and numerical acceptance criteria are not verified.",
                   "source_ids": ["sbc-residential-scope", "sbc-2024-index", "sbc201-2024"],
               },
               "stages": stages, "inspections": INSPECTIONS, "sources": SOURCES}
    path = ROOT / "edge_runtime/data/construction-inspections.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Saudi construction lifecycle evidence checklist", "", f"Generated from the research catalog on {ACCESSED}. {len(INSPECTIONS)} inspection families across {len(stages)} stages.", "", "Scope: **both low-rise residential and commercial buildings**, with separate applicability notes in every row. These are review tasks, not claims of implemented detection or a legally exhaustive checklist. Exact code applicability and acceptance rules require controlled documents and the project authority. All entries are initially **unassessed**. `video_visual` identifies a possible visible observation; acceptance still needs the evidence listed.", "", "The verified SBC1101-1102 description concerns qualifying one-/two-family dwellings no more than three stories. A low-rise apartment or mixed-use building does not automatically qualify. Commercial occupancy and systems determine its SBC201/discipline route. Energy, fire and accessibility criteria can differ; no unverified numerical thresholds are supplied. See the [both-scope comparison and source limitations](saudi-construction-inspection-study.md#low-rise-residential-and-commercial-scope).", ""]
    for stage in stages:
        lines += [f"## {stage['order']}. {stage['label']}", "", "| ID | Inspection family | Evidence class | Low-rise residential | Commercial | Required evidence |", "|---|---|---|---|---|---|"]
        for item in INSPECTIONS:
            if item["stage"] == stage["id"]:
                refs = "; ".join(f"[{ref['title']}]({ref['url']})" + (f" {ref['section']}" if ref["section"] else " (family only)") for ref in item["code_references"])
                applicability = item["building_applicability"]
                lines += [f"| `{item['id']}` | {item['title']}<br>{refs}<br>Applicability: {item['applicability_basis'].replace('_', ' ')} | `{item['evidence_type']}` | {applicability['low_rise_residential']} | {applicability['commercial']} | {'; '.join(item['required_evidence'])} |"]
        lines += [""]
    (ROOT / "docs/saudi-construction-checklist.md").write_text("\n".join(lines), encoding="utf-8")
    assert len({i["id"] for i in INSPECTIONS}) == len(INSPECTIONS)
    assert all(i["stage"] in {s["id"] for s in stages} for i in INSPECTIONS)
    assert all(set(i["building_applicability"]) == {"low_rise_residential", "commercial"} and
               all(isinstance(v, str) and v for v in i["building_applicability"].values()) for i in INSPECTIONS)
    assert all(i["default_assessment"] == "unassessed" and i["automation_status"] == "research_only" for i in INSPECTIONS)
    print(f"Wrote {len(INSPECTIONS)} inspection families / {len(stages)} stages; no compliance decisions.")
