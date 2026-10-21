"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const profiles = {
    factory: {label: "Factory", title: "Factory inspection", description: "Inspect visible product faults using an approved reference or a supported local surface model.", scope: "Visible product / surface inspection", hint: "Reference mode needs an approved good image; surface mode needs a matching view.", limitation: "Visible product differences only. No electrical diagnosis, calibrated probability or field validation."},
    freight: {label: "Freight", title: "Freight packet readiness", description: "Read document pages locally and review extracted text and packet fields.", scope: "Local OCR · document review", hint: "Hold each page steady for capture and local OCR.", limitation: "OCR and document-type rules require human review. No signature or authenticity verification."},
    roads: {label: "Roads", title: "Road damage review", description: "Run a local road-damage model and track candidate regions for human review.", scope: "Local detection · human review required", hint: "Use a visible road view and review the configured region.", limitation: "Model candidates require review. No GPS, depth or engineering severity. Synthetic mode uses a separate heuristic."},
    construction: {label: "Construction", title: "Construction video inspection", description: "Review real site footage locally and plan Saudi construction inspections across building stages.", scope: "Real video · selected visual candidates", hint: "Select a real site video and a named visual model. Review each observation.", limitation: "Visual candidates only. No structural strength, hidden work, calibrated dimensions or Saudi code-compliance determination."},
  };
  let selected = "factory";
  let latest = null;
  let latestStateReceivedAt = null;
  let busy = false;
  let polling = false;
  let lastStatePollStartedAt = 0;
  let previewBusy = false;
  let previewUrl = null;
  let lastPreviewAttempt = 0;
  let previewFrameKey = null;
  let toastTimer = null;
  let lastSession = null;
  let configurationTouched = false;
  let firstState = true;
  let constructionCatalog = null;
  let constructionCapabilities = null;
  let recordedVideos = [];
  let publicPresets = [];
  let recordingEvidence = null;
  let evidenceFrames = [];
  let evidenceRequest = 0;
  let shownEvidenceFrame = null;
  let requestedConstructionStage = "all";
  let requestedDamageVariant = null;
  let requestedDamageCheck = "";
  let taskCatalog = {tasks: [
    {id:"walls_finishes",title:"Walls and finishes",guidance:"Hold a close, evenly lit view of the wall. Include correct paint and dry surfaces for comparison. Damp-looking appearance does not measure moisture.",categories:[{id:"damp_appearance",title:"Damp/stained appearance",status:"unavailable",reason:"Waiting for verified model evidence."},{id:"paint_finish",title:"Peeling, blisters and efflorescence",status:"unavailable",reason:"Each condition needs independent validation."}]},
    {id:"brackets_railings",title:"Brackets and railings",guidance:"Show the whole bracket, then each fixing close up. Supply the approved expected positions when checking for missing components.",categories:[{id:"fastener_visible_fault",title:"Missing or visibly unseated fastenings",status:"unavailable",reason:"Visible state only; torque and concealed anchors remain unassessed."}]},
    {id:"tile_installation",title:"Tile installation",guidance:"Show a steady, near-overhead view with a measured reference on the tile plane. Enter the specified joint width and allowed deviation.",categories:[{id:"joint_geometry",title:"Joint spacing and alignment",status:"research",reason:"Measurement validation is required before fault decisions."},{id:"tile_condition",title:"Grout and tile condition",status:"unavailable",reason:"Installed-tile evidence is required."}]},
  ]};
  let inspectionUrl = null;
  let inspectionBusy = false;
  let inspectedKey = null;
  let displayedEvidence = null;
  let requestedResearchModel = "";
  let researchCatalogLoaded = false;
  // Frozen evaluation summaries apply only to these exact weight hashes.
  // A replacement checkpoint must supply its own evidence, never inherit scores.
  const researchEvaluations = {
    "043cd896209071d3f2607eaa5a061c2b2922df046262151ddf6b9ad5a2b2c40b": ["Peeling paint and publisher Mold appearance; not a dampness model.", "19 TEST photographs: peeling 4/22 boxes matched, 9 unmatched predictions (30.8% precision, 18.2% recall). Mold 1/3 matched; one prediction cannot establish reliability.", "Photo-only evidence, unknown physical sites. The owner damp-wall replay is unassessed."],
    "cb6e8fe431df9dfc93cee642f79db3770f8999782e611e6158d7c790c86f9127": ["Peeling paint and publisher Mold appearance; not a dampness model.", "19 TEST photographs: peeling 6/22 boxes matched, 40 unmatched predictions (13.0% precision, 27.3% recall). Mold 0/3 matched.", "Photo-only evidence, unknown physical sites. Both categories fail the quality target."],
    "b036d57e57708e12b251ce2fc1170b8c638883ed65fac0c2cb6849d3034a8c52": ["Publisher loose bolt/nut and missing-component appearance; arbitrary plate gaps are not trained.", "98 TEST photos: unseated 19/65 matched, 160 annotation-unmatched boxes; absence 6/71 matched, 67 unmatched. Publisher Fixed photos flagged: 21/30 and 8/30.", "Incomplete cross-class annotations and unknown sites. The owner mounting-plate gap remains unresolved."],
    "3b115f02158bc48254792f6fe4bf6c32801a14912123ecac1a91464818dc6e12": ["Publisher loose bolt/nut and missing-component appearance; arbitrary plate gaps are not trained.", "98 TEST photos: unseated 0/65 matched; absence 4/71 matched. No annotation-unmatched boxes at the frozen point, but almost every annotated instance is missed.", "No independent installation/video qualification. Silence does not prove a secure connection."],
    "498a2a678dc8488397fb5f7cac9804b7fb09994772b9d68fd73ce31a836ce467": ["Extended publisher bolt/nut fit; not a qualified plate-gap detector.", "98 reused development photos: unseated 24/65 matched, 79 annotation-unmatched boxes (23.3% precision, 36.9% recall); absence 4/71 matched. Fixed photos: 9/30 flagged unseated.", "Owner fastening video: zero candidates on all 207 frames despite the visible plate gap. This requested case is still missed."],
    "6091f973b636bb13858e7912967b602e32a281a45f7a3e6183179b12ba6ba83d": ["Actual polygon masks for publisher loose/missing bolt-or-nut appearance.", "98 TEST photos: 14/65 unseated and 10/71 absence masks matched at mask IoU ≥0.5. Fixed photos flagged: 6/30 and 13/30.", "Poor localization, inadequate sample counts and unknown physical sites. No real-video release."],
    "c5d0a46ce93de6f78d3c17b94e2e9362d01f0727e39a7d99c280e3fdf7fad293": ["Actual polygon masks for publisher loose/missing bolt-or-nut appearance.", "98 TEST photos: 17/65 unseated masks matched at mask IoU ≥0.5; all 71 absence instances missed. Fixed photos flagged unseated: 10/30.", "Poor localization, inadequate sample counts and unknown physical sites. No real-video release."],
    "654384b6653b055583fe2b2a5e73920795ff5d2eb7844d988fd6c9b813649505": ["Visible leakage/stain appearance from publisher boxes; no humidity percentage or cause measurement.", "98 TEST photos, actual CPU adapter: 54/125 boxes matched, 1,917 unmatched predictions (2.74% precision, 43.2% recall). All 51 other-defect negative photos flagged.", "Owner damp video: candidates on all 330 frames, including broad cluttered regions. This has not localized the reported wall fault reliably."],
  };
  const text = (id, value) => { $(id).textContent = value; };
  const number = (value, digits = 0) => Number.isFinite(Number(value)) && value !== null && value !== undefined ? Number(value).toLocaleString(undefined, {minimumFractionDigits: digits, maximumFractionDigits: digits}) : "—";
  const toggle = (id, visible) => $(id).classList.toggle("hidden", !visible);
  const metric = (id, value, unit, digits = 0) => {
    const node = $(id);
    node.replaceChildren(document.createTextNode(number(value, digits)));
    if (unit) { const small = document.createElement("small"); small.textContent = unit; node.append(small); }
  };
  const tag = (id, label, tone = "neutral") => { text(id, label); $(id).className = `status-tag ${tone}`; };

  function notify(message, error = false) {
    text("toast", message);
    $("toast").className = `toast${error ? " error" : ""}`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toggle("toast", false), 4500);
  }

  async function request(path, body) {
    const response = await fetch(path, body === undefined ? {cache: "no-store"} : {
      method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body),
    });
    if (!response.ok) {
      let message = `Request failed (${response.status})`;
      try { const data = await response.json(); message = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data); } catch (_) { /* preserve HTTP status */ }
      throw new Error(message);
    }
    return response.json();
  }

  function chooseProfile(profile) {
    selected = profile;
    toggle("preset-evidence-note", false);
    const info = profiles[profile];
    document.querySelectorAll(".profile-button").forEach((button) => {
      const active = button.dataset.profile === profile;
      button.classList.toggle("selected", active);
      button.setAttribute("aria-pressed", String(active));
    });
    text("breadcrumb-profile", info.label);
    text("page-title", info.title);
    text("page-description", info.description);
    toggle("trigger-setting", profile === "factory");
    toggle("construction-settings", profile === "construction");
    toggle("construction-guided-settings", profile === "construction");
    if (profile === "construction" && guidedTask() && !configurationTouched) {
      $("input-fps").value=30; $("preview-fps").value=30; $("deadline").value=1000;
    }
    const realOnly = profile === "construction";
    if (!realOnly) $("construction-recording-player").pause();
    $("source").querySelector('option[value="demo"]').disabled = realOnly;
    $("source").querySelector('option[value="pdf"]').disabled = realOnly;
    if (realOnly && ["demo", "pdf"].includes($("source").value)) {
      $("source").value = "file"; toggle("file-field", true);
    }
    renderConstructionPlan();
    renderState(latest);
  }

  function guidedTask(options = null) {
    const task = options ? options.inspection_task : $("construction-task").value;
    return ["walls_finishes", "brackets_railings", "tile_installation"].includes(task) ? task : "";
  }

  function syncGuidedOptions(options) {
    $("construction-task").value = guidedTask(options);
    requestedResearchModel = guidedTask(options) ? options.research_model_manifest || "" : "";
    const roi = options.roi || [0,0,1,1];
    ["left","top","right","bottom"].forEach((side,index)=>$("material-roi-"+side).value=roi[index] ?? [0,0,1,1][index]);
    $("tile-nominal-joint").value = options.nominal_joint_mm ?? "";
    $("tile-joint-tolerance").value = options.tolerance_mm ?? "";
    $("construction-calibration-reference").value = options.calibration_reference ? JSON.stringify(options.calibration_reference,null,2) : "";
    $("construction-expected-fasteners").value = options.expected_fasteners ? JSON.stringify(options.expected_fasteners,null,2) : "";
    renderGuidedTask();
  }

  function renderResearchModels() {
    const task=guidedTask();
    const select=$("construction-research-model");
    const models=(taskCatalog.research_models || []).filter(model=>model.inspection_task===task);
    const wanted=requestedResearchModel || select.value;
    const signature=JSON.stringify([task,models.map(model=>[model.manifest,model.title])]);
    if(select.dataset.catalogSignature!==signature) {
      select.replaceChildren(new Option(task==="tile_installation" ? "No defect detector · geometry observations only" : "No detector · capture guidance only",""));
      models.forEach(model=>select.append(new Option(`${model.title || "Local model"} · NOT QUALIFIED`,model.manifest)));
      select.dataset.catalogSignature=signature;
    }
    const available=models.find(model=>model.manifest===wanted);
    select.value=available ? wanted : "";
    if(researchCatalogLoaded) requestedResearchModel=available ? wanted : "";
    const chosen=models.find(model=>model.manifest===select.value);
    const classes=chosen?.classes?.length ? ` Classes: ${chosen.classes.join(", ")}.` : "";
    text("construction-research-model-note",chosen ? `Experimental candidates only.${classes} No output is not a defect-free result.` : task==="tile_installation" ? "Geometry can report visible edge observations. No qualified tile-fault detector is selected." : "No detector is selected. Capture guidance cannot detect or rule out a fault.");
    const evaluation=chosen && researchEvaluations[chosen.model_sha256];
    text("construction-model-evaluation-title",chosen ? "FAILED / NOT QUALIFIED FOR INSPECTION" : "NO DETECTOR · CONDITIONS NOT ASSESSED");
    text("construction-model-evaluation-scope",evaluation?.[0] || (chosen ? chosen.reason || "No accepted performance evidence for this exact checkpoint." : task==="tile_installation" ? "Image-space edge measurements do not establish missing grout, tile damage or installation correctness." : "The preview and capture guidance do not assess installation correctness."));
    text("construction-model-evaluation-results",evaluation?.[1] || (chosen ? "Independent category validation has not passed. Installed weights do not establish working detection." : models.length ? "Available checkpoints are failed or unqualified experiments, not working inspection categories." : "No compatible trained detector is installed for this task."));
    text("construction-model-evaluation-limit",evaluation?.[2] || "No finding cannot establish that a wall, connection or tile installation is correct.");
  }

  function renderGuidedTask() {
    const task = guidedTask();
    toggle("construction-task-details", Boolean(task));
    toggle("construction-tile-options", task === "tile_installation");
    toggle("construction-settings", selected === "construction" && !task);
    renderResearchModels();
    const catalog = taskCatalog.tasks.find(item=>item.id===task);
    text("construction-task-guide", catalog?.guidance || "Select a task to see capture guidance and measured capability status.");
    const holder = $("construction-task-categories"); holder.replaceChildren();
    (catalog?.categories || []).forEach(category=>{
      const row=document.createElement("div"); row.className="task-category";
      const heading=document.createElement("strong"); heading.textContent=category.title || category.id;
      const state=document.createElement("span"); state.className="badge";
      const status=category.status || "unavailable";
      state.textContent=({validated:"VALIDATED",research:"NOT QUALIFIED",research_only:"NOT QUALIFIED",unavailable:"UNAVAILABLE"})[status] || status.replaceAll("_"," ").toUpperCase();
      const reason=document.createElement("p"); reason.textContent=category.reason || "No independent release evidence is recorded.";
      row.append(heading,state,reason); holder.append(row);
    });
  }

  async function loadConstructionTasks() {
    try { const catalog=await request("/api/construction/tasks"); if(Array.isArray(catalog.tasks)) { taskCatalog=catalog; researchCatalogLoaded=true; } }
    catch (_) { /* Fallback categories remain explicitly unverified. */ }
    renderGuidedTask();
  }

  function renderEvidenceAge(state) {
    const paired=state?.inspection_result;
    const age=state?.metrics?.evidence_age_ms;
    const sameSession=displayedEvidence && displayedEvidence.session_id===state?.session_id;
    const canEstimate=sameSession && Number.isFinite(age) && Number.isFinite(paired?.captured_monotonic) && Number.isFinite(displayedEvidence.captured_monotonic);
    const browserElapsed=Number.isFinite(latestStateReceivedAt) ? Math.max(0,performance.now()-latestStateReceivedAt) : 0;
    const shownAge=canEstimate ? age+Math.max(0,(paired.captured_monotonic-displayedEvidence.captured_monotonic)*1000)+browserElapsed : null;
    text("guided-evidence-age", `Shown evidence age ${shownAge===null ? "—" : "≈ "+number(shownAge,0)+" ms"}`);
    $("guided-evidence-age").title="Estimated age of the displayed inspected frame from host receipt, plus browser time since the latest status. Network transfer and camera exposure are not measured.";
  }

  function applyGuidedOptions(options) {
    const task=guidedTask();
    if(!task) { delete options.inspection_task; delete options.research_model_manifest; return; }
    options.inspection_task=task;
    // Guided controls own model selection; empty means no hidden advanced detector.
    delete options.research_model;
    const researchManifest=$("construction-research-model").value;
    if(researchManifest) {
      const model=(taskCatalog.research_models || []).find(model=>model.manifest===researchManifest && model.inspection_task===task);
      if(!model) throw new Error("Choose a compatible verified local research model, or leave the model selector empty.");
      options.research_model_manifest=researchManifest;
    } else delete options.research_model_manifest;
    const roi=["left","top","right","bottom"].map(side=>Number($("material-roi-"+side).value));
    if(roi.some(value=>!Number.isFinite(value) || value<0 || value>1) || roi[0]>=roi[2] || roi[1]>=roi[3]) throw new Error("Material area must have left < right and top < bottom, within 0–1.");
    options.roi=roi;
    [["construction-calibration-reference","calibration_reference",false],["construction-expected-fasteners","expected_fasteners",true]].forEach(([id,key,array])=>{
      const raw=$(id).value.trim();
      if(!raw) { delete options[key]; return; }
      let value; try { value=JSON.parse(raw); } catch(_) { throw new Error(`${array ? "Expected fastening positions" : "Scale reference"} must contain valid JSON.`); }
      if(!value || (array ? !Array.isArray(value) : Array.isArray(value) || typeof value!=="object")) throw new Error(`${array ? "Expected fastening positions must be an array" : "Scale reference must be an object"}.`);
      options[key]=value;
    });
    [["tile-nominal-joint","nominal_joint_mm",false],["tile-joint-tolerance","tolerance_mm",true]].forEach(([id,key,zero])=>{
      const raw=$(id).value.trim();
      if(task!=="tile_installation" || !raw) { delete options[key]; return; }
      const value=Number(raw);
      if(!Number.isFinite(value) || (zero ? value<0 : value<=0)) throw new Error("Joint width must be positive and allowed deviation must be zero or positive.");
      options[key]=value;
    });
  }

  function renderConstructionPlan() {
    const holder = $("construction-topics"); holder.replaceChildren();
    if (!constructionCatalog?.available) { text("construction-summary", "Catalog unavailable"); return; }
    const stage = $("construction-stage").value || "all";
    const rows = constructionCatalog.inspections.filter(row => stage === "all" || row.stage === stage);
    text("construction-summary", `${rows.length} inspection topics · not assessed`);
    constructionCatalog.stages.filter(item => stage === "all" || item.id === stage).forEach(item => {
      const details = document.createElement("details"), summary = document.createElement("summary");
      details.className = "construction-stage-group";
      summary.textContent = item.label; details.append(summary);
      rows.filter(row => row.stage === item.id).forEach(row => {
        const article = document.createElement("article"); article.className = "construction-topic";
        const heading = document.createElement("strong"); heading.textContent = row.title;
        const status = document.createElement("span"); status.className = "badge"; status.textContent = "NOT ASSESSED";
        const evidence = document.createElement("p"); evidence.textContent = `Evidence: ${(row.required_evidence || []).join("; ")}`;
        const method = document.createElement("p"); method.textContent = `Review method: ${String(row.evidence_type || "project review").replaceAll("_", " ")}`;
        const modelInfo = document.createElement("p");
        const planned = row.planned_capability_ids || [];
        const candidates = Object.entries(constructionCatalog.visual_models || {}).filter(([,model])=>model.installed && model.capability_ids.some(id=>planned.includes(id)));
        modelInfo.textContent = candidates.length ? `Available visual candidate model: ${candidates.map(([mode])=>mode).join(", ")}. Other aspects require separate evidence; assessment stays open.` : "Automatic inspection unavailable for this topic; collect the listed evidence.";
        const refs = document.createElement("div"); refs.className = "construction-references";
        (row.code_references || []).forEach(ref => {
          try {
            const url = new URL(ref.url); if (url.protocol !== "https:") return;
            const link = document.createElement("a"); link.href = url.href; link.target = "_blank"; link.rel = "noopener noreferrer";
            link.textContent = `${ref.title}${ref.section ? " · " + ref.section : ""}`; refs.append(link);
          } catch (_) { /* invalid source references are not clickable */ }
        });
        article.append(heading, status, method, modelInfo, evidence);
        const applicability = row.building_applicability || {};
        [["low_rise_residential", "Low-rise residential"], ["commercial", "Commercial"]].forEach(([key,label]) => {
          if (!applicability[key]) return;
          const scope = document.createElement("p"); scope.className = "construction-building-scope";
          const name = document.createElement("b"); name.textContent = `${label}: `;
          scope.append(name, document.createTextNode(applicability[key])); article.append(scope);
        });
        article.append(refs); details.append(article);
      });
      holder.append(details);
    });
  }

  function setConstructionStage(stage) {
    requestedConstructionStage = stage || "all";
    const select = $("construction-stage");
    const exists = Array.from(select.options).some(option => option.value === requestedConstructionStage);
    select.value = exists ? requestedConstructionStage : "all";
    if (!exists && constructionCatalog?.available) requestedConstructionStage = "all";
  }

  async function loadConstructionCatalog() {
    try {
      constructionCatalog = await request("/api/construction/catalog");
      constructionCatalog.stages.forEach(stage => {
        const option = document.createElement("option"); option.value = stage.id; option.textContent = stage.label;
        $("construction-stage").append(option);
      });
      // State/presets can arrive before the asynchronous catalog options.
      setConstructionStage(requestedConstructionStage);
      renderConstructionPlan();
    } catch (_) { text("construction-summary", "Catalog unavailable"); }
  }

  function updateDamageChecks() {
    const select=$("construction-damage-check");
    const model=(constructionCapabilities?.models || []).find(row=>row.variant===$("construction-damage-variant").value);
    select.replaceChildren(new Option("All supported visible conditions", ""));
    (model?.capabilities || []).forEach(id=>{
      const category=(constructionCapabilities?.categories || []).find(row=>row.id===id);
      select.append(new Option(category?.label || id.replace(/^surface_/,""),model.condition_classes?.[id] || id.replace(/^surface_/,"")));
    });
    if(requestedDamageCheck && ![...select.options].some(option=>option.value===requestedDamageCheck)) {
      select.append(new Option(`Unavailable check: ${requestedDamageCheck}`,requestedDamageCheck));
      select.options[select.options.length-1].disabled=true;
    }
    select.value=requestedDamageCheck;
  }

  async function loadConstructionCapabilities() {
    const holder = $("construction-capability-list"); holder.replaceChildren();
    try {
      constructionCapabilities = await request("/api/construction/capabilities");
      const rows = constructionCapabilities.categories || [];
      const working = rows.filter(row => row.effective_status === "supported_experimental").length;
      const weak = rows.filter(row => row.effective_status === "measured_weak").length;
      const variants=$("construction-damage-variant");
      const previous=requestedDamageVariant || variants.value;
      variants.replaceChildren(new Option("Choose an installed damage model…", ""));
      (constructionCapabilities.models || []).forEach(model=>{
        const option=new Option(model.label,model.variant); option.disabled=!model.installed; variants.append(option);
      });
      const installed=[...variants.options].filter(option=>option.value && !option.disabled);
      $("construction-damage-mode-option").disabled=!installed.length;
      const defaultModel=(constructionCapabilities.models || []).find(model=>model.installed && !model.diagnostic_only);
      variants.value=previous ? (installed.some(option=>option.value===previous) ? previous : "") : defaultModel?.variant || "";
      updateDamageChecks();
      if(previous && !variants.value && selected==="construction" && $("construction-mode").value==="damage") notify("The requested damage model is unavailable. Choose an installed model before starting.",true);
      text("construction-capability-summary", `${working} experimental · ${weak} weak · ${rows.length} tracked`);
      if (!rows.length) { const empty=document.createElement("p"); empty.textContent="Category evidence is being prepared."; holder.append(empty); }
      rows.forEach(row => {
        const card=document.createElement("article"); card.className="damage-category";
        const heading=document.createElement("h3"); heading.textContent=row.label || row.id;
        const state=document.createElement("span"); state.className="badge";
      const statuses={supported_experimental:"INSTALLED · EXPERIMENTAL",measured_weak:"INSTALLED · WEAK LOCALIZATION",research_only:"DATA / MODEL RESEARCH",unsupported:"NO DETECTOR",unavailable:"MODEL UNAVAILABLE"};
        state.textContent=statuses[row.effective_status] || "UNVERIFIED";
        card.append(heading,state);
        [[row.condition,"Condition"],[row.validation_scope,"Measured evidence"],[row.evidence_summary,"Result"],[row.next_check,"Next check"]].forEach(([value,label])=>{
          if (!value) return;
          const paragraph=document.createElement("p"), name=document.createElement("b"); name.textContent=`${label}: `;
          paragraph.append(name,document.createTextNode(String(value))); card.append(paragraph);
        });
        if(row.model_ref) { const model=document.createElement("p"); model.className="muted-text"; model.textContent=`Local model: ${row.model_ref}${row.model_installed ? " · file hash verified" : " · no verified installed asset"}`; card.append(model); }
        try {
          const url=new URL(row.data_source_url);
          if(url.protocol==="https:") { const link=document.createElement("a"); link.href=url.href; link.textContent="Dataset / model source"; link.target="_blank"; link.rel="noopener noreferrer"; card.append(link); }
        } catch (_) { /* Missing source is displayed in the research documentation. */ }
        (row.video_evidence || []).forEach(evidence=>{
          const button=document.createElement("button"); button.type="button"; button.className="button small secondary category-evidence-button";
          button.textContent="View recorded evidence"; button.title=evidence.scope || evidence.case_id;
          button.addEventListener("click",()=>{
            const select=$("construction-recording");
            if(![...select.options].some(option=>option.value===evidence.case_id && !option.disabled)) { notify("This evidence recording is unavailable in the current case list.",true); return; }
            select.value=evidence.case_id; select.dispatchEvent(new Event("change"));
            $("construction-recordings").scrollIntoView({behavior:"smooth",block:"start"});
          }); card.append(button);
        });
        (row.photo_evidence || []).forEach(evidence=>{
          const button=document.createElement("button"); button.type="button"; button.className="button small secondary category-evidence-button";
          button.textContent="Load real-photo diagnostic";
          button.addEventListener("click",()=>{
            const index=publicPresets.findIndex(preset=>preset.label===evidence.preset_label && preset.available);
            if(index<0) { notify("The photo or its pinned local model is unavailable.",true); return; }
            $("sample-preset").value=String(index); $("sample-preset").dispatchEvent(new Event("change"));
            $("sample-preset").scrollIntoView({behavior:"smooth",block:"center"});
          }); card.append(button);
        });
        holder.append(card);
      });
    } catch (_) { text("construction-capability-summary", "Category evidence unavailable"); }
  }

  function videoTime(ms) {
    if (!Number.isFinite(ms)) return "—";
    return `${Math.floor(ms/60000)}:${(Math.max(0,ms)%60000/1000).toFixed(1).padStart(4,"0")}`;
  }

  async function loadRecordedVideos() {
    try {
      recordedVideos = await request("/api/construction/videos");
      renderRecordingOptions();
    } catch (_) { /* Optional exported recordings do not affect live inspection. */ }
  }

  function showDiagnostics() { return $("construction-show-diagnostics").checked; }

  function renderRecordingOptions() {
    const select = $("construction-recording"), current = select.value;
    select.replaceChildren(new Option("Choose an inspection case...", ""));
    const cases = recordedVideos.filter(record => showDiagnostics() || record.purpose === "defect_evidence" && record.evidence_available);
    cases.forEach(record => {
      const option = new Option(`${record.purpose === "defect_evidence" ? "" : "DIAGNOSTIC · "}${record.label}${record.available ? "" : " (unavailable)"}`, record.id);
      option.disabled = !record.available; select.append(option);
    });
    if (cases.some(record => record.id === current)) select.value = current;
    else if (current) select.dispatchEvent(new Event("change"));
    const verified = cases.filter(record => record.available && record.purpose === "defect_evidence" && record.evidence_available).length;
    text("construction-recording-availability", verified ? `${verified} recorded damage cases with publisher annotations. These small public cases do not establish independent field accuracy.` : "No annotated damage case is available yet. Context footage and failed detectors are not inspection demonstrations.");
  }

  function renderPresetOptions() {
    const select = $("sample-preset"), current = select.value;
    select.replaceChildren(new Option("Choose a downloaded example…", ""));
    publicPresets.forEach((preset,index) => {
      if (preset.profile === "construction" && !["defect_evidence","photo_evidence"].includes(preset.purpose) && !showDiagnostics()) return;
      const option = new Option(preset.label + (preset.available ? "" : " (not downloaded)"), String(index));
      option.disabled = !preset.available; select.append(option);
    });
    if ([...select.options].some(option => option.value === current)) select.value = current;
  }

  function evidenceMetric(label, value) {
    const card = document.createElement("div"), name = document.createElement("span"), count = document.createElement("strong");
    name.textContent = label; count.textContent = Number.isFinite(value) ? number(value) : "Not evaluated";
    card.append(name,count); return card;
  }

  function seekEvidenceFrame(frame) {
    const player = $("construction-recording-player"); player.pause();
    const token=evidenceRequest, source=player.getAttribute("src");
    const seek = () => {
      if(token!==evidenceRequest || source!==player.getAttribute("src")) return;
      player.currentTime = Math.max(0,frame.video_time_s); renderCurrentEvidence(true);
    };
    if (player.readyState >= 1) seek();
    else { player.addEventListener("loadedmetadata",seek,{once:true}); player.load(); }
  }

  function renderCaseEvidence(evidence, record) {
    recordingEvidence = evidence;
    evidenceFrames = (evidence?.frames || []).filter(frame => Number.isFinite(frame.video_time_s)).sort((a,b)=>a.video_time_s-b.video_time_s);
    shownEvidenceFrame = null;
    toggle("construction-case-evidence", Boolean(evidence));
    if (!evidence) return;
    const ownerDiagnostics={
      "guided-fastening-ssd-v1-owner":"FAILED OWNER CASE: the visible separation is between the mounting plate and wall. The old model places candidates on 25 of 207 frames; its first box at 0.63 s covers the horizontal rail beside the connection, not the plate-to-wall gap. These boxes do not establish a missing fastener or secure anchors. Expected layout and fastening torque are unassessed.",
      "guided-fastening-before-after-v2-owner":"FAILED OWNER CASE: original / old model / new model show the same real plate-to-wall separation. The new model gives zero candidates on all 207 frames. This misses the requested plate-gap issue; an empty overlay is not a correct fastening result.",
      "guided-damp-stain-ssd-v1-owner":"FAILED OWNER CASE: the owner reports humidity damage. The yellow boxes are stain-appearance proposals, not measured moisture or confirmed damage boundaries. At the midpoint they overlap broad wall, baseboard and floor regions; all 330 frames have candidates. This clutter does not provide a useful localized wall-fault assessment.",
    };
    const diagnostic=ownerDiagnostics[record.id] || (record.purpose==="failure_diagnostic" ? "FAILURE DIAGNOSTIC: this playback exposes an unresolved model case. Boxes are experimental proposals; an empty overlay does not establish that the visible installation is correct." : "");
    text("construction-diagnostic-explanation",diagnostic); toggle("construction-diagnostic-explanation",Boolean(diagnostic));
    text("construction-case-title", evidence.title || record.label);
    text("construction-case-issue", evidence.issue_summary || record.issue_summary || "See the original annotation and model output.");
    text("construction-case-why", evidence.why_it_matters ? `Why inspect: ${evidence.why_it_matters}` : "");
    const summary = evidence.summary || {};
    const hasEvaluatedFrames=evidenceFrames.some(frame=>frame.evaluated===true);
    $("construction-case-summary").replaceChildren(
      evidenceMetric("Annotated frames",summary.annotated_frames), evidenceMetric("Localized matches",hasEvaluatedFrames ? summary.localized_matches : null),
      evidenceMetric("Unmatched annotations",hasEvaluatedFrames ? summary.missed_defects : null), evidenceMetric("Unmatched model boxes",hasEvaluatedFrames ? summary.false_alerts : null),
      evidenceMetric("Unsupported annotations",summary.unsupported_annotations));
    text("construction-case-scope", evidence.annotation_scope || summary.evaluation_scope || "Annotation scope has not been supplied; this is not a validated accuracy claim.");
    text("construction-case-next", evidence.next_check ? `Next inspection: ${evidence.next_check}` : "Review the visible condition against the original footage. Code acceptance requires its separate evidence.");
    const moments = $("construction-evidence-moments"); moments.replaceChildren();
    const stride = Math.max(1,Math.ceil(evidenceFrames.length/9));
    const matchCutoff=Number.isFinite(evidence.matching_iou_threshold) ? evidence.matching_iou_threshold : .5;
    const localized=evidenceFrames.find(frame=>frame.evaluated===true && (frame.predictions || []).some(prediction=>prediction.matched_annotation_id!=null && prediction.iou>=matchCutoff));
    const missed=evidenceFrames.find(frame=>frame.evaluated===true && (frame.expected || []).some(annotation=>annotation.id!=null && annotation.evaluation_status!=="unsupported" && !(frame.predictions || []).some(prediction=>prediction.matched_annotation_id===annotation.id && prediction.iou>=matchCutoff)));
    const selected=[...new Set([localized,missed,...evidenceFrames.filter((frame,index)=>index%stride===0)].filter(Boolean))].slice(0,12);
    selected.forEach(frame => {
      const button = document.createElement("button"); button.type = "button"; button.className = "button small secondary";
      button.textContent = `View ${frame===localized ? "localized damage" : frame===missed ? "missed damage" : frame.expected?.length ? "annotated frame" : "source frame"} ${videoTime(frame.video_time_s*1000)}`;
      button.addEventListener("click",()=>seekEvidenceFrame(frame)); moments.append(button);
    });
    renderCurrentEvidence(true);
  }

  function renderCurrentEvidence(force=false) {
    if (!recordingEvidence) return;
    const now = $("construction-recording-player").currentTime;
    let low=0,high=evidenceFrames.length;
    while (low<high) { const mid=(low+high)>>1; if(evidenceFrames[mid].video_time_s<=now+1e-6) low=mid+1; else high=mid; }
    const candidates=[low-1,low].filter(index=>index>=0 && index<evidenceFrames.length);
    const completeSourceTimeline=recordingEvidence.source?.original_camera_frames===evidenceFrames.length && evidenceFrames.every((frame,index)=>frame.source_frame_index===index);
    const index=completeSourceTimeline ? low-1 : candidates.sort((a,b)=>Math.abs(evidenceFrames[a].video_time_s-now)-Math.abs(evidenceFrames[b].video_time_s-now))[0];
    const tolerance=Number.isFinite(recordingEvidence.frame_time_tolerance_s) ? Math.max(0,Math.min(.5,recordingEvidence.frame_time_tolerance_s)) : .08;
    const frame=index === undefined || index<0 || !completeSourceTimeline && Math.abs(evidenceFrames[index].video_time_s-now)>tolerance ? null : evidenceFrames[index];
    const key=frame ? index : "unlabelled";
    if(!force && shownEvidenceFrame===key) return;
    shownEvidenceFrame=key;
    const holder=$("construction-current-evidence"); holder.replaceChildren();
    if(!frame) { const p=document.createElement("p"); p.textContent="This moment has no corresponding publisher frame record. Model output here has not been verified."; holder.append(p); return; }
    const expected=frame.expected || [], predictions=frame.predictions || [];
    const evaluated=frame.evaluated===true;
    const p=document.createElement("p"); p.textContent=`Original frame ${frame.source_frame_index ?? "unknown"} · ${videoTime(frame.video_time_s*1000)} · annotation and model output are separate evidence.`; holder.append(p);
    const list=document.createElement("ul");
    expected.forEach(annotation => {
      const unsupported=annotation.evaluation_status==="unsupported";
      const cutoff=Number.isFinite(recordingEvidence.matching_iou_threshold) ? recordingEvidence.matching_iou_threshold : .5;
      const match=evaluated && !unsupported && annotation.id!=null ? predictions.find(prediction=>prediction.evaluation_status!=="unscored" && prediction.matched_annotation_id===annotation.id && Number.isFinite(prediction.iou) && prediction.iou>=cutoff) : null;
      const unevaluated=!evaluated;
      const item=document.createElement("li"); item.className=unsupported || unevaluated ? "evidence-unsupported" : match ? "evidence-match" : "evidence-miss";
      const bbox=annotation.bbox, size=recordingEvidence.frame_size_wh;
      const location=Array.isArray(bbox) && bbox.length===4 ? ` · original pixel box ${bbox.map(n=>number(n,0)).join(", ")}${Array.isArray(size) ? ` in ${size.join(" × ")} frame` : ""}` : "";
      item.textContent=`${unsupported ? "UNSUPPORTED BY THIS MODEL" : unevaluated ? "NOT ASSESSED" : match ? "LOCALIZED" : "MISSED"}: ${annotation.label || "annotated damage"}${location}${match && Number.isFinite(match.iou) ? ` · box IoU ${number(match.iou,2)}` : ""}.`;
      list.append(item);
    });
    predictions.filter(prediction=>evaluated && prediction.evaluation_status!=="unscored" && prediction.matched_annotation_id===null).forEach(prediction=>{
      const item=document.createElement("li"); item.className="evidence-false-alert";
      const cutoff=Number.isFinite(recordingEvidence.matching_iou_threshold) ? recordingEvidence.matching_iou_threshold : .5;
      const reason=prediction.iou>=cutoff ? "This is an extra box for an already matched annotation; the one-to-one rule counts it as unmatched." : "Counted as a false alert under the whole-instance matching rule; a partial damage box can fail this rule even when damage is visible.";
      item.textContent=`UNMATCHED MODEL BOX: ${prediction.label || "model candidate"}${Number.isFinite(prediction.iou) ? ` · best box IoU ${number(prediction.iou,2)}` : ""}. ${reason}`; list.append(item);
    });
    if (!evaluated || predictions.some(prediction=>prediction.evaluation_status==="unscored")) { const item=document.createElement("li"); item.textContent="Model candidates outside the annotated class or frame scope are unverified. They are not counted as localized damage or false alerts."; list.append(item); }
    if (!expected.length && !predictions.length && evaluated) { const item=document.createElement("li"); item.textContent="No annotated condition or model candidate in this evaluated frame; this is not a building inspection pass."; list.append(item); }
    holder.append(list);
  }

  async function loadPresets() {
    try {
      publicPresets = await request("/api/presets");
      renderPresetOptions();
      $("sample-preset").addEventListener("change", () => {
        const preset = publicPresets[Number($("sample-preset").value)];
        if ($("sample-preset").value === "" || !preset) return;
        chooseProfile(preset.profile);
        const separator = preset.source.indexOf(":");
        $("source").value = preset.source.slice(0, separator);
        $("file-path").value = preset.source.slice(separator+1);
        $("profile-options").value = JSON.stringify(preset.profile_options || {}, null, 2);
        if (preset.profile === "construction") {
          syncGuidedOptions(preset.profile_options || {});
          $("construction-mode").value = preset.profile_options?.mode || "ppe";
          requestedDamageVariant=preset.profile_options?.damage_variant || null;
          if(requestedDamageVariant) $("construction-damage-variant").value=requestedDamageVariant;
          requestedDamageCheck=preset.profile_options?.damage_classes?.length===1 ? preset.profile_options.damage_classes[0] : "";
          updateDamageChecks();
          setConstructionStage(preset.profile_options?.stage || "all");
          renderConstructionPlan();
        }
        $("input-fps").value = preset.input_fps || 4;
        $("deadline").value = preset.deadline_ms || 2000;
        $("video-mode").value = preset.video_mode || "realtime";
        $("preview-fps").value = preset.preview_fps ?? 6;
        $("triggered").checked = false;
        configurationTouched = true; toggle("file-field", true); renderState(latest);
        text("preset-evidence-note",preset.note || ""); toggle("preset-evidence-note",Boolean(preset.note));
        notify("Public-data settings loaded. Press Start inspection.");
      });
    } catch (error) { notify(error.message, true); }
  }

  function renderChecklist(state) {
    const holder = $("checklist");
    holder.replaceChildren();
    const checklist = state && state.checklist;
    toggle("checklist", Boolean(checklist));
    if (!checklist) return;
    Object.entries(checklist).forEach(([key, item]) => {
      const done = typeof item === "object" ? Boolean(item.captured) : Boolean(item);
      const label = typeof item === "object" ? item.label || key : key;
      const row = document.createElement("div"); row.className = `checklist-row${done ? " done" : ""}`;
      const mark = document.createElement("span"); mark.className = "check-mark"; mark.textContent = done ? "✓" : "·";
      const caption = document.createElement("span"); caption.textContent = `${label} · ${done ? "captured" : "needed"}`;
      row.append(mark, caption); holder.append(row);
    });
  }

  function renderFindings(findings, processed, explanations=[], constructionContext=null) {
    const holder = $("findings"); holder.replaceChildren();
    text("finding-count", findings.length);
    if (!findings.length) {
      const paragraph = document.createElement("p"); paragraph.className = "empty-copy";
      paragraph.textContent = constructionContext ? constructionContext.noDetector ? "No detector ran for this task. Faults have not been assessed." : processed ? "No experimental candidate in this frame. Faults may be missed; this is not an inspection pass." : "No construction assessment yet. Selected experiments do not establish reliable inspection." : processed ? "No findings reported for the latest frame." : "Findings appear here when a frame is processed.";
      holder.append(paragraph); return;
    }
    findings.forEach((finding,index) => {
      const row = document.createElement("div"); row.className = "finding";
      const dot = document.createElement("span"); dot.className = "finding-dot";
      const body = document.createElement("div"); body.textContent = finding.label || finding.kind || "Observation";
      const explanation=explanations.find(item=>item.bbox && finding.bbox && item.bbox.join(",")===finding.bbox.join(",")) || explanations[index];
      if(explanation) {
        [["what","Observed"],["where","Location"],["why_it_matters","Why inspect"],["next_check","Next check"],["evidence_basis","Evidence"]].forEach(([key,label])=>{
          if(!explanation[key]) return;
          const p=document.createElement("p"); p.className="finding-explanation";
          p.textContent=`${label}: ${explanation[key]}`; body.append(p);
        });
      }
      const meta = document.createElement("small");
      const parts = [finding.method || "opencv"];
      if (finding.score !== null && finding.score !== undefined) parts.push(`method score ${number(finding.score, 3)}`);
      if (finding.bbox) parts.push(`region ${finding.bbox.join(", ")}`);
      if (finding.task_id) parts.push(`task ${String(finding.task_id).replaceAll("_"," ")}`);
      if (finding.evidence_timestamp_ms !== null && finding.evidence_timestamp_ms !== undefined) parts.push(`video ${videoTime(finding.evidence_timestamp_ms)}`);
      if (finding.assessment_status) parts.push(String(finding.assessment_status).replaceAll("_"," "));
      if (finding.measurements) {
        const p=document.createElement("p"); p.className="finding-explanation";
        p.textContent=`Measurement: ${JSON.stringify(finding.measurements)}`; body.append(p);
      }
      meta.textContent = parts.join(" · "); body.append(meta); row.append(dot, body); holder.append(row);
    });
  }

  function renderState(state) {
    const running = Boolean(state && state.running);
    const showingExisting = Boolean(state && (running || state.last_result || previewUrl));
    const activeProfile = showingExisting ? state.profile : selected;
    const info = profiles[activeProfile] || profiles[selected];
    const result = state && state.last_result;
    const measurements = state && state.metrics || {};
    const storage = state && state.storage || {};
    const source = showingExisting ? state.source : $("source").value;
    const activeTask=activeProfile==="construction" ? guidedTask(showingExisting ? state.config?.profile_options || {} : {inspection_task:guidedTask()}) : "";
    renderGuidedTask();
    toggle("inspection-evidence-panel", Boolean(activeTask));
    toggle("construction-coverage", selected === "construction" || activeProfile === "construction");
    toggle("construction-recordings", selected === "construction" || activeProfile === "construction");
    toggle("construction-capabilities", selected === "construction" || activeProfile === "construction");
    toggle("construction-damage-settings", selected === "construction" && !guidedTask() && $("construction-mode").value === "damage");
    toggle("construction-damage-check-settings", selected === "construction" && !guidedTask() && $("construction-mode").value === "damage");
    toggle("video-mode-setting", $("source").value === "file");
    const video = state?.video || {};
    toggle("video-timeline", Boolean(video.source_fps || video.frame_count));
    text("video-time", `${videoTime(video.position_ms)} / ${videoTime(video.duration_ms)} · ${number(video.source_fps,1)} source FPS · ${String(video.mode || "").replaceAll("_"," ")}`);
    text("video-skips", `Source frames skipped: ${number(measurements.source_skipped || 0)}${Number.isFinite(video.replay_lag_ms) ? " · replay lag " + number(video.replay_lag_ms,0) + " ms" : ""}`);
    $("video-progress").max = video.duration_ms || 1;
    if (video.duration_ms) $("video-progress").value = Math.min(video.duration_ms, Math.max(0,video.position_ms || 0));
    else $("video-progress").removeAttribute("value");
    const synthetic = source === "demo";
    const camera = String(source).startsWith("camera:");
    text("source-badge", synthetic ? "SYNTHETIC DEMO" : camera ? "LIVE CAMERA · LOCAL" : String(source).startsWith("file:") ? "VIDEO REPLAY" : "LOCAL IMAGE / DOCUMENT");
    $("source-badge").className = `badge${synthetic ? "" : camera ? " live" : " replay"}`;
    $("stream-dot").classList.toggle("active", running);
    text("preview-state", running ? `${info.label.toUpperCase()} · ${state.state || "RUNNING"}` : $("preview").classList.contains("hidden") ? "NO ACTIVE STREAM" : `STOPPED · ${info.label.toUpperCase()} · LAST PREVIEW`);
    const constructionPhoto=activeProfile==="construction" && /^(image|images)(:|$)/.test(String(source));
    text("preview-scope", activeTask ? "RAW PREVIEW · inspected evidence below" : constructionPhoto ? "Real photo · selected condition candidates" : state?.description?.mode === "local_trained_surface_model" ? "LOCAL SURFACE MODEL · NARROW DOMAIN" : info.scope);
    $("preview").alt = activeTask ? "Unannotated live source preview; inspected evidence is shown separately below" : "Annotated inspection frame from the local edge runtime";
    const limits = state?.description?.limitations;
    text("profile-limitation", showingExisting && limits ? (Array.isArray(limits) ? limits.join(" ") : String(limits)) : info.limitation);
    if (activeProfile === "factory" && state?.description?.manual_review_only) {
      text("profile-limitation", "Manual review only. Candidates are unverified; no candidate does not approve a unit. Completed units count delivered observations, not physical acceptance or rejection.");
    }
    text("tools-hint", activeTask ? taskCatalog.tasks.find(task=>task.id===activeTask)?.guidance || "Follow the selected task capture guidance." : constructionPhoto ? "Review the original photo and selected condition. Other checks remain unassessed." : info.hint);
    const mismatch = running && state.profile !== selected;
    const previousProfile = showingExisting && !running && state.profile !== selected;
    toggle("setup-notice", mismatch || previousProfile || (running && configurationTouched));
    text("setup-notice", mismatch ? `${profiles[state.profile]?.label || state.profile} is running. Press Start to begin the selected ${profiles[selected].label} profile.` : previousProfile ? `The previous ${info.label} result remains below. Press Start to inspect with ${profiles[selected].label}.` : "Setup changes apply when you press Start inspection again.");
    toggle("calibrate-button", activeProfile === "factory" && state?.description?.mode !== "local_trained_surface_model");
    toggle("capture-button", activeProfile === "freight");
    toggle("trigger-button", activeProfile === "factory" && Boolean(state?.config?.triggered));
    ["stop-button", "reset-button", "calibrate-button", "capture-button", "trigger-button"].forEach((id) => { $(id).disabled = busy || !running; });
    $("start-button").disabled = busy;
    text("recording-label", $("save-evidence").checked ? "ON" : "OFF");
    const config = state?.config || {};
    const previewDisabled = running && config.preview_fps === 0;
    text("preview-placeholder-title", previewDisabled ? "Preview disabled" : "Ready when you are");
    text("preview-placeholder-copy", previewDisabled ? "Inspection continues locally. Set Preview FPS above zero and restart to show images." : "Choose a profile and source, then start inspection. Camera access begins only when selected and started.");
    if (previewDisabled) { toggle("preview", false); toggle("preview-placeholder", true); }
    const dimensions = state?.last_result?.analysis_dimensions || state?.video?.analysis_dimensions || state?.description?.capture_properties?.effective_dimensions;
    text("resolution-label", dimensions?.length === 2 ? `${dimensions[0]} × ${dimensions[1]} ANALYSIS` : state?.last_result?.metrics?.input_width ? `${state.last_result.metrics.input_width} × ${state.last_result.metrics.input_height}` : synthetic ? `${config.width || 960} × ${config.height || 540}` : "SOURCE GEOMETRY");
    const failures = [state?.error, storage.error, state?.media?.errors ? "Media encoding errors: " + state.media.errors : null, state?.media?.evidence_rejected ? "Event image queue rejected " + state.media.evidence_rejected + "; metadata retained" : null].filter(Boolean);
    toggle("error-banner", failures.length > 0);
    text("error-banner", failures.join(" · "));
    text("cloud-label", state?.cloud?.enabled ? "Cloud sync configured" : "Cloud sync off");
    const status = result?.status || "idle";
    const tone = /fault|anomal|reject|error/.test(status) ? "error" : /review|evidence|blur|duplicate|candidate|calibrat|unstable/.test(status) ? "warning" : /accept|clean|ready|complete|pass/.test(status) ? "" : "neutral";
    tag("result-status", status.replaceAll("_", " ").toUpperCase(), tone);
    text("result-title", result?.title || "No inspection yet");
    text("result-guidance", result?.guidance || "Start a source to see findings and the next suggested action.");
    text("result-action", result?.action ? result.action.replaceAll("_", " ") : "Choose a source");
    renderChecklist(result?.domain_state);
    const doc = result?.domain_state;
    const captureGuidance=doc?.capture_guidance;
    text("inspection-capture-guidance", Array.isArray(captureGuidance) ? captureGuidance.join(" ") : captureGuidance || "Keep the selected material clear and steady. Await an inspected frame.");
    text("guided-preview-fps", `Preview ${number(measurements.preview_fps,1)} fps`);
    const inspectionKind=state?.description?.inspection_kind;
    text("guided-inspection-fps", inspectionKind==="guidance" ? `Guidance ${number(measurements.guidance_fps,1)} fps · no detector` : `${inspectionKind==="geometry" ? "Geometry" : "Inspected"} ${number(measurements.inspected_fps ?? measurements.processed_fps,1)} fps`);
    renderEvidenceAge(state);
    const documentContent = doc && (doc.ocr || doc.extracted_fields || doc.ocr_text || doc.text);
    toggle("document-detail", Boolean(documentContent));
    text("document-text", documentContent ? JSON.stringify(documentContent, null, 2) : "");
    const noConstructionDetector=Boolean(activeTask && (inspectionKind==="guidance" || !showingExisting && activeTask!=="tile_installation" && !$("construction-research-model").value));
    if(activeTask) {
      if(!/error/.test(status)) tag("result-status",noConstructionDetector ? "NOT ASSESSED · NO DETECTOR" : "NOT QUALIFIED", "warning");
      if(!result?.findings?.length) text("result-action",noConstructionDetector ? "Capture guidance only" : "Review source · faults may be missed");
    }
    renderFindings(result?.findings || [], Boolean(result), result?.domain_state?.review_findings || [], activeProfile==="construction" ? {noDetector:noConstructionDetector} : null);
    metric("metric-p95", measurements.p95_ms, "ms", 1);
    metric("metric-fps", measurements.processed_fps, "fps", 1);
    metric("metric-processed", measurements.processed, "");
    text("metric-drops", `${number(measurements.dropped)} / ${number(measurements.late)}`);
    text("metric-storage", storage.error ? "ERROR" : storage.written === undefined ? "—" : `${number(storage.written)} written`);
    text("capture-rate", `Capture rate ${number(measurements.capture_fps, 1)} fps`);
    text("captured-count", `Captured ${number(measurements.captured)}`);
    text("latency-note", activeProfile === "freight" && !synthetic ? `Guidance timing · OCR ${doc?.ocr_pending ? "pending" : (doc?.ocr?.ocr_job_ms || doc?.ocr?.ocr_ms) ? number(doc.ocr.ocr_job_ms || doc.ocr.ocr_ms, 0) + " ms/page" : "not yet read"}` : measurements.p95_ms === undefined ? "Awaiting processed frames" : `Deadline ${number(config.deadline_ms)} ms · camera delay unmeasured`);
    $("metric-p95").title = `Host receipt after read to inspection decision; excludes camera exposure, camera/backend buffering and browser rendering. p50 ${number(measurements.p50_ms, 1)} ms.`;
    const factoryQueue = activeProfile === "factory";
    text("queue-note", factoryQueue ? "Bounded FIFO · overflow reported" : "Latest-frame queue · stale frames dropped");
    text("storage-note", `Pending ${number(storage.pending)} · lost ${number(storage.lost_events)}`);
    text("health-queue", number(measurements.queue_depth));
    text("health-age", `${number(measurements.frame_age_ms, 1)} ms`);
    text("health-errors", number(measurements.inspection_errors));
    text("health-missed", number(measurements.missed_units));
    text("health-gaps", number(measurements.coverage_gaps));
    text("health-uptime", measurements.uptime_s === undefined ? "—" : `${number(measurements.uptime_s)} s`);
    text("health-session", state?.session_id || "—");
    $("health-session").title = state?.session_id || "";
    const unhealthy = Boolean(failures.length || Number(storage.lost_events) || Number(measurements.inspection_errors) || Number(measurements.missed_units) || Number(measurements.coverage_gaps));
    const delayed = Number(measurements.late) > 0 || Number(measurements.dropped) > 0;
    tag("health-tag", unhealthy ? "ATTENTION" : running ? delayed ? "CHECK COVERAGE" : "RUNNING" : "IDLE", unhealthy ? "error" : delayed && running ? "warning" : running ? "" : "neutral");
    text("health-note", factoryQueue ? config.triggered ? "Factory uses bounded FIFO. Trigger each unit explicitly; overflow and missed deadlines are reported. Camera exposure and buffering delay are unmeasured." : "Factory uses bounded FIFO in continuous mode; overflow or skipped source frames create visible coverage gaps. Every-item coverage is unverified. Camera delay is unmeasured." : config.video_mode === "every_frame" && String(state?.source || "").startsWith("file:") ? "Sequential source playback decodes without source skipping. The latest-frame inference queue can still drop decoded frames; every-frame inspection is unverified." : "Latest-frame queue: stale observations and source frames skipped for real-time video are reported separately. Every-frame coverage and camera delay are unverified.");
  }

  async function pollState() {
    if (polling) return;
    polling = true;
    lastStatePollStartedAt = performance.now();
    try {
      const state = await request("/api/state");
      if (firstState) {
        firstState = false;
        const restoreGuided=state.profile==="construction" && Boolean(guidedTask(state.config?.profile_options || {}));
        if ((state.running || restoreGuided) && profiles[state.profile]) { selected = state.profile; chooseProfile(selected); }
        if (state.config && (state.running || restoreGuided)) {
          $("input-fps").value = state.config.input_fps || 12;
          $("deadline").value = state.config.deadline_ms || 350;
          $("save-evidence").checked = Boolean(state.config.save_evidence);
          $("triggered").checked = Boolean(state.config.triggered);
          const pathKind = ["file", "image", "images", "pdf"].find(kind => String(state.source).startsWith(`${kind}:`));
          if (pathKind) { $("source").value = pathKind; $("file-path").value = state.source.slice(pathKind.length + 1); }
          else if (["demo", "camera:0"].includes(state.source)) $("source").value = state.source;
          $("profile-options").value = JSON.stringify(state.config.profile_options || {}, null, 2);
          $("source-hold").value = state.config.source_hold_seconds || 30;
          $("preview-fps").value = state.config.preview_fps ?? 6;
          $("video-mode").value = state.config.video_mode || "realtime";
          if (state.profile === "construction") {
            syncGuidedOptions(state.config.profile_options || {});
            $("construction-mode").value = state.config.profile_options?.mode || "ppe";
            requestedDamageVariant=state.config.profile_options?.damage_variant || null;
            if(requestedDamageVariant) $("construction-damage-variant").value=requestedDamageVariant;
            requestedDamageCheck=state.config.profile_options?.damage_classes?.length===1 ? state.config.profile_options.damage_classes[0] : "";
            updateDamageChecks();
            setConstructionStage(state.config.profile_options?.stage || "all");
          }
          toggle("file-field", ["file", "image", "images", "pdf"].includes($("source").value));
        }
      }
      if (lastSession !== null && state.session_id !== lastSession) clearPreview();
      lastSession = state.session_id;
      latest = state;
      latestStateReceivedAt = performance.now();
      text("connection-label", "Edge runtime connected");
      $("connection-dot").className = "tiny-dot";
      renderState(state);
      if(state.running && state.profile==="construction" && guidedTask(state.config?.profile_options || {}) && !document.hidden) void pollInspectionEvidence();
    } catch (error) {
      text("connection-label", "Runtime connection lost");
      $("connection-dot").className = "tiny-dot error";
      text("preview-state", "CONNECTION LOST · PREVIEW MAY BE STALE");
      $("stream-dot").classList.remove("active");
    } finally { polling = false; }
  }

  function clearPreview() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null; $("preview").removeAttribute("src");
    previewFrameKey=null;
    toggle("preview", false); toggle("preview-placeholder", true);
    if(inspectionUrl) URL.revokeObjectURL(inspectionUrl);
    inspectionUrl=null; inspectedKey=null; displayedEvidence=null; $("inspection-evidence").removeAttribute("src");
    delete $("inspection-evidence").dataset.sequence;
    delete $("inspection-evidence").dataset.capturedMonotonic;
    toggle("inspection-evidence",false); text("inspection-evidence-time","Awaiting inspection");
  }

  async function pollInspectionEvidence() {
    const snapshot=latest;
    const result=snapshot?.inspection_result;
    if(inspectionBusy || document.hidden || snapshot?.profile!=="construction" || !guidedTask(snapshot.config?.profile_options || {}) || result?.sequence===undefined) return;
    const session=result.session_id || snapshot.session_id;
    const key=`${session}:${result.sequence}`;
    if(key===inspectedKey) return;
    inspectionBusy=true;
    try {
      const response=await fetch(`/api/inspection.jpg?sequence=${encodeURIComponent(result.sequence)}&session_id=${encodeURIComponent(session)}&t=${Date.now()}`,{cache:"no-store"});
      if(!response.ok) return;
      const blob=await response.blob();
      if(latest?.session_id!==snapshot.session_id || !guidedTask(latest.config?.profile_options || {})) return;
      const url=URL.createObjectURL(blob);
      try {
        const decoded=new Image(); decoded.src=url; await decoded.decode();
      } catch(error) { URL.revokeObjectURL(url); throw error; }
      if(latest?.session_id!==snapshot.session_id || !guidedTask(latest.config?.profile_options || {})) { URL.revokeObjectURL(url); return; }
      if(displayedEvidence?.session_id===session && Number.isFinite(displayedEvidence.sequence) && Number(result.sequence)<=displayedEvidence.sequence) { URL.revokeObjectURL(url); return; }
      $("inspection-evidence").src=url;
      if(inspectionUrl) URL.revokeObjectURL(inspectionUrl);
      inspectionUrl=url; inspectedKey=key; toggle("inspection-evidence",true);
      displayedEvidence={session_id:session,captured_monotonic:result.captured_monotonic,sequence:Number(result.sequence)};
      $("inspection-evidence").dataset.sequence=String(result.sequence);
      $("inspection-evidence").dataset.capturedMonotonic=String(result.captured_monotonic);
      const sourceTime=result.source_time_ms ?? result.domain_state?.evidence_timestamp_ms;
      text("inspection-evidence-time",`Frame ${result.sequence}${sourceTime!==undefined && sourceTime!==null ? " · video "+videoTime(sourceTime) : " · source time unavailable"}`);
      text("inspection-evidence-caption",`Exact inspected frame · ${String(result.domain_state?.assessment_status || result.status || "observation").replaceAll("_"," ")}. The raw preview is shown separately above.`);
      renderEvidenceAge(latest);
    } catch(_) { /* State polling reports connection failures; a mismatched frame is never substituted. */ }
    finally { inspectionBusy=false; }
  }

  async function pollPreview() {
    const restoreStoppedPreview=latest?.last_result && !latest?.running && !previewUrl;
    if (previewBusy || (!latest?.running && !restoreStoppedPreview) || latest?.config?.preview_fps === 0 || document.hidden) return;
    const highRateGuided=latest?.profile==="construction" && guidedTask(latest.config?.profile_options || {});
    if(!highRateGuided && performance.now()-lastPreviewAttempt<165) return;
    lastPreviewAttempt=performance.now();
    previewBusy = true;
    const session = latest.session_id;
    try {
      const response = await fetch(`/api/preview.jpg?t=${Date.now()}`, {cache: "no-store"});
      if (!response.ok) return;
      const frameSequence=response.headers.get("X-Frame-Sequence");
      const frameSession=response.headers.get("X-Session-Id");
      const frameKey=frameSequence!==null && frameSession!==null ? `${frameSession}:${frameSequence}` : null;
      if(frameKey!==null && frameKey===previewFrameKey) return;
      const blob = await response.blob();
      if (latest?.session_id !== session) return;
      const next = URL.createObjectURL(blob);
      $("preview").src = next;
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      previewUrl = next; toggle("preview", true); toggle("preview-placeholder", false);
      previewFrameKey=frameKey;
    } catch (_) { /* state polling communicates network errors */ }
    finally { previewBusy = false; }
  }

  function eventTime(value) {
    if (value === null || value === undefined) return "—";
    const date = new Date(typeof value === "number" ? (value < 1e12 ? value * 1000 : value) : value);
    return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
  }

  async function pollEvents() {
    try {
      const events = await request("/api/events?limit=20");
      if (!Array.isArray(events)) return;
      const body = $("events-body"); body.replaceChildren();
      if (!events.length) {
        const row = document.createElement("tr"), cell = document.createElement("td");
        cell.colSpan = 4; cell.className = "table-empty"; cell.textContent = "No inspection events yet. Accepted evidence and review events appear here.";
        row.append(cell); body.append(row); return;
      }
      events.forEach((event) => {
        const row = document.createElement("tr");
        [eventTime(event.created_at), event.profile || "—", String(event.kind || "event").replaceAll("_", " ")].forEach((value) => { const cell = document.createElement("td"); cell.textContent = value; row.append(cell); });
        const cell = document.createElement("td"), details = document.createElement("details"), summary = document.createElement("summary"), pre = document.createElement("pre");
        summary.textContent = "View record"; pre.textContent = JSON.stringify(event.payload || {}, null, 2);
        details.append(summary, pre); cell.append(details); row.append(cell); body.append(row);
      });
    } catch (_) { /* do not replace useful existing history on transient failure */ }
  }

  async function command(path, body, success) {
    if (busy) return;
    busy = true; renderState(latest);
    try {
      await request(path, body);
      if (success) notify(success);
      await pollState(); await pollEvents();
    } catch (error) { notify(error.message, true); }
    finally { busy = false; renderState(latest); }
  }

  document.querySelectorAll(".profile-button").forEach((button) => button.addEventListener("click", () => chooseProfile(button.dataset.profile)));
  ["source", "file-path", "input-fps", "deadline", "save-evidence", "triggered", "profile-options", "source-hold", "preview-fps", "video-mode", "construction-stage", "construction-mode", "construction-damage-variant", "construction-damage-check", "construction-task", "construction-research-model", "material-roi-left", "material-roi-top", "material-roi-right", "material-roi-bottom", "tile-nominal-joint", "tile-joint-tolerance", "construction-calibration-reference", "construction-expected-fasteners"].forEach((id) => $(id).addEventListener("change", () => {
    configurationTouched = true;
    if(id==="construction-task") { requestedResearchModel=""; $("construction-research-model").value=""; }
    if(id==="construction-research-model") requestedResearchModel=$("construction-research-model").value;
    toggle("preset-evidence-note",false);
    toggle("file-field", ["file", "image", "images", "pdf"].includes($("source").value));
    renderState(latest);
    if (id === "construction-stage") { requestedConstructionStage = $("construction-stage").value; renderConstructionPlan(); }
    if (id === "construction-damage-variant") { requestedDamageVariant=$("construction-damage-variant").value; requestedDamageCheck=""; updateDamageChecks(); }
    if (id === "construction-damage-check") requestedDamageCheck=$("construction-damage-check").value;
    if(id === "construction-task" && guidedTask()) {
      $("preview-fps").value=30; $("input-fps").value=30; $("deadline").value=1000;
      renderGuidedTask();
    }
  }));
  $("start-button").addEventListener("click", async () => {
    let source = $("source").value;
    if (["file", "image", "images", "pdf"].includes(source)) {
      const path = $("file-path").value.trim();
      if (!path) { notify("Enter an absolute path on the edge host.", true); $("file-path").focus(); return; }
      source = `${source}:${path}`;
    }
    const fps = Number($("input-fps").value), deadline = Number($("deadline").value);
    if (!Number.isFinite(fps) || fps < 1 || fps > 30 || !Number.isFinite(deadline) || deadline < 20 || deadline > 5000) {
      notify("Use 1–30 input FPS and a deadline between 20 and 5,000 ms.", true); return;
    }
    let profileOptions;
    try { profileOptions = JSON.parse($("profile-options").value || "{}"); if (!profileOptions || Array.isArray(profileOptions) || typeof profileOptions !== "object") throw new Error(); }
    catch (_) { notify("Profile options must be a JSON object.", true); return; }
    if (selected === "construction" && guidedTask()) {
      try { applyGuidedOptions(profileOptions); }
      catch(error) { notify(error.message,true); return; }
      profileOptions.stage=$("construction-stage").value;
    } else if (selected === "construction") {
      applyGuidedOptions(profileOptions);
      profileOptions.mode = $("construction-mode").value;
      profileOptions.stage = $("construction-stage").value;
      if(profileOptions.mode!=="crack") delete profileOptions.crack_variant;
      if(profileOptions.mode==="damage") {
        const variant=$("construction-damage-variant").value;
        const option=[...$("construction-damage-variant").options].find(item=>item.value===variant && !item.disabled);
        if(!variant || !option) { notify("Choose an installed damage model.",true); return; }
        profileOptions.damage_variant=variant;
        const check=$("construction-damage-check").value;
        if(check && ![...$("construction-damage-check").options].some(item=>item.value===check && !item.disabled)) { notify("Choose a condition supported by this model.",true); return; }
        if(check) profileOptions.damage_classes=[check]; else delete profileOptions.damage_classes;
      } else { delete profileOptions.damage_variant; delete profileOptions.damage_classes; }
    }
    const sourceHold = Number($("source-hold").value);
    if (!Number.isFinite(sourceHold) || sourceHold < 1 || sourceHold > 300) { notify("Use a page hold time between 1 and 300 seconds.", true); return; }
    const previewFps = Number($("preview-fps").value);
    if (!Number.isFinite(previewFps) || previewFps < 0 || previewFps > 30) { notify("Use a preview rate between 0 and 30 FPS.", true); return; }
    await command("/api/start", {profile: selected, source, input_fps: fps, width: 960, height: 540, deadline_ms: deadline, save_evidence: $("save-evidence").checked, triggered: selected === "factory" && $("triggered").checked, profile_options: profileOptions, source_hold_seconds: sourceHold, preview_fps: previewFps, video_mode: $("video-mode").value}, "Inspection started");
    configurationTouched = false;
  });
  $("stop-button").addEventListener("click", () => command("/api/stop", {}, "Inspection stopped"));
  $("construction-show-diagnostics").addEventListener("change", () => {
    renderRecordingOptions(); renderPresetOptions();
  });
  $("construction-load-source").addEventListener("click",()=>{
    const record=recordedVideos.find(item=>item.id===$("construction-recording").value);
    const index=publicPresets.findIndex(preset=>record?.source_preset_label && preset.label===record.source_preset_label && preset.available);
    if(index<0) return;
    $("sample-preset").value=String(index);
    $("sample-preset").dispatchEvent(new Event("change"));
  });
  $("construction-recording").addEventListener("change", async () => {
    const token=++evidenceRequest;
    const record = recordedVideos.find(item=>item.id === $("construction-recording").value);
    const player = $("construction-recording-player"); player.pause();
    renderCaseEvidence(null,record);
    $("construction-recording-play").disabled = !record;
    $("construction-load-source").disabled = !publicPresets.some(preset=>record?.source_preset_label && preset.label===record.source_preset_label && preset.available);
    if (!record) {
      player.removeAttribute("src"); player.load();
      text("construction-recording-note", "Object boxes alone do not establish damage or a construction mistake. Diagnostic footage is available separately.");
      text("construction-recording-credit", ""); return;
    }
    player.src = record.url;
    text("construction-recording-note", `${record.purpose === "defect_evidence" ? "Offline per-frame model recording at original camera timing. " : "DIAGNOSTIC / CONTEXT ONLY: "}${record.note || "Read the evaluation scope before interpreting this recording."}`);
    text("construction-recording-credit", record.attribution);
    [[record.source_url, "Source"], [record.license_url, "License"]].forEach(([value,label]) => {
      try {
        const url = new URL(value); if (url.protocol !== "https:") return;
        const link = document.createElement("a"); link.href = url.href; link.textContent = label; link.target = "_blank"; link.rel = "noopener noreferrer";
        $("construction-recording-credit").append(document.createTextNode(" · "), link);
      } catch (_) { /* absent/invalid attribution links are not executable */ }
    });
    if(record.evidence_available) {
      try {
        const evidence=await request(record.evidence_url);
        if(token===evidenceRequest) renderCaseEvidence(evidence,record);
      } catch(error) {
        if(token===evidenceRequest) text("construction-recording-note",`Annotation evidence could not load: ${error.message}. This recording is not an evaluated inspection case.`);
      }
    }
  });
  $("construction-recording-play").addEventListener("click", async () => {
    const player = $("construction-recording-player");
    if (!player.getAttribute("src")) return;
    try { if (player.paused) await player.play(); else player.pause(); }
    catch (error) { notify(`Recorded video could not play: ${error.message}`, true); }
  });
  ["play", "pause", "ended"].forEach(event => $("construction-recording-player").addEventListener(event, () => {
    text("construction-recording-play", $("construction-recording-player").paused ? "Play recording" : "Pause recording");
  }));
  ["timeupdate","seeked","loadeddata"].forEach(event=>$("construction-recording-player").addEventListener(event,()=>renderCurrentEvidence()));
  [["reset-button", "reset", "Profile reset"], ["capture-button", "capture", "Capture requested; quality checks still apply"], ["calibrate-button", "calibrate", "Reference calibration requested"], ["trigger-button", "trigger", "Unit inspection triggered"]].forEach(([id, action, message]) => {
    $(id).addEventListener("click", () => command(`/api/actions/${action}`, {}, message));
  });
  $("export-button").addEventListener("click", async (event) => {
    event.preventDefault();
    try {
      const response = await fetch("/api/export", {cache: "no-store"});
      if (!response.ok) throw new Error(`Export failed (${response.status})`);
      const url = URL.createObjectURL(await response.blob()), link = document.createElement("a");
      link.href = url; link.download = "inspectflow-export.json"; document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { notify(error.message, true); }
  });
  window.addEventListener("pagehide", () => { if (previewUrl) URL.revokeObjectURL(previewUrl); if(inspectionUrl) URL.revokeObjectURL(inspectionUrl); });
  chooseProfile(selected);
  loadPresets();
  loadConstructionCatalog();
  loadConstructionCapabilities();
  loadConstructionTasks();
  loadRecordedVideos();
  pollState(); pollEvents();
  setInterval(()=>{
    const guidedRunning=latest?.running && latest.profile==="construction" && guidedTask(latest.config?.profile_options || {}) && !document.hidden;
    if(performance.now()-lastStatePollStartedAt >= (guidedRunning ? 200 : 500)-5) void pollState();
  },100);
  setInterval(pollPreview, 33);
  setInterval(pollInspectionEvidence, 250);
  setInterval(()=>{if(displayedEvidence && !document.hidden)renderEvidenceAge(latest);},100);
  setInterval(pollEvents, 2000);
})();

