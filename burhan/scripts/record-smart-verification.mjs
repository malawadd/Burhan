import {readFileSync,writeFileSync,mkdirSync,readdirSync,existsSync,copyFileSync} from 'node:fs';
import {createHash} from 'node:crypto';import {spawnSync} from 'node:child_process';import path from 'node:path';
const root=path.resolve(import.meta.dirname,'..');process.chdir(root);
function node(args){const result=spawnSync(process.execPath,args,{encoding:'utf8'});if(result.status!==0)throw Error(result.stdout+result.stderr);return result.stdout;}
const tests=[...readdirSync('tests').filter(n=>n.endsWith('.test.mjs')).map(n=>'tests/'+n),...readdirSync('src/engine').filter(n=>n.endsWith('.test.mjs')).map(n=>'src/engine/'+n)];
const types=node(['node_modules/typescript/bin/tsc','--noEmit']);
const testLog=node(['--test',...tests]);
const assets=node(['scripts/verify-assets.mjs']);
const alignment=node(['scripts/check-native-alignment.mjs']);
const metadata=JSON.parse(readFileSync('delivery/build-metadata.json','utf8'));
const apk=readFileSync('delivery/'+metadata.apk),sha256=createHash('sha256').update(apk).digest('hex');
if(sha256!==metadata.sha256)throw Error('APK hash differs from build metadata');
const lint=readFileSync('android/app/build/reports/lint-results-debug.xml','utf8');
const errors=(lint.match(/severity="Error"/g)||[]).length,warnings=(lint.match(/severity="Warning"/g)||[]).length;
if(errors)throw Error('Android lint failed');
const nativeLog=readFileSync('.tools/smart-android-checks.log','utf8');
if(!nativeLog.includes('BUILD SUCCESSFUL'))throw Error('Android lint/instrumentation compilation has not completed');
const parity=readFileSync('.tools/smart-parity.log','utf8');
if(!parity.includes('surface512')||!parity.includes('wasmRawOutputMaximumAbsoluteError'))throw Error('Four-model real WASM verification missing');
const backendLog=readFileSync('.tools/smart-backend-tests.log','utf8');
if(!/\d+ passed/.test(backendLog)||/\d+ failed/.test(backendLog))throw Error('Backend tests did not pass');
const artifact=name=>{const file=path.join(root,'../output/smart-smoke',name);return existsSync(file)?JSON.parse(readFileSync(file,'utf8')):{status:'not run'};};
const real=artifact('backend-real.json'),browser=artifact('browser-verification.json'),switching=artifact('provider-switching.json');
const selection=artifact('v12-codex-selection.json'),v12Browser=artifact('v12-browser-verification.json'),tunnel=artifact('v12-tunnel.json');
const unitDir='android/app/build/test-results/testDebugUnitTest';
const unitXml=existsSync(unitDir)?readdirSync(unitDir).filter(n=>n.endsWith('.xml')).map(n=>readFileSync(path.join(unitDir,n),'utf8')).join('\n'):'';
const uiTestFile=path.join(unitDir,'TEST-com.burhan.site.InspectionCameraUiTest.xml');
if(!existsSync(uiTestFile))throw Error('Native camera layout test results missing');
const uiXml=readFileSync(uiTestFile,'utf8');
if(!/failures="0"/.test(uiXml)||! /errors="0"/.test(uiXml))throw Error('Native camera layout tests did not pass');
const uiScreenshots=path.join(root,'../output/native-ui');
const screenshotNames=existsSync(uiScreenshots)?readdirSync(uiScreenshots).filter(n=>n.endsWith('.png')):[];
if(screenshotNames.length<6)throw Error('Actual native View layout renders missing');
const nativeCameraUi={status:'passed',version:metadata.version,verifiedAt:new Date().toISOString(),framework:'Robolectric 4.16.1 / SDK35 native graphics / actual Android Material Views',
 tests:Number(uiXml.match(/tests="(\d+)"/)?.[1]??0),cases:[...uiXml.matchAll(/<testcase name="([^"]+)"/g)].map(m=>m[1]),
 screenshots:screenshotNames,scope:'Camera UI rendered with a blank preview fixture. No camera, model, network request or physical device was started by these layout tests.',
 coverage:['320x640 and 360x800 portrait; 640x360 landscape','English and Arabic; normal, 160% and 200% text','Persistent capture/import/review bounds, target sizes, label clipping and word boundaries','Storage warning, readiness, saved count and independent camera/guide state','Full guidance, manual mode, offline installed detector choices and stale alternative removal','Subtle hint tap/swipe behavior without a new card; selected state preserved on dismissal'],
 phoneValidation:'Pending on the user\'s phone; layout renders do not validate hardware camera, focus or thermal performance.'};
const verification={app:'Burhan',version:metadata.version,verifiedAt:new Date().toISOString(),apk:metadata,
 automated:{typescript:'passed',domainBackupDecoderSmartTests:Number(testLog.match(/(?:tests\s+|# tests )(\d+)/)?.[1]??0),backendTests:Number(backendLog.match(/(\d+) passed/)?.[1]),modelHashes:4,
   numericalParity:'Exact Python-output decoder parity plus real execution of all four WASM models; browser resize differences remain disclosed.',
   android:{build:'assembleDebug passed',unitTests:(unitXml.match(/<testcase /g)||[]).length,nativeCameraUi,lint:{errors,warnings},instrumentation:'assembleDebugAndroidTest compiled; not executed on a phone',elf:'All six ARM64 libraries have 16KB LOAD alignment',signature:'APK v2 verified; zipalign16KB passed'}},
 realProvider:{provider:real.provider??'not run',endpoint:real.endpoint??'',guidanceChanged:real.guidanceChanged??false,turns:(real.turns??[]).map(t=>({status:t.status,scene:t.result?.scene,guidance:t.result?.guidance,durationSeconds:t.durationSeconds})),quality:real.qualityRespondedTo??[],scope:real.scope??''},
 browser,switching,v12:{selection,browser:v12Browser,tunnel,hardwareCameraValidation:'not run; physical phone required',nativeLogicCoverage:['Scene-scoped completion and dismissal with minimum60-second cooldown','Every camera preference independently persisted on and off','Room Retake atomic replacement, rollback and inspection ownership; exact-ID removal','Normalized markup geometry and immutable detector baseline'],appLogicCoverage:['Concurrent history/capture persistence and stale autosave protection','Old/new portable backup round trips and immutable report evidence','Task-bound guidance and confirmation-aware proposal normalization']},limitations:['User reported the previous app worked on their phone. The updated native build has not been run on a physical device by this agent.',
 'Live AWS model execution/deployment needs account credentials and model access; an SDK contract test is not a live AWS result.',
 'Scene suggestions and existing defect models remain experimental; no all-defect, structural-safety or code-compliance qualification.',
 'The workflow photo test and synthetic numerical fixture do not establish construction-scene or damage accuracy.'],physicalDeviceChecklist:'docs/device-checklist.md'};
mkdirSync('delivery',{recursive:true});
writeFileSync('delivery/verification.json',JSON.stringify(verification,null,2)+'\n');
writeFileSync('delivery/checks.log',[types,testLog,assets,alignment,backendLog,nativeLog,parity].join('\n'));
copyFileSync('android/app/build/reports/lint-results-debug.txt','delivery/android-lint.txt');
mkdirSync('delivery/evaluation',{recursive:true});
mkdirSync('delivery/evaluation/native-ui',{recursive:true});
writeFileSync('delivery/evaluation/native-ui/verification.json',JSON.stringify(nativeCameraUi,null,2)+'\n');
for(const name of screenshotNames)copyFileSync(path.join(uiScreenshots,name),path.join('delivery/evaluation/native-ui',name));
for(const name of ['backend-real.json','browser-verification.json','provider-switching.json','v12-codex-selection.json','v12-browser-verification.json','v12-tunnel.json','v12-browser-codex-trace.json']){const file=path.join(root,'../output/smart-smoke',name);if(existsSync(file))copyFileSync(file,'delivery/evaluation/'+name);}
console.log(`Recorded ${verification.automated.domainBackupDecoderSmartTests} app tests, ${verification.automated.backendTests} backend tests, ${verification.automated.android.unitTests} native unit test(s); lint ${errors} errors/${warnings} warnings. APK ${sha256}`);

for(const name of readdirSync(path.join(root,'../output/smart-smoke')).filter(n=>/^v12-browser-.*\.png$/.test(n)))copyFileSync(path.join(root,'../output/smart-smoke',name),path.join('delivery/evaluation',name));
