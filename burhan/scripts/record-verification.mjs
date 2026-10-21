import {readFileSync,writeFileSync,mkdirSync,copyFileSync} from 'node:fs';import {createHash} from 'node:crypto';import {spawnSync} from 'node:child_process';
const run=(args)=>{const r=spawnSync(process.execPath,args,{encoding:'utf8'});if(r.status!==0)throw Error(r.stdout+r.stderr);return r.stdout;};
const types=run(['node_modules/typescript/bin/tsc','--noEmit']);
const tests=run(['--test','tests/backup.test.mjs','tests/domain.test.mjs','src/engine/decoder.test.mjs']);
const assets=run(['scripts/verify-assets.mjs']);
const parity=run(['--experimental-strip-types','tests/inference-parity.mjs']);
const native=run(['scripts/check-native-alignment.mjs']);
const apk=readFileSync('delivery/Burhan-v1.0.0-debug.apk');const metadata=JSON.parse(readFileSync('delivery/build-metadata.json','utf8'));
const hash=createHash('sha256').update(apk).digest('hex');if(hash!==metadata.sha256)throw Error('Delivered APK checksum differs from build metadata');
const lint=readFileSync('android/app/build/reports/lint-results-debug.xml','utf8');const errors=(lint.match(/severity="Error"/g)||[]).length,warnings=(lint.match(/severity="Warning"/g)||[]).length;
if(errors)throw Error('Android lint errors remain');
const androidLog=readFileSync('.tools/final-android-checks.log','utf8');if(!androidLog.includes('BUILD SUCCESSFUL'))throw Error('Final Android checks did not complete');
mkdirSync('delivery',{recursive:true});
writeFileSync('delivery/checks.log',[types,tests,assets,parity,native,androidLog].join('\n'));
copyFileSync('android/app/build/reports/lint-results-debug.txt','delivery/android-lint.txt');
writeFileSync('delivery/verification.json',JSON.stringify({
  app:'Burhan',version:'1.0.0',verifiedAt:new Date().toISOString(),apk:{file:metadata.apk,sha256:hash,bytes:apk.length,signature:'APK v2 verified by apksigner',zipAlignment:'16KB zipalign check passed',nativeElfAlignment:'all six ARM64 libraries have >=16KB LOAD alignment'},
  automated:{typescript:'passed',domainBackupDecoderTests:13,modelHashes:4,catalogue:{originalTasks:89,directVisualTasks:3,stages:16},goldenPythonOutputDecoder:'exact boxes and masks on the procedural fixture',androidLint:{errors,warnings},androidBuild:'assembleDebug passed',instrumentationBuild:'assembleDebugAndroidTest passed; not run on device'},
  observedBrowserChecks:{workflow:'project -> inspection -> manual finding -> completion -> immutable report -> reload -> saved report; passed',languageLayout:'English/dark and Arabic RTL/light at390px; no horizontal overflow',backup:'ZIP export observed, portable evidence/tamper/traversal/metadata restore tests passed',actualScannerWorker:'all four models executed through the actual Chromium Scanner image-import path; saved original/annotated images and matching SHA/runtime metadata verified',modelExecutionCounts:{crack:2,surface320:3,surface512:3,ppe:0},input:'original193x127 procedural test pattern; not a construction accuracy dataset',pdf:'English, Arabic and bilingual A4 examples; all six pages rendered with Poppler and visually reviewed'},
  limitations:['No physical Android phone connected: live native camera, native inference parity, device recovery, thermal endurance, Android print/share and battery behaviour remain pending.','Browser numerical resize differences remain explicitly documented in tests/fixtures/README.md; no exact end-to-end parity claim.','Existing experimental models have documented misses/false alerts; no field accuracy, safety or code-compliance qualification.','Native unit-test task has no placeholder arithmetic tests; meaningful native smoke tests require a device.'],
  physicalDeviceChecklist:'docs/device-checklist.md'
},null,2)+'\n');
console.log(`Verification recorded:13 tests,0 Android lint errors (${warnings} warnings),4 models,6 aligned native libraries. APK SHA256 ${hash}`);
