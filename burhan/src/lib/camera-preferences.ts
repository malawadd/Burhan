import {Capacitor} from '@capacitor/core';
import {BurhanNative} from '../native';
import definitionsSource from '../../public/camera-settings.json';
export type CameraPreferenceKey='aiSuggestionHints'|'volumeButtonCapture'|'pinchToZoom'|'quickCaptureReview'|'undoLastCapture'|'keepScreenAwake'|'doubleTapProtection'|'captureHaptics'|'rememberLastZoom'|'liveDetectionOverlays'|'focusLock'|'framingGrid'|'cameraLevel'|'higherDetailPhotos'|'photoMarkup'|'individualPhotoSharing';
export type CameraPreferences=Record<CameraPreferenceKey,boolean>;
export const cameraDefaults=Object.fromEntries(definitionsSource.settings.map(d=>[d.key,d.default])) as CameraPreferences;
export interface CameraSettingDefinition {key:CameraPreferenceKey;default:boolean;label:{en:string;ar:string};description:{en:string;ar:string};nativeOnly:boolean;group:'capture'|'assistance'|'framing'|'review'}
const key='burhan-camera-preferences';
export async function getCameraPreferences():Promise<CameraPreferences>{const raw=Capacitor.isNativePlatform()?(await BurhanNative.getCameraPreferences()).preferences:JSON.parse(localStorage.getItem(key)||'{}');return Object.fromEntries(Object.entries(cameraDefaults).map(([k,v])=>[k,typeof raw?.[k]==='boolean'?raw[k]:v])) as CameraPreferences;}
export async function setCameraPreferences(preferences:CameraPreferences){if(Capacitor.isNativePlatform())await BurhanNative.setCameraPreferences({preferences});else localStorage.setItem(key,JSON.stringify(preferences));}
export async function cameraDefinitions(){return definitionsSource.settings as CameraSettingDefinition[];}
