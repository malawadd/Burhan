import type {ResearchInputs,Language} from './types';

export function readMeasurements(data:FormData):ResearchInputs{
 const result:ResearchInputs={};
 for(const key of ['nominalJointMm','toleranceMm','measuredJointMm'] as const){const raw=String(data.get(key)??'').trim();if(!raw)continue;const value=Number(raw);if(!Number.isFinite(value)||value<0||value>1000000)throw Error('Invalid physical measurement');result[key]=value;}
 const notes=String(data.get('measurementNotes')??'').trim();if(notes)result.notes=notes.slice(0,2000);
 return result;
}
export function ResearchMeasurements({language,initial={}}:{language:Language;initial?:ResearchInputs}){
 const ar=language==='ar',text=(en:string,arabic:string)=>ar?arabic:en;
 return <fieldset><legend>{text('Physical measurements (optional)','القياسات الفعلية (اختياري)')}</legend><p className="muted">{text('Record readings from your instrument and the approved specification. These are entered values, not camera measurements or automatic acceptance.','سجّل قراءات جهاز القياس والمواصفات المعتمدة. هذه قيم مدخلة وليست قياسات بالكاميرا أو قبولاً آلياً.')}</p>
 {([['nominalJointMm','Specified joint width (mm)','عرض الفاصل المحدد (مم)'],['toleranceMm','Specified tolerance (mm)','السماحية المحددة (مم)'],['measuredJointMm','Measured width (mm)','العرض المقاس (مم)']] as const).map(([key,en,arabic])=><label key={key}>{text(en,arabic)}<input type="number" name={key} min="0" max="1000000" step="any" defaultValue={initial[key]??''}/></label>)}
 <label>{text('Instrument / reference / notes','الجهاز / المرجع / ملاحظات')}<textarea name="measurementNotes" maxLength={2000} defaultValue={initial.notes??''}/></label></fieldset>;
}
