"""Download tiny English PP-OCRv4 ONNX exports from their official RapidAI registry."""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
FILES={
 'ch_PP-OCRv4_det_mobile.onnx': ('https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/onnx/PP-OCRv4/det/ch_PP-OCRv4_det_mobile.onnx','d2a7720d45a54257208b1e13e36a8479894cb74155a5efe29462512d42f49da9'),
 'en_PP-OCRv4_rec_mobile.onnx': ('https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.0/onnx/PP-OCRv4/rec/en_PP-OCRv4_rec_mobile.onnx','e8770c967605983d1570cdf5352041dfb68fa0c21664f49f47b155abd3e0e318'),
}

def main():
 out=ROOT/'models/freight';out.mkdir(parents=True,exist_ok=True)
 report={'model':'PP-OCRv4 English mobile; RapidAI ONNX exports from PaddleOCR','source_registry':'rapidocr 3.9.0/default_models.yaml','license':'Apache-2.0','files':[]}
 for name,(url,sha) in FILES.items():
  data=urllib.request.urlopen(url,timeout=60).read()
  if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Model SHA256 mismatch: '+name)
  (out/name).write_bytes(data)
  report['files'].append({'name':name,'bytes':len(data),'sha256':sha,'url':url})
 for name,url in [('PaddleOCR-LICENSE.txt','https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/LICENSE'),('RapidOCR-LICENSE.txt','https://raw.githubusercontent.com/RapidAI/RapidOCR/main/LICENSE')]:
  (out/name).write_bytes(urllib.request.urlopen(url,timeout=30).read())
 (out/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
