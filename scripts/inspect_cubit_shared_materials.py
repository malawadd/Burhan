"""Read public author-linked shared-resource folder names; no account or weights load."""
from html.parser import HTMLParser
from pathlib import Path
import json,re
from urllib.request import urlopen

class Text(HTMLParser):
    def __init__(self):super().__init__();self.hide=0;self.data=[]
    def handle_starttag(self,tag,attrs):
        if tag in {'script','style'}:self.hide+=1
    def handle_endtag(self,tag):
        if tag in {'script','style'}:self.hide=max(0,self.hide-1)
    def handle_data(self,data):
        if not self.hide and data.strip():self.data.append(data.strip())


def main():
    urls=['https://drive.google.com/drive/folders/1smAGS7WpTrNQPt0enS1FqQt9n4DC9bKu?usp=sharing',
          'https://drive.google.com/drive/folders/1-20JAwuyyd1-Fbpa8f8jHUwtI5B0Frx5?usp=sharing',
          'https://drive.google.com/drive/folders/12U8bT4dwBGC6k-8m7u1iTy7MtWobE_71',
          'https://drive.google.com/drive/folders/1FXhk2F6vmsY-2BfvxBoyNgdzoc3B8J1Q']
    records=[]
    for url in urls:
        with urlopen(url,timeout=60) as response:document=response.read().decode()
        parser=Text();parser.feed(document)
        records.append({'url':url,'title':re.search(r'<title>(.*?)</title>',document).group(1),
                        'visible_text':parser.data,'file_ids':sorted(set(re.findall(r'data-id="([^\"]+)"',document))-{'_gd'})})
    out=Path(__file__).resolve().parents[1]/'datasets/construction/finishes-expansion/source-records/cubit-shared-materials-audit.json'
    out.write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
