"""Read only author CUBIT/CUPID release metadata; never execute downloaded code."""
from pathlib import Path
import hashlib
import html
import json
import re
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'datasets/construction/finishes-expansion/source-records'


def get(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def main():
    from pip._vendor import truststore
    truststore.inject_into_ssl()
    OUT.mkdir(parents=True, exist_ok=True)
    snapshots = {}
    for name in ['CUBIT', 'CUPID-defect-detector']:
        base = f'https://api.github.com/repos/BenyunZhao/{name}'
        repo = json.loads(get(base))
        commit = json.loads(get(base + '/commits/' + repo['default_branch']))['sha']
        tree = json.loads(get(base + '/git/trees/' + commit + '?recursive=1'))
        readme = get(f'https://raw.githubusercontent.com/BenyunZhao/{name}/{commit}/README.md')
        license_bytes = get(f'https://raw.githubusercontent.com/BenyunZhao/{name}/{commit}/LICENSE')
        (OUT / (name + '-README.md')).write_bytes(readme)
        (OUT / (name + '-LICENSE.txt')).write_bytes(license_bytes)
        snapshots[name] = {'commit': commit, 'source': f'https://github.com/BenyunZhao/{name}/tree/{commit}',
                           'readme_sha256': hashlib.sha256(readme).hexdigest(), 'license_sha256': hashlib.sha256(license_bytes).hexdigest(),
                           'weight_members': [r for r in tree['tree'] if Path(r['path']).suffix in {'.onnx','.pt','.pth','.safetensors'}],
                           'author_links': re.findall(r'https://(?:drive.google.com|pan.baidu.com)[^\s)\"<>]+', readme.decode())}
    folder = 'https://drive.google.com/drive/folders/1DOCF8E7itN2Qve-rFjDcPRAS3wbwGClL?usp=sharing'
    document = get(folder).decode()
    labels = [html.unescape(s) for s in re.findall(r'aria-label="([^"]+)"', document)]
    snapshots['author_folder'] = {'url': folder, 'visible_file_labels': [s for s in labels if any(k in s.lower() for k in ['zip','cubit','weight','train','test','val','model'])],
                                  'visible_file_ids': re.findall(r'data-id="([^"]+)"', document)[:40]}
    video = 'https://drive.google.com/file/d/12u3hCvJKBQV5c3Z0kX0gaJrAxbcNmv2O/view'
    document = get(video).decode()
    snapshots['author_video'] = {'url': video, 'title': html.unescape(re.search(r'<title>(.*?)</title>', document).group(1)),
                                'media_type': 'Author demonstration, not a verified raw labelled sequence',
                                'local_download': False, 'video_license': 'Not independently declared/verified; dataset CC BY4 does not automatically license every demonstration asset',
                                'ground_truth_crosswalk': 'No downloadable original-frame/timecode annotation mapping verified'}
    target = OUT / 'cubit-author-release-audit.json'
    payload = (json.dumps(snapshots, indent=2) + '\n').encode()
    target.write_bytes(payload)
    print(json.dumps({'path': str(target), 'sha256': hashlib.sha256(payload).hexdigest(), 'audit': snapshots}, indent=2))


if __name__ == '__main__':
    main()
