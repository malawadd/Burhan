"""Build the portable HTML and designer ZIP from existing Burhan product assets.

No dependencies or network access are required. Run: python design/build-guide.py
The app and APK are not modified by this builder.
"""
from pathlib import Path
import base64
import hashlib
import json
import os
import re
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
APP = HERE.parent
ASSETS = HERE / 'assets'
ASSETS.mkdir(exist_ok=True)


def copy_if_changed(source, target):
    # Windows may keep a mapped font file open. Rebuilds need not rewrite identical bytes.
    if not target.exists() or source.read_bytes() != target.read_bytes():
        shutil.copy2(source, target)


def svg_header(width, height, title):
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><title>{title}</title>'


def mark(fill='#ffd43b', x=0, y=0, scale=1):
    return f'<g transform="translate({x} {y}) scale({scale})"><g transform="translate(0 10) skewY(-15)" fill="{fill}"><rect x="0" y="14" width="8" height="22"/><rect x="12" y="0" width="8" height="36"/><rect x="24" y="8" width="8" height="28"/></g></g>'


def lockup(reverse=False):
    bg = '#ffd43b' if reverse else '#111315'
    ink = '#111315' if reverse else '#f4f2ea'
    accent = '#111315' if reverse else '#ffd43b'
    return (svg_header(650, 160, 'Burhan editable brand lockup') + f'<rect width="650" height="160" fill="{bg}"/>'
            + mark(accent, 30, 24, 2.4)
            + f'<g fill="{ink}" font-family="Noto Sans,Arial,sans-serif"><text x="144" y="88" font-size="67" font-weight="850" letter-spacing="-2.7">BURHAN</text><text x="146" y="119" font-size="17" letter-spacing="1.5">BUILT ON EVIDENCE</text><text x="608" y="86" text-anchor="end" font-family="Noto Sans Arabic,Arial,sans-serif" font-size="34">برهان</text></g></svg>')


def drawing(x, y, width, height):
    # Abstract graphic, never an invented detector result or UI screenshot.
    result = f'<g transform="translate({x} {y})"><rect width="{width}" height="{height}" fill="#191c1f" stroke="#34383d" stroke-width="2"/>'
    for i in range(1, 9):
        result += f'<path d="M{width*i/9:.1f} 0V{height}M0 {height*i/9:.1f}H{width}" stroke="#34383d" opacity=".6"/>'
    result += f'<path d="M{width*.17:.1f} {height*.8:.1f}V{height*.37:.1f}L{width*.48:.1f} {height*.16:.1f}L{width*.8:.1f} {height*.34:.1f}V{height*.78:.1f}L{width*.48:.1f} {height*.9:.1f}Z M{width*.48:.1f} {height*.16:.1f}V{height*.9:.1f} M{width*.17:.1f} {height*.37:.1f}L{width*.48:.1f} {height*.56:.1f}L{width*.8:.1f} {height*.34:.1f}" fill="none" stroke="#ffd43b" stroke-width="3"/>'
    result += f'<path d="M0 22V0H22 M{width-22} 0H{width}V22 M0 {height-22}V{height}H22 M{width-22} {height}H{width}V{height-22}" stroke="#ffd43b" stroke-width="5" fill="none"/></g>'
    return result


def poster(fmt):
    w, h = {'portrait': (1080, 1920), 'square': (1080, 1080), 'wide': (1920, 1080)}[fmt]
    result = svg_header(w, h, f'Burhan suggested {fmt} content canvas')
    result += f'<rect width="{w}" height="{h}" fill="#111315"/>'
    left = 154 if fmt == 'wide' else 86
    result += mark('#ffd43b', left, 84, 2.1)
    result += f'<text x="{left+108}" y="152" fill="#f4f2ea" font-family="Noto Sans,Arial,sans-serif" font-weight="850" font-size="64" letter-spacing="-2">BURHAN</text>'
    result += f'<text x="{w-left}" y="145" fill="#ffd43b" font-family="Noto Sans Arabic,Arial,sans-serif" font-size="38" text-anchor="end">برهان</text>'
    if fmt == 'portrait':
        result += '<g font-family="Noto Sans,Arial,sans-serif" font-weight="850" letter-spacing="-5"><text x="86" y="410" fill="#f4f2ea" font-size="100">BUILT ON</text><text x="86" y="550" fill="#ffd43b" font-size="128">EVIDENCE.</text></g>'
        result += '<text x="86" y="646" fill="#a9adb2" font-family="Noto Sans,Arial,sans-serif" font-size="27">CONSTRUCTION INSPECTION / HUMAN REVIEW</text>'
        result += drawing(86, 748, 908, 646)
        result += '<rect y="1510" width="1080" height="410" fill="#ffd43b"/><g fill="#111315" font-family="Noto Sans,Arial,sans-serif"><text x="86" y="1620" font-size="48" font-weight="750">Inspect with context.</text><text x="86" y="1684" font-size="48" font-weight="750">Keep the evidence.</text><text x="86" y="1756" font-size="23">Offline detection · Optional online guidance</text></g>'
    elif fmt == 'square':
        result += '<g font-family="Noto Sans,Arial,sans-serif" font-weight="850" letter-spacing="-4"><text x="86" y="350" fill="#f4f2ea" font-size="100">BUILT ON</text><text x="86" y="474" fill="#ffd43b" font-size="117">EVIDENCE.</text></g>'
        result += drawing(682, 562, 312, 272)
        result += '<g fill="#f4f2ea" font-family="Noto Sans,Arial,sans-serif" font-size="38" font-weight="750"><text x="86" y="630">INSPECT.</text><text x="86" y="697">CAPTURE.</text><text x="86" y="764">REVIEW.</text></g><rect y="890" width="1080" height="190" fill="#ffd43b"/><text x="86" y="960" fill="#111315" font-family="Noto Sans,Arial,sans-serif" font-size="27" font-weight="700">From observation to inspection record.</text>'
    else:
        result += '<g font-family="Noto Sans,Arial,sans-serif" font-weight="850" letter-spacing="-7"><text x="154" y="397" fill="#f4f2ea" font-size="152">BUILT ON</text><text x="154" y="578" fill="#ffd43b" font-size="160">EVIDENCE.</text></g>'
        result += drawing(1370, 252, 396, 432)
        result += '<rect y="810" width="1920" height="270" fill="#ffd43b"/><g fill="#111315" font-family="Noto Sans,Arial,sans-serif"><text x="154" y="914" font-size="47" font-weight="750">Inspect with context. Keep the evidence.</text><text x="154" y="979" font-size="25">Offline detection · Optional online guidance · Human-reviewed findings</text></g>'
    return result + '</svg>'


source_css = (APP / 'src/styles.css').read_text(encoding='utf-8')
raw_tokens = dict(re.findall(r'--([\w-]+):([^;{}]+)', source_css.split(':root[data-theme=light]')[0]))
tokens = {
    'product': 'Burhan', 'referenceVersion': '1.3.0',
    'workspaceColors': raw_tokens,
    'nativeCameraColors': {'background': '#111315', 'panel': '#1d2024', 'text': '#f4f2ea', 'secondary': '#b9bfc6', 'primary': '#ffd43b'},
    'geometry': {'workspaceRadiusPx': 3, 'nativeButtonRadiusDp': 5, 'browserSheetRadiusPx': 18, 'webPrimaryShadow': '3px 3px 0 #615224', 'cameraTouchTargetDp': 48},
    'motion': {'nativeHintMs': 160, 'nativeHintSlideDp': 6, 'browserHintMs': 180, 'hintAutoTimeout': False, 'respectReducedMotion': True},
    'contentSuggestions': {'workingSpacingGridPx': 4, 'contentMarginPercent': 8, 'portrait': [1080, 1920], 'square': [1080, 1080], 'wide': [1920, 1080]},
    'note': 'Product values are source references. Content margins and canvas layouts are creator suggestions, not platform safe-area requirements.'
}
copy_text = '''BURHAN / برهان
CREATOR COPY STARTER — product reference 1.3.0

EXISTING BRAND LINE
Built on evidence.

EXISTING PRODUCT HEADLINE
See clearly. Build confidently.
رؤية أوضح. بناء بثقة.

SUGGESTED ARABIC CONTENT LINE
كل فحص يبدأ بدليل.

PRODUCT DESCRIPTION
Burhan helps inspectors choose relevant checks, collect visual evidence, review observations and keep a clear inspection record. Supported detection runs on device; optional smart guidance uses a paired online backend.

وصف مقترح
يساعد برهان الفاحص على اختيار بنود الفحص المناسبة وجمع الأدلة البصرية ومراجعة الملاحظات وتوثيقها في سجل واضح. يعمل الكشف المدعوم على الجهاز، بينما يستخدم المساعد الذكي الاختياري خادماً متصلاً بالإنترنت.

USEFUL MICROCOPY
Explore / استكشاف
Capture evidence / التقاط دليل
Review findings / مراجعة الملاحظات
Generate report / إنشاء تقرير
Human review required / المراجعة البشرية مطلوبة

EXPANDED INSPECTIONS — OPTIONAL IN 1.3
Expanded inspection models / نماذج الفحص الموسعة
Off by default / مغلق افتراضياً
Walls & finishes / الجدران والتشطيبات
Brackets & railings / الحوامل والدرابزين
Tile installation / تركيب البلاط
Experimental candidates / ملاحظات مرشحة تجريبية
5 model-backed checks + 25 guided manual checks. Three bundled models work locally when enabled. The 30 check names do not mean 30 automatic detectors.
٥ فحوصات بنماذج تجريبية و٢٥ فحصاً يدوياً موجهاً. تعمل النماذج الثلاثة المضمنة محلياً عند التفعيل. لا تعني أسماء الفحوصات الثلاثين وجود ثلاثين كاشفاً آلياً.
Stain appearance is not a moisture measurement. Mold-like appearance is not identification. Fastener-absence appearance is not proof that required hardware is missing. Tile measurements are entered physical readings.

ACCURACY OF THE MESSAGE
AI proposes. The inspector chooses.
A model observation is not a confirmed finding.
Empty detection is not an automatic pass.
A saved inspection report is not an engineering certification.
Avoid promising every-defect detection, invented accuracy percentages or a fully offline online assistant.
'''

asset_data = {}


def asset(key, filename, text, mime):
    (ASSETS / filename).write_text(text, encoding='utf-8')
    asset_data[key] = {'filename': filename, 'content': text, 'mime': mime}


asset('mark', 'Burhan-mark.svg', svg_header(48, 58, 'Burhan three-bar mark') + mark('#ffd43b', 7, 4) + '</svg>', 'image/svg+xml')
asset('logo-dark', 'Burhan-lockup-dark.svg', lockup(), 'image/svg+xml')
asset('logo-yellow', 'Burhan-lockup-yellow.svg', lockup(True), 'image/svg+xml')
asset('tokens', 'Burhan-design-tokens.json', json.dumps(tokens, indent=2, ensure_ascii=False), 'application/json')
asset('copy', 'Burhan-copy-starter-en-ar.txt', copy_text, 'text/plain;charset=utf-8')
asset('credits', 'Burhan-reference-credits.txt', 'Expanded camera reference: PHELE retained TRAIN photograph train_101.jpg, PHELE dataset https://data.mendeley.com/datasets/yzsxcvc8dp/4, CC BY 4.0 https://creativecommons.org/licenses/by/4.0/. Imported diagnostic image, not independent model qualification. Native references are rendered UI fixtures with blank camera previews. Tile measurement screenshot contains illustrative user-entered numbers, not measured ground truth. Other content canvases are original vector creator directions.', 'text/plain;charset=utf-8')
for fmt in ('portrait', 'square', 'wide'):
    asset('poster-' + fmt, 'Burhan-content-' + fmt + '.svg', poster(fmt), 'image/svg+xml')

screens = {
    'EXPANDED_OFF': APP.parent / 'output/playwright/v13-settings-off.png',
    'EXPANDED_ON': APP.parent / 'output/playwright/v13-settings-on.png',
    'EXPANDED_CAMERA': APP.parent / 'output/playwright/v13-expanded-camera.png',
    'EXPANDED_REVIEW': APP.parent / 'output/playwright/v13-tile-measurements.png',
    'EXPANDED_NATIVE': APP.parent / 'output/native-ui/inspection-en-360x800-expanded-research.png',
    'WORKSPACE': APP.parent / 'output/playwright/burhan-design-workspace.png',
    'REPORTS': APP.parent / 'output/playwright/burhan-design-reports.png',
    'HINT': APP / 'delivery/evaluation/v12-browser-hint.png',
    'SELECTED': APP / 'delivery/evaluation/v12-browser-selected.png',
    'REVIEW': APP / 'delivery/evaluation/v12-browser-review.png',
    'NATIVE': APP / 'delivery/evaluation/native-ui/inspection-en-360x800-subtle-hint.png',
    'ARABIC': APP / 'delivery/evaluation/native-ui/inspection-ar-360x800-font160.png',
    'SETTINGS': APP / 'delivery/evaluation/v12-browser-settings.png'
}
html = (HERE / 'guide.template.html').read_text(encoding='utf-8')
favicon = svg_header(64, 64, 'Burhan') + '<rect width="64" height="64" rx="9" fill="#111315"/>' + mark('#ffd43b', 12, 2, 1.2) + '</svg>'
html = html.replace('@@FAVICON@@', base64.b64encode(favicon.encode()).decode())
for token, filename in [('LATIN', 'NotoSans-Regular.ttf'), ('ARABIC', 'NotoSansArabic-Regular.ttf')]:
    path = APP / 'assets/fonts' / filename
    html = html.replace('@@FONT_' + token + '@@', base64.b64encode(path.read_bytes()).decode())
    copy_if_changed(path, ASSETS / filename)
for name in ['OFL-Arabic.txt', 'OFL-Latin.txt']:
    copy_if_changed(APP / 'assets/fonts' / name, ASSETS / name)
screen_manifest = {}
for token, path in screens.items():
    data = path.read_bytes()
    html = html.replace('@@SCREEN_' + token + '@@', 'data:image/png;base64,' + base64.b64encode(data).decode())
    target = ASSETS / ('Burhan-' + token.lower() + '-reference.png')
    target.write_bytes(data)
    screen_manifest[target.name] = {'source': os.path.relpath(path, APP).replace('\\', '/'), 'sha256': hashlib.sha256(data).hexdigest()}
html = html.replace('@@ASSETS_JSON@@', json.dumps(asset_data, ensure_ascii=False).replace('<', '\\u003c'))
assert not re.search(r'@@[A-Z_]+@@', html), 'Unreplaced template tokens'
output = HERE / 'Burhan-Design-Language.html'
output.write_text(html, encoding='utf-8')
manifest = {'product': 'Burhan', 'referenceVersion': '1.3.0', 'standaloneHtml': output.name, 'screens': screen_manifest,
            'sourceReferences': ['src/styles.css', 'src/engine/Scanner.css', 'src/i18n.ts', 'android/app/src/main/java/com/burhan/site/InspectionCameraUi.java'],
            'nativeFixtures': 'Rendered Android Material Views with blank camera preview. Not physical-phone footage.',
            'contentCanvases': 'Original vector content directions, not product screenshots.'}
(ASSETS / 'asset-manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
(HERE / 'README.txt').write_text('''BURHAN — DESIGN LANGUAGE & CREATOR KIT

Open Burhan-Design-Language.html in a modern browser. It is self-contained: fonts, UI references and interactive examples work offline. You can send this single HTML file to an editor or designer.

The companion assets folder contains three-bar mark and editable lockup SVGs, exact tokens, three suggested content layouts, English/Arabic copy, thirteen UI-reference PNGs and the Noto variable font files with original OFL license texts.

Keep the assets folder when editing the SVGs in a graphics tool; install/select Noto Sans and Noto Sans Arabic for live editable text. The HTML does not need the assets folder to display.

Native PNGs are actual rendered layout fixtures with a deliberately blank camera preview. Browser camera references use an imported evaluation image; workspace and report references show demo records in the current product. The interactive HTML phone and content canvases are labelled reconstructions/design suggestions, not detector recordings. Capture real phone footage for live-camera demonstrations.

Reference: Burhan 1.3.0, version code 5. Guide creation does not change the app or APK. Source references are recorded in assets/asset-manifest.json. Rebuild with python design/build-guide.py from the Burhan folder.
''', encoding='utf-8')
archive = HERE / 'Burhan-Designer-Kit.zip'
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    z.write(output, output.name)
    z.write(HERE / 'README.txt', 'README.txt')
    for path in sorted(ASSETS.iterdir()):
        if path.is_file():
            z.write(path, 'assets/' + path.name)
print(json.dumps({'html': str(output), 'htmlBytes': output.stat().st_size, 'designerKit': str(archive), 'zipBytes': archive.stat().st_size, 'screens': len(screens), 'assets': len(list(ASSETS.iterdir()))}, indent=2))
