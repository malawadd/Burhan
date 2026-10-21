"""Real local file adapters; no network or downloaded dataset required."""
import time

import cv2
import numpy as np
import pypdfium2 as pdfium
from fastapi.testclient import TestClient

from edge_runtime.app import create_app
from edge_runtime.sources import LocalSource


def image_file(path, color=120):
    ok, encoded = cv2.imencode('.png', np.full((80, 120, 3), color, np.uint8))
    assert ok
    encoded.tofile(path)
    return path


def test_unicode_still_image_repeats_without_becoming_a_video(tmp_path):
    path = image_file(tmp_path / 'مصنع.png')
    source = LocalSource('image:' + str(path), 'factory', 2)
    try:
        source.open()
        for i in range(10):
            frame, timestamp = source.read()
            assert frame.shape == (80, 120, 3)
            assert timestamp == i * 500
            frame[:] = 0  # adapters return independent frames
        assert source.backend == 'local_image'
    finally:
        source.close()


def test_folder_holds_each_image_then_reports_eof(tmp_path):
    image_file(tmp_path / '01.png', 40)
    image_file(tmp_path / '02.png', 200)
    source = LocalSource('images:' + str(tmp_path), 'freight', 1)
    try:
        source.open()
        assert [int(source.read()[0][0, 0, 0]) for _ in range(6)] == [40]*3 + [200]*3
        assert source.read() == (None, None)
    finally:
        source.close()


def test_pdf_pages_render_locally_and_close(tmp_path):
    path = tmp_path / 'pages.pdf'
    doc = pdfium.PdfDocument.new()
    page = doc.new_page(200, 300)
    page.close()
    doc.save(str(path))
    doc.close()
    source = LocalSource('pdf:' + str(path), 'freight', 1)
    try:
        source.open()
        image, _ = source.read()
        assert image.shape == (600, 400, 3)
        assert image.min() == 255
        source.read()
        source.read()
        assert source.read() == (None, None)
    finally:
        source.close()
    path.unlink()  # verify PDF handle released on Windows


def test_real_image_api_and_profile_options(tmp_path):
    path = image_file(tmp_path / 'page.png')
    with TestClient(create_app(tmp_path / 'events')) as client:
        response = client.post('/api/start', json={
            'profile': 'freight', 'source': 'image:' + str(path),
            'profile_options': {'stable_frames': 2}, 'deadline_ms': 2000,
        })
        assert response.status_code == 200
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline:
            state = client.get('/api/state').json()
            if state['metrics']['processed']:
                break
            time.sleep(.04)
        assert state['metrics']['inspection_errors'] == 0
        assert state['last_result']['provenance'] == 'local_document'
        assert state['config']['profile_options']['stable_frames'] == 2
