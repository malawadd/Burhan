"""Freeze the separately accepted 1171-view adaptation; never start a fit."""
from freeze_construction_broad_stage2 import ROOT, freeze_candidate

if __name__ == '__main__':
    freeze_candidate(
        ROOT/'datasets/construction/evcivil-round4-real-lowlight-gain',
        'evcivil-yolov8n-r4-real-lowlight-640-gain',
        'c298085a4fd7ab4b1051f432c807c73f67f0d3902948f47b5f51c5d7430882a0',
        combined=True,
    )
