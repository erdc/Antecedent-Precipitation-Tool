# -*- mode: python ; coding: utf-8 -*-
# command to compile with this file: 
#  pyinstaller main_ex.spec --noconfirm
# note: If mkl_intel_thread.2.dll error uninstall and reinstall numpy with pip

import os
import sys

# Construct the path to the distributed package from conda env path
conda_env_path = sys.prefix
distributed_path = os.path.join(conda_env_path, 'Lib', 'site-packages', 'distributed')
sys.setrecursionlimit(sys.getrecursionlimit() * 5)


block_cipher = None

added_files = [
    ( '.\\images\\*', 'images' ),
    ('.\\data\\*', 'data'),
    ( '.\\proj.db', '.'),
    (distributed_path, 'distributed')
]

a = Analysis(
    ['main_ex.py'],
    pathex=['.'],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'cftime',
		'cftime._strptime',
		'matplotlib.backends.backend_pdf',
		'osgeo._gdal',
        's3fs'
        ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main_ex',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='.\\images\\Graph.ico'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_ex',
)
