# -*- mode: python ; coding: utf-8 -*-
# Build with:
#   pyinstaller main.spec --noconfirm
#
# Expects to be run from the project root that contains:
#   main.py, engine.py, gui.py, config.py,
#   adapters/, core/, data/, images/, url_manifest.json

import os
import matplotlib
from PyInstaller.utils.hooks import collect_all

# Dynamically find the matplotlib cache directory during the build
mpl_cache_dir = matplotlib.get_cachedir()

# Collect s3fs + its transitive runtime needs
s3fs_datas, s3fs_binaries, s3fs_hiddenimports = collect_all('s3fs')

block_cipher = None

# Data files bundled next to the executable (or inside the COLLECT folder).
# Paths use forward slashes so the same .spec works on Windows and Linux.
# Note: tool needs to run at least once to create mpl cache as it should not
# be tracked by the git repo (probably shouldn't distribute fonts)
added_files = [
    ('data', 'data'),                   # GHCN lists, WBD.zip, logos, NetCDFs, etc.
    ('url_manifest.json', '.'),         # used by config.load_manifest()
    ('images/*', 'images'),             # Graph.ico and any other assets
    (mpl_cache_dir, 'mpl_cache'),       # AUTOMATICALLY INCLUDE MATPLOTLIB CACHE
]

a = Analysis(
    ['main.py'],
    pathex=['.'],
    binaries=s3fs_binaries,
    datas=added_files + s3fs_datas,
    hiddenimports=[
        # Core scientific stack
        'pandas',
        'numpy',
        'numpy._core',
        'numpy._core._exceptions',
        'numpy.core._methods',
        'scipy',
        'matplotlib',
        'matplotlib.backends.backend_pdf',
        'matplotlib.backends.backend_tkagg',
        'PIL',
        'PIL.Image',
        's3fs',
        'fsspec',
        'aiobotocore',
        'botocore',
        'boto3',
        's3transfer',
        'jmespath',
        'dateutil',
        'urllib3',
        # Geospatial
        'geopandas',
        'shapely',
        'shapely.geometry',
        'shapely.ops',
        'fiona',
        'pyproj',
        'rtree',
        # Network / IO
        'requests',
        'urllib3',
        'pypdf',
        'geopy',
        # Tkinter (explicit for some environments)
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        # Application packages (helps PyInstaller find them after the adapters/core split)
        'adapters',
        'adapters.precip_adapter',
        'adapters.plotter_adapter',
        'adapters.usgs_adapter',
        'adapters.nwm_adapter',
        'adapters.wimp_adapter',
        'adapters.pdsi_adapter',
        'adapters.area_adapter',
        'core',
        'core.precip',
        'core.plotting',
        'core.area_sampling',
        'core.precip_downloader',
        'core.gridded_downloader',
        'core.usgs_stream',
        'core.nwm_stream',
        'core.wimp_analysis',
        'core.pdsi',
        'engine',
        'gui',
        'config',
    ]+ s3fs_hiddenimports,
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
    name='APT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,                       # keep console for logging; set False for pure GUI
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='images/Graph.ico',            # relative path; works on Windows & Linux
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='APT',
)
