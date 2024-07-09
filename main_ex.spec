# -*- mode: python ; coding: utf-8 -*-
# command to compile with this file: 
#  pyinstaller main_ex.spec --noconfirm


block_cipher = None

added_files = [
    ( '.\\images\\*', 'images' ),
    ('.\\data\\*', 'data'),
    ( '.\\version', '.' ),
    ( '.\\v\\main_ex', 'v' ),
    ( '.\\proj.db', '.')
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
