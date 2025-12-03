# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = [('C:\\Kuliah\\STMH\\python_ambulance_stmh\\simulation_sukabumi', 'simulation_sukabumi')]
binaries = []
hiddenimports = ['nicegui', 'starlette', 'uvicorn', 'folium', 'networkx', 'matplotlib', 'pandas', 'numpy', 'branca', 'jinja2', 'osmnx', 'geopandas', 'shapely', 'pyogrio', 'pyogrio._io', 'pyogrio._geometry', 'pyogrio.core', 'pyogrio._env', 'pyproj']
datas += copy_metadata('osmnx')
datas += copy_metadata('nicegui')
datas += copy_metadata('folium')
datas += copy_metadata('geopandas')
datas += copy_metadata('shapely')
datas += copy_metadata('networkx')
datas += copy_metadata('pyogrio')
datas += copy_metadata('pyproj')
tmp_ret = collect_all('osmnx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('nicegui')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyogrio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pyproj')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('geopandas')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('shapely')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AmbulanceRouting',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
