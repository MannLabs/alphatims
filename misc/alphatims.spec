# -*- mode: python ; coding: utf-8 -*-

import pkgutil
import os
import sys
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE, TOC


hidden_imports = [
	i[1] for i in pkgutil.iter_modules() if i[2]
]
exe_name = 'alphatims'
script_name = 'alphatims_pyinstaller.py'
location = os.getcwd()


def collect_pkg_data(package, include_py_files=False, subdir=None):
    from PyInstaller.utils.hooks import get_package_paths, remove_prefix, PY_IGNORE_EXTENSIONS
    # Accept only strings as packages.
    if type(package) is not str:
        raise ValueError
    try:
        pkg_base, pkg_dir = get_package_paths(package)
    except ImportError:
        print(package)
        return TOC()
    if subdir:
        pkg_dir = os.path.join(pkg_dir, subdir)
    # Walk through all file in the given package, looking for data files.
    data_toc = TOC()
    for dir_path, dir_names, files in os.walk(pkg_dir):
        for f in files:
            extension = os.path.splitext(f)[1]
            if include_py_files or (extension not in PY_IGNORE_EXTENSIONS):
                source_file = os.path.join(dir_path, f)
                dest_folder = remove_prefix(dir_path, os.path.dirname(pkg_base) + os.sep)
                dest_file = os.path.join(dest_folder, f)
                data_toc.append((dest_file, source_file, 'DATA'))
    return data_toc


pkg_data = [
		collect_pkg_data(package_name, False) for package_name in hidden_imports
]


block_cipher = None


a = Analysis(
		[script_name],
		pathex=[location],
		binaries=[],
		datas=[],
		hiddenimports=hidden_imports,
		hookspath=[],
		runtime_hooks=[],
		excludes=[],
		win_no_prefer_redirects=False,
		win_private_assemblies=False,
		cipher=block_cipher,
		noarchive=False
)
pyz = PYZ(
		a.pure,
		a.zipped_data,
		cipher=block_cipher
)
exe = EXE(
		pyz,
	  a.scripts,
	  [],
	  exclude_binaries=True,
	  name=exe_name,
	  debug=False,
	  bootloader_ignore_signals=False,
	  strip=False,
	  upx=True,
	  console=True
)
coll = COLLECT(
		exe,
		a.binaries,
		a.zipfiles,
		a.datas,
		*pkg_data,
		strip=False,
		upx=True,
		upx_exclude=[],
		name=exe_name
)
