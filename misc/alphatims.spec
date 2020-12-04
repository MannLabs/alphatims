# -*- mode: python ; coding: utf-8 -*-

import pkgutil
import os
import sys
from PyInstaller.building.build_main import Analysis, PYZ, EXE, COLLECT, BUNDLE, TOC
from PyInstaller.utils.hooks import get_package_paths, remove_prefix, PY_IGNORE_EXTENSIONS, copy_metadata, collect_all
import pkg_resources
import importlib.metadata

##################### User definitions
exe_name = 'alphatims_gui'
script_name = 'alphatims_pyinstaller.py'
icon = 'alpha_logo.ico'
block_cipher = None
location = os.getcwd()
project = "alphatims"
remove_tests = True
#####################


requirements = {
	req.split()[0] for req in importlib.metadata.requires(project)
}
requirements.add(project)
requirements.add("distributed")
hidden_imports = set()
datas = []
binaries = []
checked = set()
while requirements:
	requirement = requirements.pop()
	checked.add(requirement)
	if requirement in ["pywin32"]:
		continue
	try:
		module_version = importlib.metadata.version(requirement)
	except (
		importlib.metadata.PackageNotFoundError,
		ModuleNotFoundError,
		ImportError
	):
		continue
	try:
		datas_, binaries_, hidden_imports_ = collect_all(
			requirement,
			include_py_files=True
		)
	except ImportError:
		continue
	datas += datas_
	# binaries += binaries_
	hidden_imports_ = set(hidden_imports_)
	if "" in hidden_imports_:
		hidden_imports_.remove("")
	if None in hidden_imports_:
		hidden_imports_.remove(None)
	requirements |= hidden_imports_ - checked
	hidden_imports |= hidden_imports_

if remove_tests:
	hidden_imports = sorted(
		[h for h in hidden_imports if "tests" not in h.split(".")]
	)
else:
	hidden_imports = sorted(hidden_imports)

a = Analysis(
	[script_name],
	pathex=[location],
	binaries=binaries,
	datas=datas,
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
	console=True,
	icon=icon
)
coll = COLLECT(
	exe,
	a.binaries,
	a.zipfiles,
	a.datas,
	strip=False,
	upx=True,
	upx_exclude=[],
	name=exe_name
)

# exe = EXE(
# 		pyz,
# 		a.scripts,
# 		a.binaries,
# 		a.zipfiles,
# 		a.datas,
# 		[],
# 		name=exe_name,
# 		debug=False,
# 		bootloader_ignore_signals=False,
# 		strip=False,
# 		upx=True,
# 		upx_exclude=[],
# 		runtime_tmpdir=None,
# 		console=True,
# 		icon=icon
# )
