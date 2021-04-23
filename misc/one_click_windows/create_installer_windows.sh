rm -rf dist
rm -rf build
conda create -n alphatimsinstaller python=3.8 -y
conda activate alphatimsinstaller
cd ../..
rm -rf dist
rm -rf build
python setup.py sdist bdist_wheel
cd misc/one_click_windows
pip install "../../dist/alphatims-0.2.5-py3-none-any.whl[plotting]"
pip install pyinstaller==4.2
# TODO https://stackoverflow.com/questions/54175042/python-3-7-anaconda-environment-import-ssl-dll-load-fail-error/60405693#60405693
pyinstaller ../pyinstaller/alphatims.spec -y
conda deactivate

FILE="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if test -f "$FILE"; then
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" alphatims_innoinstaller.iss
else
  mkdir inno
  is.exe //SILENT //DIR=inno
  "inno/ISCC.exe" alphatims_innoinstaller.iss
fi


# if false; then
#   is.exe /SILENT /DIR=.
#   inno\ISCC.exe alphatims_innoinstaller.iss
# else
#   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" alphatims_innoinstaller.iss
# fi
