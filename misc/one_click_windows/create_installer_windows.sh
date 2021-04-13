rm -rf dist
rm -rf build
conda create -n alphatimsinstaller python=3.8 -y
# conda create -n alphatimsinstaller python=3.8
conda activate alphatimsinstaller
# call conda install git -y
# call pip install 'git+https://github.com/MannLabs/alphatims.git#egg=alphatims[gui]' --use-feature=2020-resolver
# brew install freetype
cd ../..
rm -rf dist
rm -rf build
python setup.py sdist bdist_wheel
cd misc/one_click_linux
pip install "../../dist/alphatims-0.2.2-py3-none-any.whl[plotting]"
pip install pyinstaller==4.2
pyinstaller ../pyinstaller/alphatims.spec -y
conda deactivate

FILE="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if test -f "$FILE"; then
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" alphatims_innoinstaller.iss
else
  mkdir inno
  is.exe /SILENT /DIR=./inno
  "inno/ISCC.exe" alphatims_innoinstaller.iss
fi


# if false; then
#   is.exe /SILENT /DIR=.
#   inno\ISCC.exe alphatims_innoinstaller.iss
# else
#   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" alphatims_innoinstaller.iss
# fi
