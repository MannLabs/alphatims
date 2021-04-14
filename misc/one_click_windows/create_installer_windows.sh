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
cd misc/one_click_windows
pip install "../../dist/alphatims-0.2.2-py3-none-any.whl[plotting]"
pip install pyinstaller==4.2
pyinstaller ../pyinstaller/alphatims.spec -y
conda deactivate

echo "start weird stuff"
FILE="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if test -f "$FILE"; then
  echo "0"
  "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" alphatims_innoinstaller.iss
  echo "0a"
else
  mkdir inno
  echo "1"
  is.exe //SILENT //DIR=inno
  echo "2"
  ls -lah
  echo "2a"
  ls -lah inno
  "inno/ISCC.exe" alphatims_innoinstaller.iss
  echo "3"
fi


# if false; then
#   is.exe /SILENT /DIR=.
#   inno\ISCC.exe alphatims_innoinstaller.iss
# else
#   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" alphatims_innoinstaller.iss
# fi
