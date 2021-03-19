rm -rf dist
rm -rf build
conda env remove -n alphatimsinstaller
conda create -n alphatimsinstaller python=3.8 pip=20.2 -y
# conda create -n alphatimsinstaller python=3.8
conda activate alphatimsinstaller
# call conda install git -y
# call pip install 'git+https://github.com/MannLabs/alphatims.git#egg=alphatims[gui]' --use-feature=2020-resolver
# brew install freetype
pip install '../../.[plotting]'
pip install pyinstaller==4.2
pyinstaller ../pyinstaller/alphatims.spec -y
conda deactivate
# mv dist/alphatims dist/AlphaTims
mkdir -p dist/alphatims_installer_linux/usr/local/bin
cp dist/alphatims dist/alphatims_installer_linux/usr/local/bin
mkdir dist/alphatims_installer_linux/DEBIAN
cp control dist/alphatims_installer_linux/DEBIAN
dpkg-deb --build --root-owner-group dist/alphatims_installer_linux/
