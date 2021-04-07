#!bash
rm -rf dist
rm -rf build
conda env remove -n alphatimsinstaller
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
pip install "../../dist/alphatims-0.2.0-py3-none-any.whl[plotting]"
pip install pyinstaller==4.2
pyinstaller ../pyinstaller/alphatims.spec -y
conda deactivate
# mv dist/alphatims dist/AlphaTims
mkdir -p dist/alphatims_gui_installer_linux/usr/local/bin
mv dist/AlphaTims dist/alphatims_gui_installer_linux/usr/local/bin/alphatims
mkdir dist/alphatims_gui_installer_linux/DEBIAN
cp control dist/alphatims_gui_installer_linux/DEBIAN
dpkg-deb --build --root-owner-group dist/alphatims_gui_installer_linux/
