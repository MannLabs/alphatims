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
python -m pyinstaller ../pyinstaller/alphatims.spec -y
conda deactivate
mkdir -p dist/alphatims/Contents/Resources
cp ../alpha_logo.icns dist/alphatims/Contents/Resources
mv dist/alphatims_gui dist/alphatims/Contents/MacOS
cp Info.plist dist/alphatims/Contents
cp alphatims_terminal dist/alphatims/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
cp ../alpha_logo.png Resources/alpha_logo.png
pkgbuild --root dist/alphatims --identifier org.alphatims.0.1.210330 --version 0.1.210330 --install-location /Applications/AlphaTims.app --scripts scripts alphatims.pkg
productbuild --distribution distribution.xml --resources Resources --package-path alphatims.pkg dist/alphatims_gui_installer_macos.pkg
# productsign --sign "Developer ID Installer: John Doe" dist/alphatims_gui_installer_macos.pkg dist/alphatims_gui_installer_macos_signed.pkg
