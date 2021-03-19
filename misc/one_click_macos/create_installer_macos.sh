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
# mv dist/alphatims dist/AlphaTims.app
# tar -czf dist/AlphaTims.app.zip dist/AlphaTims.app
# chmod +x dist/alphatims.app
# TODO No console is opened and program not blocked untill close, meaning loose threads!
# pkgbuild --root dist/alphatims --identifier org.alphatims.0.1.210317 --install-location /Applications/AlphaTims.app --scripts scripts --version 0.1.210317 alphatims.pkg
# productbuild --distribution distribution.xml --resources Resources --package-path alphatims.pkg dist/alphatims_installer_macos.pkg
mkdir -p dist/alphatims/Contents/Resources
cp ../alpha_logo.icns dist/alphatims/Contents/Resources
mv dist/alphatims_gui dist/alphatims/Contents/MacOS
cp Info.plist dist/alphatims/Contents
cp alphatims_terminal dist/alphatims/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
pkgbuild --root dist/alphatims --identifier org.alphatims.0.1.210317 --version 0.1.210317 --install-location /Applications/AlphaTims.app --scripts scripts alphatims.pkg
productbuild --distribution distribution.xml --resources Resources --package-path alphatims.pkg dist/alphatims_installer_macos.pkg
