rm -rf dist
rm -rf build
FILE=alphatims.pkg
if test -f "$FILE"; then
  rm alphatims.pkg
fi
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
cd misc/one_click_macos
pip install pyinstaller==5.6.2
pip install "../../dist/alphatims-1.0.7-py3-none-any.whl[plotting-stable,stable,legacy-stable]"
conda list
pyinstaller ../pyinstaller/alphatims.spec -y
conda deactivate
mkdir -p dist/alphatims/Contents/Resources
cp ../alpha_logo.icns dist/alphatims/Contents/Resources
mv dist/alphatims_gui dist/alphatims/Contents/MacOS
cp Info.plist dist/alphatims/Contents
cp alphatims_terminal dist/alphatims/Contents/MacOS
cp ../../LICENSE.txt Resources/LICENSE.txt
cp ../alpha_logo.png Resources/alpha_logo.png
chmod 777 scripts/*

if false; then
  # https://scriptingosx.com/2019/09/notarize-a-command-line-tool/
  for f in $(find dist/alphatims -name '*.so' -or -name '*.dylib'); do codesign --sign "Developer ID Application: Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (7QSY5527AQ)" $f; done
  codesign --sign "Developer ID Application: Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (7QSY5527AQ)" dist/alphatims/Contents/MacOS/alphatims_gui --force --options=runtime --entitlements entitlements.xml
  pkgbuild --root dist/alphatims --identifier de.mpg.biochem.alphatims.app --version 1.0.7 --install-location /Applications/AlphaTims.app --scripts scripts alphatims.pkg --sign "Developer ID Installer: Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (7QSY5527AQ)"
  productbuild --distribution distribution.xml --resources Resources --package-path alphatims.pkg dist/alphatims_gui_installer_macos.pkg --sign "Developer ID Installer: Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (7QSY5527AQ)"
  requestUUID=$(xcrun altool --notarize-app --primary-bundle-id "de.mpg.biochem.alphatims.app" --username "willems@biochem.mpg.de" --password "@keychain:Alphatims-develop" --asc-provider 7QSY5527AQ --file dist/alphatims_gui_installer_macos.pkg 2>&1 | awk '/RequestUUID/ { print $NF; }')
  request_status="in progress"
  while [[ "$request_status" == "in progress" ]]; do
      echo "$request_status"
      sleep 10
      request_status=$(xcrun altool --notarization-info "$requestUUID" --username "willems@biochem.mpg.de" --password "@keychain:Alphatims-develop" | awk -F ': ' '/Status:/ { print $2; }' )
  done
  xcrun altool --notarization-info "$requestUUID" --username "willems@biochem.mpg.de" --password "@keychain:Alphatims-develop"
  xcrun stapler staple dist/alphatims_gui_installer_macos.pkg
else
  pkgbuild --root dist/alphatims --identifier de.mpg.biochem.alphatims.app --version 1.0.7 --install-location /Applications/AlphaTims.app --scripts scripts alphatims.pkg
  productbuild --distribution distribution.xml --resources Resources --package-path alphatims.pkg dist/alphatims_gui_installer_macos.pkg
fi
