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
mv dist/alphatims dist/AlphaTims.app
tar -czf dist/AlphaTims.app.zip dist/AlphaTims.app
# chmod +x dist/alphatims.app
# TODO No console is opened and program not blocked untill close, meaning loose threads!
