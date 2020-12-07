conda create -n alphatimsinstaller python=3.8 pip=20.2 -y
# conda create -n alphatimsinstaller python=3.8
conda activate alphatimsinstaller
# call conda install git -y
# call pip install 'git+https://github.com/MannLabs/alphatims.git#egg=alphatims[gui]' --use-feature=2020-resolver
# brew install freetype
pip install '../../.[gui]'
pip install pyinstaller
pyinstaller ../pyinstaller/alphatims.spec -y
conda deactivate
