rm -rf ../docs/build
conda env remove -n alphatimsdocs
conda create -n alphatimsdocs python=3.8 pip=20.2 -y
# conda create -n alphatimsinstaller python=3.8
conda activate alphatimsdocs
# call conda install git -y
# call pip install 'git+https://github.com/MannLabs/alphatims.git#egg=alphatims[gui]' --use-feature=2020-resolver
# brew install freetype
pip install '../../.'
pip install sphinx_rtd_theme
pip install sphinx
make html
conda deactivate
