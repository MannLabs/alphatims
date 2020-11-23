conda create -n alphatimsinstaller python=3.8 -y
conda activate alphatimsinstaller
pip install git+https://github.com/MannLabs/alphatims.git --use-feature=2020-resolver
pip install pyinstaller
cd alphatims/misc
pyinstaller -y -n alphatims alphatims_pyinstaller.py
