call conda create -n alphatimsinstaller python=3.8 -y
call conda activate alphatimsinstaller
REM call conda install git -y
call pip install git+https://github.com/MannLabs/alphatims.git --use-feature=2020-resolver
call pip install pyinstaller
call cd alphatims/misc
call pyinstaller alphatims.spec -y
call conda deactivate
