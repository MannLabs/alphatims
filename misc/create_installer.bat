call conda create -n alphatimsinstaller python=3.8 -y
call conda activate alphatimsinstaller
REM call conda install git -y
REM call pip install 'git+https://github.com/MannLabs/alphatims.git#egg=alphatims[gui,cli]' --use-feature=2020-resolver
call pip install ../.[gui,cli] --use-feature=2020-resolver
call pip install pyinstaller
call pyinstaller alphatims.spec -y
REM call "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" alphatims_innoinstaller.iss
call iscc alphatims_innoinstaller.iss
call conda deactivate
