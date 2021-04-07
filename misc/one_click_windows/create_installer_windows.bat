call rmdir dist /s /q
call rmdir build /s /q
call conda env remove -n alphatimsinstaller
call conda create -n alphatimsinstaller python=3.8 -y
REM call conda create -n alphatimsinstaller python=3.8 -y
call conda activate alphatimsinstaller
REM call conda install git -y
REM call pip install 'git+https://github.com/MannLabs/alphatims.git#egg=alphatims[gui]'
REM call conda install freetype
call pip install ../../.[plotting]
REM
call pip install pyinstaller==4.2
call pyinstaller ../pyinstaller/alphatims.spec -y
call conda deactivate
call "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" alphatims_innoinstaller.iss
REM call iscc alphatims_innoinstaller.iss
