call rmdir dist /s /q
call rmdir build /s /q
call conda env remove -n alphatimsinstaller
call conda create -n alphatimsinstaller python=3.8 -y
REM call conda create -n alphatimsinstaller python=3.8 -y
call conda activate alphatimsinstaller
REM call conda install git -y
REM call pip install 'git+https://github.com/MannLabs/alphatims.git#egg=alphatims[gui]'
REM call conda install freetype
REM call pip install ../../.[plotting]
REM call pip install pyinstaller==4.10
REM call pyinstaller ../pyinstaller/alphatims.spec -y
call cd ../..
call rmdir dist /s /q
call rmdir build /s /q
call python setup.py sdist bdist_wheel
call cd misc/one_click_windows
call pip install "../../dist/alphatims-1.0.9-py3-none-any.whl[plotting-stable,stable,legacy-stable]"
call pip install pyinstaller==4.10
call pyinstaller ../pyinstaller/alphatims.spec -y
call conda deactivate


if false; then
  REM call mkdir inno
  REM call pip install wget
  REM call python -m wget https://jrsoftware.org/download.php/is.exe
  call is.exe /SILENT /DIR=.
  call inno\ISCC.exe alphatims_innoinstaller.iss
else
  call "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" alphatims_innoinstaller.iss
  REM call iscc alphatims_innoinstaller.iss
fi
