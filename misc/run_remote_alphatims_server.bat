@echo off

SET user=username
SET server=servername
SET alphatims_executable=full_path_to_alphatims

echo Provided file: "%~1"
echo User: %user%
echo Server: %server%
echo Alphatims executable: %alphatims_executable%

ssh -X %user%@%server% "%alphatims_executable% gui --bruker_raw_data $(echo '%~1' | tr '\\' '/')"
pause
