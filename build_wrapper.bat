@echo off
REM Build script for Python wrapper DLL

REM Set compiler path (adjust if needed)
set CC="C:\Program Files (x86)\Embarcadero\Dev-Cpp\TDM-GCC-64\bin\gcc.exe"

REM Collect C source files (excluding test files and main files)
set SOURCES=
for %%f in (*.c) do (
    echo %%f | findstr /i "test_ vn_ threadpool" >nul
    if errorlevel 1 (
        set SOURCES=!SOURCES! %%f
    )
)

REM Compile as shared library
%CC% -shared -std=c99 -O2 -fPIC -DPYTHON_WRAPPER ^
    python_wrapper.c ^
    tensor.c conv2d.c pool.c ln.c ffn.c attn.c ^
    -o y9_wrapper.dll ^
    -Wl,--out-implib,y9_wrapper.lib

if %ERRORLEVEL% EQU 0 (
    echo Build successful: y9_wrapper.dll
) else (
    echo Build failed
    exit /b 1
)
