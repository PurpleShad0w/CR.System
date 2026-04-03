@echo off
setlocal
REM Usage:
REM   render_with_log.cmd "path\to\render_report_pptx.py" --template "..." --assembled "..." --out "..." --info-template "..." --slide-types "..."
REM This captures stdout+stderr to a log file next to the output pptx.

set LOG=%~dp0render.log
python %* > "%LOG%" 2>&1

echo.
echo Log written to: %LOG%
pause
