@echo off
REM Educational MCP Server - Quick Launcher for Windows
REM Double-click this file to start the interactive setup assistant

title ResilientDB Educational MCP Server

echo.
echo ========================================================================
echo   ResilientDB Educational MCP Server - Quick Launcher
echo ========================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8 or higher from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo Python found - starting launcher...
echo.

REM Run the interactive launcher
python launch.py

pause
