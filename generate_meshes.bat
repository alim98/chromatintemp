@echo off
echo Chromatin Mesh Generation Pipeline
echo ================================

:: Use conda activate with a safer approach
call conda.bat activate chromatin || echo Conda activation failed, but continuing anyway...

:: Create directory for results
mkdir results\meshes 2>nul
mkdir results\analysis_output 2>nul
mkdir results\visualizations 2>nul
mkdir results\html_pointcloud 2>nul
mkdir results\html 2>nul

:: Process first 2 samples
echo Processing first 2 samples...
python scripts\generate_meshes.py --max_samples 2 --output_dir results\meshes --smooth 10 --decimate 5000 --points 1024

:: Visualize one sample as example
echo Generating visualization for sample 1...
python scripts\visualize_mesh_comparison.py --sample_id 1 --save

:: Test the generated meshes
echo Testing generated meshes...
python scripts\test_meshes.py --test_dataloader

:: Analyze the meshes
echo Analyzing mesh properties...
python scripts\analyze_meshes.py

:: Generate HTML report
echo Generating comprehensive HTML report...
python scripts\generate_mesh_report.py

echo.
echo Pipeline completed!
echo Results saved to:
echo - Meshes: results\meshes\meshes
echo - Point clouds: results\meshes\pointclouds
echo - Visualizations: results\html_pointcloud
echo - Validation: results\analysis_output\mesh_validation_meshes.csv
echo - Analysis: results\analysis_output\mesh_analysis.csv
echo - Plots: results\visualizations\mesh_analysis
echo - HTML Report: results\html\mesh_report.html

:: Open the HTML report
echo Opening HTML report...
start "" "results\html\mesh_report.html"

:: Pause to view results
pause 