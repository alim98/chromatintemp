@echo off
REM Batch file to generate point clouds with different point counts

if "%1"=="" (
    echo Usage: generate_multiscale_pointclouds.bat SAMPLE_ID
    echo Example: generate_multiscale_pointclouds.bat 0011
    exit /b
)

set SAMPLE_ID=%1
set OUTPUT_DIR=data\pointclouds_multiscale
set HTML_DIR=results\html_pointcloud

echo Creating directories...
mkdir %OUTPUT_DIR% 2>nul
mkdir %HTML_DIR% 2>nul

echo Generating point cloud with 1,000 points...
python utils\pointcloud.py --sample_id %SAMPLE_ID% --max_points 1000 --sample_rate 1.0 --output_dir %OUTPUT_DIR%
python utils\pointcloud_to_html.py --input "%OUTPUT_DIR%\%SAMPLE_ID%.ply" --output "%HTML_DIR%\%SAMPLE_ID%_1000.html" --title "Sample %SAMPLE_ID% - 1,000 Points"

echo Generating point cloud with 10,000 points...
python utils\pointcloud.py --sample_id %SAMPLE_ID% --max_points 10000 --sample_rate 1.0 --output_dir %OUTPUT_DIR%
python utils\pointcloud_to_html.py --input "%OUTPUT_DIR%\%SAMPLE_ID%.ply" --output "%HTML_DIR%\%SAMPLE_ID%_10000.html" --title "Sample %SAMPLE_ID% - 10,000 Points"

echo Generating point cloud with 100,000 points...
python utils\pointcloud.py --sample_id %SAMPLE_ID% --max_points 100000 --sample_rate 1.0 --output_dir %OUTPUT_DIR%
python utils\pointcloud_to_html.py --input "%OUTPUT_DIR%\%SAMPLE_ID%.ply" --output "%HTML_DIR%\%SAMPLE_ID%_100000.html" --title "Sample %SAMPLE_ID% - 100,000 Points"

echo Generating point cloud with 1,000,000 points...
python utils\pointcloud.py --sample_id %SAMPLE_ID% --max_points 1000000 --sample_rate 1.0 --output_dir %OUTPUT_DIR%
python utils\pointcloud_to_html.py --input "%OUTPUT_DIR%\%SAMPLE_ID%.ply" --output "%HTML_DIR%\%SAMPLE_ID%_1000000.html" --title "Sample %SAMPLE_ID% - 1,000,000 Points"

echo Generating full resolution point cloud...
python utils\pointcloud.py --sample_id %SAMPLE_ID% --sample_rate 0.1 --output_dir %OUTPUT_DIR%
python utils\pointcloud_to_html.py --input "%OUTPUT_DIR%\%SAMPLE_ID%.ply" --output "%HTML_DIR%\%SAMPLE_ID%_full.html" --title "Sample %SAMPLE_ID% - Full Resolution"

echo Done! Point clouds saved to %OUTPUT_DIR% and HTML files saved to %HTML_DIR% 