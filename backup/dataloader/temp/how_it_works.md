from dataloader.mesh_dataloader import get_mesh_dataloader
from dataloader.highres_image_dataloader import get_highres_image_dataloader
from dataloader.lowres_image_dataloader import get_lowres_image_dataloader

# Point cloud dataloader
mesh_loader = get_mesh_dataloader(
    root_dir="data/nuclei_sample_1a_v1",
    class_csv_path="chromatin_classes_and_samples.csv",
    max_points=10000,
    cache_dir="data/pointclouds_cache"
)

# High-res image dataloader
highres_loader = get_highres_image_dataloader(
    root_dir="data/nuclei_sample_1a_v1",
    class_csv_path="chromatin_classes_and_samples.csv",
    target_size=(224, 224),
    slices_per_sample=5
)

# Low-res image dataloader for coarse texture
lowres_loader = get_lowres_image_dataloader(
    root_dir="data/nuclei_sample_1a_v1",
    class_csv_path="chromatin_classes_and_samples.csv",
    target_size=(64, 64),
    z_window_size=5,
    z_stride=3
)