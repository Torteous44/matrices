# Documentation Figures

This directory will contain automatically generated figures during test runs, including:

- F-matrix epipolar overlay visualizations
- Sparse point cloud visualizations
- Dense reconstruction views
- Mesh screenshots
- Animated GIF of 3D model orbit

The figures will be generated when you run the pipeline or tests. They are not included in the repository but will be created during runtime.

## Expected Generated Figures

When running the pipeline, the following figures may be generated:

- `epipolar_lines.png`: Visualization of epipolar geometry showing fundamental matrix constraints
- `sparse_reconstruction.png`: Visualization of the sparse point cloud with camera poses
- `dense_reconstruction.png`: Visualization of the dense point cloud
- `mesh_reconstruction.png`: Visualization of the textured mesh
- `reconstruction.gif`: Animated orbit around the reconstructed 3D model
- Various other diagnostic visualizations

## How to Generate Figures

To generate these figures, run the pipeline with visualization enabled:

```bash
python scripts/run_pipeline.py --images data/sample --visualise
```

For test-generated figures:

```bash
python -m pytest tests/test_pipeline.py -v
``` 