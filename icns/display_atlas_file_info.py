from icns import common

from nibabel import Nifti1Image, load

print('=== ADHD 200 Atlas Information ===')

for atlas_type in common.atlas_types:
    print(f'Atlas: {atlas_type}')
    path_to_atlas = common.create_template_path(atlas_type)
    print(f'Path to atlas: {path_to_atlas}')
    atlas_image: Nifti1Image = load(path_to_atlas)
    print(f'Atlas shape: {atlas_image.shape}')
