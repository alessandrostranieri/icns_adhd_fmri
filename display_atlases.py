from nibabel import load, Nifti1Image
from nilearn import plotting

from icns.common import create_template_path, Atlas

cc200_atlas_file = create_template_path(Atlas.CC200)
cc200_atlas_image: Nifti1Image = load(cc200_atlas_file)

plotting.plot_roi(cc200_atlas_image, title="cc200")
plotting.show()

aal_atlas_file = create_template_path(Atlas.AAL)
aal_atlas_image: Nifti1Image = load(aal_atlas_file)

plotting.plot_roi(aal_atlas_image, title="AAL")
plotting.show()
