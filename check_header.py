import os; os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import nibabel as nib

files = [
    "input/pretrain/sub-001_ses-1_task-rest_acq-EPI_bold.nii.gz",
    "input/target/driftRespCardlessBOLD_B16_Final_scan5.nii",
]

for fpath in files:
    img = nib.load(fpath)
    hdr = img.header
    print(f"\n{'='*60}")
    print(f"  {os.path.basename(fpath)}")
    print(f"{'='*60}")
    print(f"  hdr['pixdim']  : {list(hdr['pixdim'])}")
    print(f"  get_zooms()    : {hdr.get_zooms()}")
    print(f"  xyzt_units     : {hdr.get_xyzt_units()}")
    print(f"  shape          : {img.shape}")
    print()
    print("  How we read it:")
    zooms = hdr.get_zooms()
    sz = float(zooms[2])
    sy = float(zooms[0])
    sx = float(zooms[1])
    tr_raw = float(zooms[3]) if len(zooms) >= 4 else None
    time_unit = hdr.get_xyzt_units()[1]
    print(f"    zooms[0] = {zooms[0]:.4f}  -> sy (voxel size along H axis) = {sy} mm")
    print(f"    zooms[1] = {zooms[1]:.4f}  -> sx (voxel size along W axis) = {sx} mm")
    print(f"    zooms[2] = {zooms[2]:.4f}  -> sz (voxel size along Z axis) = {sz} mm")
    if tr_raw is not None:
        print(f"    zooms[3] = {tr_raw:.4f}  -> TR raw, time_unit='{time_unit}'")
        if time_unit == "msec" or (time_unit == "sec" and tr_raw > 100):
            t_gap = tr_raw / 1000.0
            print(f"    heuristic: TR={tr_raw} with unit='{time_unit}' looks like ms -> divide by 1000 -> t_gap={t_gap} s")
        else:
            t_gap = tr_raw
            print(f"    unit is '{time_unit}' and TR={tr_raw} <= 100 -> t_gap={t_gap} s  (no conversion)")
    print()
    print(f"  NIfTI standard: pixdim[0]=qfac, pixdim[1..3]=voxel sizes, pixdim[4]=TR")
    print(f"  nibabel get_zooms() returns pixdim[1..N] for an N-dim image")
    print(f"  So for a 4D [H,W,Z,T] volume: zooms = (pixdim[1], pixdim[2], pixdim[3], pixdim[4])")
