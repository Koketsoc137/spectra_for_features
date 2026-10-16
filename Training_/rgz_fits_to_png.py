import glob
import os
from multiprocessing import Pool

import numpy as np
from PIL import Image

FITS_DIR = "/idia/projects/camil/Koketso/radio_galaxy_zoo/images_full"
# ImageFolder root is radiogalaxyzoo/images (single class folder "rgz"), next to galaxyzoo2
PNG_DIR = "/idia/projects/camil/Koketso/radiogalaxyzoo/images/rgz"


def read_fits(path):
    #minimal reader for the single-HDU float32 FIRST cutouts (no astropy in the venv)
    raw = open(path, "rb").read()
    header = {}
    for i in range(0, len(raw), 80):
        card = raw[i:i + 80].decode("ascii", "replace")
        if card.startswith("END"):
            start = (i // 2880 + 1) * 2880
            break
        if card[8:10] == "= ":
            header[card[:8].strip()] = card[10:].split("/")[0].strip()
    assert header["BITPIX"] == "-32", path
    nx, ny = int(header["NAXIS1"]), int(header["NAXIS2"])
    return np.frombuffer(raw, ">f4", nx * ny, start).reshape(ny, nx)


def convert(path):
    out = os.path.join(PNG_DIR, os.path.basename(path).replace(".fits", ".png"))
    if os.path.exists(out):
        return
    img = np.nan_to_num(read_fits(path).astype(np.float64))
    # MiraBest-style preprocessing: zero pixels below 3 sigma (MAD noise), scale to 0-255, north up
    noise = 1.4826 * np.median(np.abs(img - np.median(img)))
    img = np.where(img > 3 * noise, img, 0)
    if img.max() > 0:
        img = img / img.max() * 255
    Image.fromarray(np.flipud(img).astype(np.uint8), mode="L").save(out)


if __name__ == "__main__":
    os.makedirs(PNG_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(FITS_DIR, "*.fits")))
    print(f"converting {len(paths)} FITS -> {PNG_DIR}")
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 4))) as pool:
        pool.map(convert, paths, chunksize=500)
    print(f"done: {len(glob.glob(os.path.join(PNG_DIR, '*.png')))} PNGs")
