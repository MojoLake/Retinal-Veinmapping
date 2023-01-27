from pathlib import Path
from PIL import Image


def get_rgb(dir):
    files = Path(dir).glob('*')
    ret = []

    for file in files:
        image = Image.open(file)
        image.load()

        cur = Image.new('RGB', image.size, (255, 255, 255))
        cur.paste(image, None)
        ret.append(cur)

    return ret


