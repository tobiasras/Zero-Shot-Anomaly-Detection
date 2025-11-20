import urllib.request
from util.paths import PROJECT_ROOT

weights_dir = PROJECT_ROOT / "models"
weights_dir.mkdir(exist_ok=True)

url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
dest = weights_dir / "sam_vit_h_4b8939.pth"

urllib.request.urlretrieve(url, dest)

print("Downloaded to:", dest)