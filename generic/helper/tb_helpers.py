import io
import PIL
import torchvision

def pltfig_to_tensor(fig, format="jpeg", dpi=100, **kwargs):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, **kwargs)
    buf.seek(0)
    image = PIL.Image.open(buf)
    return torchvision.transforms.ToTensor()(image)