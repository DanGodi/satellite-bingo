from pathlib import Path
from PIL import Image
from IPython.display import display, clear_output
import json
import ipywidgets as widgets

def show_image(img_path, max_size=(1024, 1024)):
    img = Image.open(img_path)
    img.thumbnail(max_size)
    display(img)

def label_images_interactively(folder, features, out_file, resume=True):
    folder = Path(folder)
    out_file = Path(out_file)
    image_paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp")]
    )
    mapping = {}
    if out_file.exists() and resume:
        with open(out_file, "r") as f:
            try:
                mapping = json.load(f)
            except Exception:
                mapping = {}

    if not image_paths:
        print("No images found:", folder)
        return mapping

    # UI state
    idx = 0
    checkboxes = [widgets.Checkbox(value=False, description=f) for f in features]
    sel_box = widgets.VBox(checkboxes)
    btn_next = widgets.Button(description="Next", button_style="primary")
    btn_clear = widgets.Button(description="Clear", button_style="")
    btn_select_all = widgets.Button(description="Select All")
    btn_skip = widgets.Button(description="Skip")
    btn_quit = widgets.Button(description="Quit", button_style="danger")
    out = widgets.Output()

    def save_mapping():
        with open(out_file, "w") as f:
            json.dump(mapping, f, indent=2)

    def render():
        out.clear_output(wait=True)
        if idx >= len(image_paths):
            with out:
                print("Labeling complete. Mapping saved at:", out_file)
            return
        p = image_paths[idx]
        with out:
            print(f"Image {idx+1}/{len(image_paths)}: {p.name}")
            show_image(p)
        # set checkbox states from existing mapping if any
        existing = set(mapping.get(str(p), []))
        for cb in checkboxes:
            cb.value = cb.description in existing

    def on_next(b):
        nonlocal idx
        p = image_paths[idx]
        selected = [cb.description for cb in checkboxes if cb.value]
        mapping[str(p)] = selected
        save_mapping()
        idx += 1
        if idx >= len(image_paths):
            render()
            return
        render()

    def on_select_all(b):
        for cb in checkboxes:
            cb.value = True

    def on_clear(b):
        for cb in checkboxes:
            cb.value = False

    def on_skip(b):
        nonlocal idx
        p = image_paths[idx]
        mapping[str(p)] = []
        save_mapping()
        idx += 1
        if idx < len(image_paths):
            render()
        else:
            render()

    def on_quit(b):
        out.clear_output(wait=True)
        with out:
            print("Exiting. Mapping saved at:", out_file)

    btn_next.on_click(on_next)
    btn_select_all.on_click(on_select_all)
    btn_clear.on_click(on_clear)
    btn_skip.on_click(on_skip)
    btn_quit.on_click(on_quit)

    controls = widgets.HBox([btn_next, btn_select_all, btn_clear, btn_skip, btn_quit])
    display(widgets.VBox([out, sel_box, controls]))

    # initial render
    render()

    return mapping
