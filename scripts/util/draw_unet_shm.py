# this shim is outdated and depends on a function that no longer exists, it's only here for historical reasons'

if __name__ == '__main__':
    for variable in dir():
        if not variable.startswith("_"):
            del globals()[variable]
    import os, sys, codecs
    try:
        import larch.pickle as pickle
    except ImportError:
        import pickle as pickle
    try:
        import scripts.util.draw_unet as draw_unet
    except ImportError:
        import draw_unet as draw_unet
    os.chdir("/")
    args = pickle.loads(codecs.decode(sys.argv[1:][0].encode(), "base64"))
    pil_image = draw_unet.draw_unet(args["modelA"], args["modelB"], args["weights"], style=args["style"], show_labels=args["show_labels"])
    sys.stdout.write(codecs.encode(pickle.dumps(pil_image), "base64").decode())

#using the shm

# import scripts.util.draw_unet_shm as draw_unet_shm
# draw_unet_path = os.path.abspath(draw_unet_shm.__file__)

# unet_vis_out = subprocess.run([sys.executable, draw_unet_path, codecs.encode(pickle.dumps({"modelA": "A", "modelB": "B", "weights": slALL, "style": shared.UnetVisualizerStyle.pygal_style, "show_labels": shared.UnetVisualizerStyle.show_labels}), "base64").decode()], env={}, capture_output=True, text=True)
# print(unet_vis_out.stderr)
# unet_vis = pickle.loads(codecs.decode(unet_vis_out.stdout.encode(), "base64"))
