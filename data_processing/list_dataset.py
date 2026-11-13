import h5py
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()

    with h5py.File(args.path, 'r') as f:
        v_lst = []
        npix_lst = []
        ap_set = set()
        vmin, vmax = 1e9, 0
        npix_min, npix_max = 1e9, 0
        for scene_name, scene in f.items():
            h, w = scene['image'].shape[1:3]
            print(scene_name, (h, w))
            v_lst.append(scene.attrs['n_frames'])
            npix_lst.append(h*w)
            ap_set.add(round(w / h, 4))

    print(f"AP set    : {sorted(list(ap_set))}")
    print(f"Min #views: {min(v_lst)}")
    print(f"Avg #views: {sum(v_lst) / len(v_lst):.0f}")
    print(f"Max #views: {max(v_lst)}")
    print(f"Min #pixs : {min(npix_lst)}")
    print(f"Avg #pixs : {sum(npix_lst) / len(v_lst):.0f}")
    print(f"Max #pixs : {max(npix_lst)}")
