import numpy as np

def build_anchor_shapes(sizes, ratios):
    #
    anchors = []
    for i, s in enumerate(sizes):
        for r in ratios[i]:
            r2 = np.sqrt(r)
            anchors.append((s*r2, s/r2))
    anchors = np.round(np.array(anchors), 3)

    for a in anchors:
        print '\t{}, {},'.format(a[0], a[1])


if __name__ == '__main__':
    #
    sizes = [2, 3, 6, 10, 20, 28]
    ratios = [[1.0/3.0, 0.5, 1.0, 2.0, 3.0] for _ in sizes]
    ratios[-1] = [0.5, 1.0, 2.0]

    build_anchor_shapes(sizes, ratios)
