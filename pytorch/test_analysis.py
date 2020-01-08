import matplotlib.pyplot as plt
from glob import glob
import numpy as np
from os.path import basename, dirname, join

# resdir = '/data/mike/vm/results_paired_mmi'
def test(resdir):
    npys = glob(join(resdir,'*an.npy'))
    npys = npys+glob(join(resdir,'*og.npy'))
    npys = npys+glob(join(resdir,'*vm.npy'))
    rows = 300# int((len(npys)/3))
    cols = 3
    results = np.zeros((rows,cols))


    for npy in npys:
        # Load npy
        metric = np.mean(np.load(npy))

        # Row
        it = int(basename(npy).split('_')[0])
        row = int(it/500)

        # Col
        type = basename(npy)[-6:-4]
        if type=='an':
            col = 0
        elif type=='og':
            col = 1
        elif type=='vm':
            col = 2

        results[row, col] = metric

    results[results==0]=np.nan
    x = np.linspace(0,150000, rows)
    yan = results[:,0]
    yog = results[:,1]
    yvm = results[:,2]
    # Time to plot
    fig,ax = plt.subplots()
    ax.plot(x,yan, label="Ants")
    ax.plot(x,yog, label="Original")
    ax.plot(x,yvm, label="Voxelmorph")
    plt.legend(bbox_to_anchor=(1.05,1),loc='lower left', borderaxespad=0.)
    fname = join(resdir, basename(resdir)+'.png')
    plt.savefig(fname,bbox_inches='tight')
    print('Saving figure to ' + fname)

    # Find best model
    best = int(np.argmin(yvm) * 500)
    # Run best model on volumes
    bestmodelfn = str(best).zfill(7) + '.ckpt'
    name = basename(resdir)[8:]
    bestmodelfn = join(dirname(resdir),'models_' + name,bestmodelfn)
    print('best model is ' + bestmodelfn)
    return bestmodelfn