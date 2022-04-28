import stat
from pathlib import Path
from maprfanalysis import bbo
from maprfanalysis.figures import *

fits = select_fits(bbo.fits)
logz = compute_logz(fits)
prob = compute_prob(logz)

fits = fits.query('dur==4')\
           .join(prob.query('dur==dur.max()')
                     .idxmax(axis=1)
                     .rename('best')
                     .droplevel('dur'),
                 on='block_id')\
           .replace(model_ids)\
           .query('model_id==best')\
           .reset_index()\
           .drop(columns='best')\
           .fit_id

outpath = Path(bbo.__file__).parent / 'scripts' / 'compute-snr.sh'
template = 'python -m maprfanalysis.bbo compute-snr --keep=0.95 {fit_id}\n'
with outpath.open('w') as script:
    for fit_id in fits:
        script.write(template.format(fit_id=fit_id))
outpath.chmod(outpath.stat().st_mode | stat.S_IEXEC)