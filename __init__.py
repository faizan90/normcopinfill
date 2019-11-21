import matplotlib as mpl
mpl.rc('font', size=14)

from matplotlib import use as plt_use
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

plt_use('AGG')

plt.ioff()
plt.rc('axes', axisbelow=True)
register_matplotlib_converters()

from .core.infill import NormCopulaInfill
