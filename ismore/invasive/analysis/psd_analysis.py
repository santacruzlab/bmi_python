from db.tracker import models
import nitime.algorithms as tsa
import glob
import scipy.signal

def parse_br(folder):
	files = glob.glob(folder+'/*.ns6')
	for f in files:
		#models.parse_blackrock_file(None, [f], None) 

		# Channels 1-5:
		hdf = tables.openFile(f+'.hdf')
		td = hdf.root.channel.channel00002.Value[:][:60000]

		f1, psd =scipy.signal.welch(td, nperseg=30000)
		plt.plot(f1*30000, np.log10(psd), label=f)
		input('Continue?')
	plt.legend()