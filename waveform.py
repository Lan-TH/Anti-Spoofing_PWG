import wave
import struct,numpy
from scipy import *
from pylab import *
 
def Plot_fft_freq_chart(filename,plot=False):
	wavefile = wave.open(filename, 'r') # open for writing
	nchannels = wavefile.getnchannels()
	sample_width = wavefile.getsampwidth()
	framerate = wavefile.getframerate()
	numframes = wavefile.getnframes()
	  
	print("channel",nchannels)
	print("sample_width",sample_width)
	print("framerate",framerate)
	print("numframes",numframes)
	  
	y = numpy.zeros(numframes)
	  
	for i in range(numframes):
		val = wavefile.readframes(1)
		left = val[0:2]
		#right = val[2:4]
		v = struct.unpack('h', left )[0]
		y[i] = v
	  
	Fs = framerate
	try:
		data, freqs, bins, im = specgram(y, NFFT=1024, Fs=Fs, noverlap=900)
		mm=data[127]
		mm=10. * np.log10(mm+1e-4)
 
	except Exception as e:
		print("error is: ",e)
		return -50
 
	freq1khz_value=mean(mm)
	print(freq1khz_value)
	if plot:
		show()
	return freq1khz_value
 
 
filename1="vocoder2.wav"
Plot_fft_freq_chart(filename1,True)
