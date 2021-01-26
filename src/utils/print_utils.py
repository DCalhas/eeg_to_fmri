




def print_message(m, file_output=None, verbose=False):

	if(verbose):
		if(file_output==None):
			print(m, flush=True)
		else:
			print(m, file=file_output, flush=True)