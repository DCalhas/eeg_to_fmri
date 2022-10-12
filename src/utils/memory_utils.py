import resource

def limit_CPU_memory(bytes):
	soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
	resource.setrlimit(resource.RLIMIT_DATA, (bytes, hard))#heap of the process
	resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024*26, hard))#virtual memory allocation
	resource.setrlimit(resource.RLIMIT_RSS, (1024*1024*1024*8, 1024*1024*1024*10))#resident memory
	resource.setrlimit(resource.RLIMIT_NPROC, (4,4))
	#resource.setrlimit(resource.RLIMIT_SWAP, (1024*1024*1024*2, hard))