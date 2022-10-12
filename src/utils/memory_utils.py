import resource

def limit_CPU_memory(bytes):
	soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
	resource.setrlimit(resource.RLIMIT_DATA, (bytes, hard))
	resource.setrlimit(resource.RLIMIT_AS, (1024*1024*1024*26, hard))
	resource.setrlimit(resource.RLIMIT_RSS, (1024*1024*1024*16, hard))
	#resource.setrlimit(resource.RLIMIT_SWAP, (1024*1024*1024*2, hard))