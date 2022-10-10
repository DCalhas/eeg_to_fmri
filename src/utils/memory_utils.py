import resource

def limit_CPU_memory(bytes):
	soft, hard = resource.getrlimit(resource.RLIMIT_AS)
	
	resource.setrlimit(resource.RLIMIT_AS, (bytes, 31*1024*1024/4))