import resource

def limit_CPU_memory(bytes):
	soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
	resource.setrlimit(resource.RLIMIT_DATA, (bytes, hard))