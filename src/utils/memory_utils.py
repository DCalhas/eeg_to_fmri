import resource

def limit_CPU_memory(bytes):
	soft, hard = resource.getrlimit(resource.RLIMIT_AS)
	print(hard)
	print(soft)
	resource.setrlimit(resource.RLIMIT_AS, (bytes, hard))