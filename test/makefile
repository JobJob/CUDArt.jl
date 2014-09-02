all: vadd.ptx

vadd.ptx: vadd.cu
	nvcc -ptx -gencode=arch=compute_30,code=sm_30 vadd.cu

.PHONY: clean
clean:
	$(RM) vadd.ptx
