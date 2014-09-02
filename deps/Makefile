all: libwrapcuda.so utils.ptx

libwrapcuda.so: wrapcuda.c
	nvcc --shared --compiler-options=-fPIC \
		-gencode=arch=compute_30,code=sm_30 \
		--linker-options=-fPIC wrapcuda.c -o libwrapcuda.so

utils.ptx: utils.cu
	nvcc -ptx -gencode=arch=compute_30,code=sm_30 utils.cu

.PHONY: clean
clean:
	$(RM) libwrapcuda.so utils.ptx
