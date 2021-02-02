CC = @g++
CCFLAGS = -Isrc -Isrc/util -Isrc/gpu_required -Isrc/gpu_extra_credit -ggdb3 -Wall -Wno-write-strings

NVCC = @/usr/local/cuda-11.0/bin/nvcc
NVCCFLAGS = -Isrc -Isrc/util -Isrc/gpu_required -Isrc/gpu_given -Isrc/gpu_extra_credit -arch=sm_35 -Wno-deprecated-gpu-targets

rm = @rm
ifeq ($(OS),Windows_NT)
	rm = @del
endif

default: chmod objs match

debug: CCFLAGS += -O0 -ggdb0
debug: NVCCFLAGS += -g -G -O0
debug: default

chmod:
	@chmod 700 .
	@chmod 700 src

OBJs_match = \
	objs/utils.o \
	objs/mmio.o \
	\
	objs/onewaywrapper.o \
	\
	objs/check_handshaking.o \
	objs/markFilterEdges.o \
	objs/exclusive_prefix_sum.o \
	objs/packGraph.o \
	\


objs:
	@chmod 700 .
	@mkdir -p objs

objs/utils.o: src/util/utils.cpp src/util/utils.hpp
	${CC} ${CCFLAGS} -o $@ -c src/util/utils.cpp

objs/mmio.o: src/util/mmio.cpp src/util/mmio.hpp
	${CC} ${CCFLAGS} -o $@ -c src/util/mmio.cpp
	

objs/onewaywrapper.o: src/onewaywrapper.cu src/onewaywrapper.hpp src/gpu_extra_credit/extra.cu
	${NVCC} ${NVCCFLAGS} -o $@ -c src/onewaywrapper.cu
	

objs/check_handshaking.o: src/gpu_required/check_handshaking.cu
	${NVCC} ${NVCCFLAGS} -o $@ -c src/gpu_required/check_handshaking.cu
	
objs/markFilterEdges.o: src/gpu_required/markFilterEdges.cu
	${NVCC} ${NVCCFLAGS} -o $@ -c src/gpu_required/markFilterEdges.cu
	
objs/exclusive_prefix_sum.o: src/gpu_required/exclusive_prefix_sum.cu
	${NVCC} ${NVCCFLAGS} -o $@ -c src/gpu_required/exclusive_prefix_sum.cu

objs/packGraph.o: src/gpu_given/packGraph.cu
	${NVCC} ${NVCCFLAGS} -o $@ -c src/gpu_given/packGraph.cu
	
	
match: ${OBJs_match} src/main.cu
	@chmod 700 .
	${NVCC} ${NVCCFLAGS} ${OBJs_match} src/main.cu -o $@
	

submit: $(wildcard src/gpu_*/*)
	@chmod 700 .
	tar cvf submit_me.tar src/onewaywrapper.cu src/gpu_required/* src/gpu_extra_credit/extra.cu
	

clean:
	rm -rf match submit_me.tar objs/*.o

