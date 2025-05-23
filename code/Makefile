KOKKOS_PATH = ${PSCRATCH}/Kokkos
KOKKOS_DEVICES = "Cuda,OpenMP"
EXE_NAME = "tricount"

SRC = $(wildcard *.cpp)

default: build
	echo "Start Build"

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper 
EXE = ${EXE_NAME}.cuda
KOKKOS_ARCH = "Ampere80"
KOKKOS_CUDA_OPTIONS = "enable_lambda"
else
CXX = g++
EXE = ${EXE_NAME}.host
KOKKOS_ARCH = "BDW"
endif


# ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
# CXX = ${KOKKOS_PATH}/bin/nvcc_wrapper
# EXE = ${EXE_NAME}.cuda
# KOKKOS_ARCH = "Ampere80"
# KOKKOS_CUDA_OPTIONS = "enable_lambda"
# # added flags to allow openmp in the host device in .cuda executable
# # NVCC_WRAPPER_DEFAULT_COMPILER = g++
# # KOKKOS_CXXFLAGS += -Xcompiler -fopenmp
# EXTRA_LIBS += -lgomp
# else
# CXX = g++
# EXE = ${EXE_NAME}.host
# KOKKOS_ARCH = "BDW"
# # added these just in case. They are probably not necessary
# # KOKKOS_CXXFLAGS += -fopenmp
# EXTRA_LIBS += -lgomp
# endif

CXXFLAGS = -O3
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

OBJ = $(SRC:.cpp=.o)
LIB =

include $(KOKKOS_PATH)/Makefile.kokkos

build: $(EXE)

$(EXE): $(OBJ) $(KOKKOS_LINK_DEPENDS)
	$(LINK) $(KOKKOS_LDFLAGS) $(LINKFLAGS) $(EXTRA_PATH) $(OBJ) $(KOKKOS_LIBS) $(LIB) -o $(EXE)

clean: kokkos-clean
	rm -f *.o *.cuda *.host

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

test: $(EXE)
	./$(EXE)
