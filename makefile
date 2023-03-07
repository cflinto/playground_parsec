# Set the compiler
#CC = gcc
NVCC = nvcc

# Set compiler flags and paths
CFLAGS = -g `pkg-config --cflags parsec`
LDFLAGS = `pkg-config --libs parsec`
PKG_CONFIG_PATH := $(PKG_CONFIG_PATH):/usr/local/lib/pkgconfig

# Set library paths and flags for CUDA
CUDA_LIB_DIR = /usr/local/cuda/lib64
CUDA_LIBS = -lcudart -lcublas
CUDA_INCLUDES = -I/usr/local/cuda/include

# Set the source files and output file
JDF_FILE = LBM.jdf
OUTPUT_FILE = LBM
CU_FILE = LBM.cu
O_FILE = LBM_CU.o
C_FILE = LBM.c
H_FILE = LBM_common.h

all: $(OUTPUT_FILE)
	@echo "Done."

# Link the object files to generate the executable
$(OUTPUT_FILE): $(C_FILE) $(O_FILE)
	@echo "Linking object files..."
	$(NVCC) $(CFLAGS) $(O_FILE) $(C_FILE) -o $(OUTPUT_FILE) -L$(CUDA_LIB_DIR) $(CUDA_LIBS) $(LDFLAGS)

# Compile the JDF file to generate LBM.c
$(C_FILE): $(JDF_FILE) $(H_FILE)
	@echo "Compiling JDF file..."
	/usr/local/bin/parsec-ptgpp -i $(JDF_FILE) -o $(OUTPUT_FILE)

# Compile the CUDA source file to generate LBM.o
$(O_FILE): $(CU_FILE) $(H_FILE)
	@echo "Compiling CUDA file..."
	$(NVCC) $(CFLAGS) $(CUDA_INCLUDES) -c $(CU_FILE) -o $(O_FILE)

clean:
	rm -f $(O_FILE) $(OUTPUT_FILE) $(O_FILE) $(C_FILE) LBM.o
