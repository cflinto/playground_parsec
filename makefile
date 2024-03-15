# Set the compiler
#CC = gcc
NVCC = nvcc
# PTGCC=/usr/local/bin/parsec-ptgpp
PTGCC=/home/cflint/parsec_project/parsec/build/parsec/interfaces/ptg/ptg-compiler/parsec-ptgpp

# For some reason, the .c file is in build/parsec/include on plafrim
# PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/home/cflint/parsec_project/parsec/build/parsec/include
# Set compiler flags and paths
CFLAGS = -g `pkg-config --cflags parsec`
NVCCFLAGS = -arch=sm_80
LDFLAGS = `pkg-config --libs parsec`

# Set library paths and flags for CUDA
CUDA_LIB_DIR = /usr/local/cuda/lib64
CUDA_LIBS = -lcudart -lcublas
CUDA_INCLUDES = -I/usr/local/cuda/include

PARSEC_DIR = /home/cflint/parsec_project/parsec
PARSEC_LIBS = -L$(PARSEC_DIR)/build -L$(PARSEC_DIR)/build/parsec
PARSEC_INCLUDES = -I$(PARSEC_DIR)/build/parsec/include -I$(PARSEC_DIR)/parsec/include -I$(PARSEC_DIR) -I$(PARSEC_DIR)/build
# PARSEC_LIB_DIR=
# PARSEC_INCLUDES=

# Set the source files and output file
JDF_FILE = LBM.jdf
OUTPUT_FILE = LBM
CU_FILE = LBM.cu
O_FILE = LBM_CU.o
C_FILE = LBM.c
H_FILE = LBM_common.h

# Disable built-in rules
.SUFFIXES:


all: $(OUTPUT_FILE)
	@echo "Done."

# Link the object files to generate the executable
$(OUTPUT_FILE): $(C_FILE) $(O_FILE)
	@echo "Linking object files..."
	# @echo "CFLAGS: $(CFLAGS)"
	$(NVCC) $(CFLAGS) $(O_FILE) $(C_FILE) $(PARSEC_INCLUDES) $(PARSEC_LIBS) -o $(OUTPUT_FILE) $(CUDA_LIBS) $(LDFLAGS)

# Compile the JDF file to generate LBM.c
$(C_FILE): $(JDF_FILE) $(H_FILE)
	@echo "Compiling JDF file..."
	${PTGCC} -i $(JDF_FILE) -o $(OUTPUT_FILE)

# Compile the CUDA source file to generate LBM.o
$(O_FILE): $(CU_FILE) $(H_FILE)
	@echo "Compiling CUDA file..."
	# @echo "CFLAGS: $(CFLAGS)"
	$(NVCC) $(CFLAGS) $(CUDA_INCLUDES) $(PARSEC_INCLUDES) -c $(CU_FILE) -o $(O_FILE)

clean:
	rm -f $(O_FILE) $(OUTPUT_FILE) $(O_FILE) $(C_FILE) LBM.o
