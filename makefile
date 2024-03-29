# Set the compiler
#CC = gcc
NVCC = nvcc
# PTGCC=/usr/local/bin/parsec-ptgpp
PTGCC=/home/cflint/parsec_project/parsec/build/parsec/interfaces/ptg/ptg-compiler/parsec-ptgpp

# For some reason, the .c file is in build/parsec/include on plafrim
# PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/home/cflint/parsec_project/parsec/build/parsec/include
# Set compiler flags and paths
CFLAGS = -O3 `pkg-config --cflags parsec`
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
JDF_FILE_NON_PARAMETRIZED = LBM_non_parametrized.jdf
OUTPUT_FILE = LBM
OUTPUT_FILE_NON_PARAMETRIZED = LBM_non_parametrized
CU_FILE = LBM.cu
O_FILE = LBM_CU.o
C_FILE = LBM.c
C_FILE_NON_PARAMETRIZED = LBM_non_parametrized.c
H_FILE = LBM_common.h


# if READ_HORIZONTAL_SLICES_BLOCK_NUM is not defined, set it to 1024
READ_VERTICAL_SLICES_BLOCK_NUM ?= 256
READ_VERTICAL_SLICES_THREAD_NUM ?= 256

# Disable built-in rules
.SUFFIXES:


all:  $(OUTPUT_FILE) $(OUTPUT_FILE_NON_PARAMETRIZED)
	@echo "Done."

# Link the object files to generate the executable
$(OUTPUT_FILE): $(C_FILE) $(O_FILE)
	@echo "Linking object files..."
	# @echo "CFLAGS: $(CFLAGS)"
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $(O_FILE) $(C_FILE) $(PARSEC_INCLUDES) $(PARSEC_LIBS) -o $(OUTPUT_FILE) $(CUDA_LIBS) $(LDFLAGS) \
		-DREAD_VERTICAL_SLICES_BLOCK_NUM=${READ_VERTICAL_SLICES_BLOCK_NUM} -DREAD_VERTICAL_SLICES_THREAD_NUM=${READ_VERTICAL_SLICES_THREAD_NUM}

# Link the non-parametrized object files to generate the non-parametrized executable
$(OUTPUT_FILE_NON_PARAMETRIZED): $(C_FILE_NON_PARAMETRIZED) $(O_FILE)
	@echo "Linking non-parametrized object files..."
	# @echo "CFLAGS: $(CFLAGS)"
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $(O_FILE) $(C_FILE_NON_PARAMETRIZED) $(PARSEC_INCLUDES) $(PARSEC_LIBS) -o $(OUTPUT_FILE_NON_PARAMETRIZED) $(CUDA_LIBS) $(LDFLAGS)

# Compile the JDF file to generate LBM.c
$(C_FILE): $(JDF_FILE) $(H_FILE)
	@echo "Compiling JDF file..."
	${PTGCC} -i $(JDF_FILE) -o $(OUTPUT_FILE)

# Compile the non-parametrized JDF file to generate LBM_non_parametrized.c
$(C_FILE_NON_PARAMETRIZED): $(JDF_FILE_NON_PARAMETRIZED) $(H_FILE)
	@echo "Compiling non-parametrized JDF file..."
	${PTGCC} -i $(JDF_FILE_NON_PARAMETRIZED) -o $(OUTPUT_FILE_NON_PARAMETRIZED)

# Compile the CUDA source file to generate LBM.o
$(O_FILE): $(CU_FILE) $(H_FILE)
	@echo "Compiling CUDA file..."
	# @echo "CFLAGS: $(CFLAGS)"
	$(NVCC) $(CFLAGS) $(NVCCFLAGS) $(CUDA_INCLUDES) $(PARSEC_INCLUDES) -c $(CU_FILE) -o $(O_FILE) \
		-DREAD_VERTICAL_SLICES_BLOCK_NUM=${READ_VERTICAL_SLICES_BLOCK_NUM} -DREAD_VERTICAL_SLICES_THREAD_NUM=${READ_VERTICAL_SLICES_THREAD_NUM}

clean:
	rm -f $(O_FILE) $(OUTPUT_FILE) $(O_FILE) $(C_FILE) LBM.o $(OUTPUT_FILE_NON_PARAMETRIZED) $(O_FILE_NON_PARAMETRIZED) $(C_FILE_NON_PARAMETRIZED) LBM_non_parametrized.o
