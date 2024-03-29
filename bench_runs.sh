#!/bin/sh


is_on_cluster=0

job_id=$SLURM_JOB_ID
if [ -z "$job_id" ] ; then
	echo "Warning: SLURM_JOB_ID is not set, using 42 instead"
	job_id=42
else
	is_on_cluster=1
	user_alias=`pwd | cut -d/ -f3`
fi

root_directory=`pwd`
runs_directory="$root_directory/jobs/runs"
job_output_directory="$runs_directory/$job_id"
test_app_source_directory="$root_directory/playground_parsec"
running_script_path="$test_app_source_directory/bench_runs.sh"

# Load modules (only if on a cluster)
if [ "$is_on_cluster" -eq 1 ] ; then
	source "$test_app_source_directory/modules.sh"
fi

if [ -d "$job_output_directory" ] ; then
	echo "folder $job_output_directory already exists, aborting" 1>&2
	exit
fi

mkdir -p "$job_output_directory"

job_log_file="$job_output_directory/log"
job_config_file="$job_output_directory/config"
touch "$job_log_file"
touch "$job_config_file"

t2000_num=`nvidia-smi -L | grep T2000 | wc -l`
k40m_num=`nvidia-smi -L | grep K40m | wc -l`
v100_num=`nvidia-smi -L | grep V100 | wc -l`
p100_num=`nvidia-smi -L | grep P100 | wc -l`
a100_num=`nvidia-smi -L | grep A100 | wc -l`

# We expect at least one GPU
if [ "$is_on_cluster" -eq 1 ] ; then
	if [ `expr "$t2000_num" + "$k40m_num" + "$v100_num" + "$p100_num" + "$a100_num"` -eq 0 ] ; then
		echo "Error: no GPU detected" 1>&2
		exit
	fi
fi

arch='unknown'
target_architecture='UNKNOWN'
vram_size_gb=-1

if [ "$t2000_num" -gt 0 ] ; then
	arch='t2000'
	maxgpu=1
	target_architecture="T2000"
	vram_size_gb=1
	vram_size_b="((long)(((size_t)1)<<30)*(long)$vram_size_gb)"
fi
if [ "$k40m_num" -gt 0 ] ; then
	arch='k40m'
	maxgpu=1
	target_architecture="K40M"
	vram_size_gb=11
	vram_size_b="((long)(((size_t)1)<<30)*(long)$vram_size_gb)"
fi
if [ "$v100_num" -gt 0 ] ; then
	arch='v100'
	maxgpu=2
	target_architecture="V100"
	vram_size_gb=15
	vram_size_b="((long)(((size_t)1)<<30)*(long)$vram_size_gb)"
fi
if [ "$p100_num" -gt 0 ] ; then
	arch='p100'
	maxgpu=2
	target_architecture="P100"
	vram_size_gb=15
	vram_size_b="((long)(((size_t)1)<<30)*(long)$vram_size_gb)"
fi
if [ "$a100_num" -gt 0 ] ; then
	arch='a100'
	maxgpu=2
	target_architecture="A100"
	vram_size_gb=38
	vram_size_b="((long)(((size_t)1)<<30)*(long)$vram_size_gb)"
fi

cd "$job_output_directory"

echo 't2000_num='$t2000_num >> $job_config_file
echo 'k40m_num='$k40m_num >> $job_config_file
echo 'v100_num='$v100_num >> $job_config_file
echo 'p100_num='$p100_num >> $job_config_file
echo 'a100_num='$a100_num >> $job_config_file
echo 'arch='$arch >> $job_config_file
echo 'hostname='$HOSTNAME >> $job_config_file
echo 'job_id='$job_id >> $job_config_file
echo 'node_name='$SLURMD_NODENAME >> $job_config_file
echo 'lscpu :' >> $job_config_file
lscpu >> $job_config_file
echo '#########################################' >> $job_config_file
echo 'nvidia-smi :' >> $job_config_file
nvidia-smi >> $job_config_file


date_begin=`date +%s`
echo "Starting the experiments at $date_begin" >> $job_log_file

echo "Path to the script that is running: $running_script_path" >> $job_log_file
echo "Content of the script that is running:" >> $job_log_file
echo "################################################################################" >> $job_log_file
echo "(content of the script begins here)" >> $job_log_file
cat "$running_script_path" >> $job_log_file
echo "(content of the script ends here)" >> $job_log_file
echo "################################################################################" >> $job_log_file


PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/home/cflint/parsec_project/parsec/build/parsec/include
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/cflint/parsec_project/parsec/build/parsec

export PKG_CONFIG_PATH
export LD_LIBRARY_PATH

# Perform all the combinations thread_num x block_num
possible_thread_nums="64 128 192 256"
possible_block_nums="256 512 1024 2048"

# Generate the list of all the combinations
# We will use the following format: "thread_num block_num"
# The list will be separated by a newline character
# The list will be stored in the variable "combinations"
combinations=""
for thread_num in $possible_thread_nums ; do
	for block_num in $possible_block_nums ; do
		combinations="$combinations$thread_num $block_num\n"
	done
done

# Experiments are integers zxy, where x is the overlap_x and y is the step_kernel_version, and z is for the multiple runs of the same configuration
# Hence, we go like this: 10, 11, 12, 20, 21, 22, 30, 31, 32, , 40, 41, 42, ..., 320, 321, 322
for experiment_configuration in `seq 10 32322` ;
do
	# if mod 10 is not 0, 1, or 2, then we skip
	if [ `expr $experiment_configuration % 10` -gt 2 ] ; then
		continue
	fi
	# The format is zzxxy
	# If xx is more than 32, we skip
	if [ `expr $experiment_configuration % 1000` -gt 322 ] ; then
		continue
	fi
	# If zz is more than 31
	if [ `expr $experiment_configuration / 1000` -gt 31 ] ; then
		continue
	fi

	experiment_directory="$job_output_directory/experiment_$experiment_configuration"
	experiment_log_file="$experiment_directory/log"

	mkdir -p "$experiment_directory"
	touch "$experiment_log_file"

	echo "Starting experiment $experiment_configuration" >> $job_log_file

	echo "Starting experiment $experiment_configuration" >> $experiment_log_file

	execution_output_file="$job_output_directory/experiment_$experiment_configuration/output.txt"

	touch "$execution_output_file"

    # Set the parameters of this experiment:
	# overlap_x = (experiment_configuration/10)%100
	overlap_x=`expr $experiment_configuration / 10 % 100`
	step_kernel_version=`expr $experiment_configuration % 10`

	echo "overlap_x=$overlap_x" >> $experiment_log_file
	echo "step_kernel_version=$step_kernel_version" >> $experiment_log_file




	# Compile the application (make clean, then make)
	cd "$test_app_source_directory"
	make clean
	# The first make always fails, so we run two makes
	make || make

    cd "$experiment_directory"

    # Run the application
    echo "Running the application" >> $experiment_log_file
    echo "Running the application" >> $job_log_file
    
    $test_app_source_directory/LBM $overlap_x $step_kernel_version > "$execution_output_file" 2>&1
done


date_end=`date +%s`
echo "Ending the experiments at $date_end" >> $job_log_file
echo "Total time: `expr $date_end - $date_begin` seconds" >> $job_log_file
