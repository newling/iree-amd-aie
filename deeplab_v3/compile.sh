# Usage: ./compile.sh <intermediate_fn> <output_fn>
#
# Print commands being run, and fail immediately on error:
set -ex

if [ "$#" -ne 3 ]; then
    echo "Usage: ./compile.sh <iree_build_dir> <torch_onnx_fn> <output_dir>"
    exit 1
fi

# iree build directory
iree_build_dir=$1
iree_build_dir=$(realpath $iree_build_dir)

# deeplabv3 torch onnx model, available currently at https://github.com/nod-ai/npu-benchmark/blob/main/deeplabv3.torch-onnx.mlir
torch_onnx_fn=$2
torch_onnx_fn=$(realpath $torch_onnx_fn)

output_dir=$3
output_dir=$(realpath $output_dir)

# Make output_dir if it doesn't exist:
mkdir -p $output_dir

# Search for iree-compile and iree-e2e-matmul-test in the user provided directory.
iree_compile=""
for dir in "${iree_build_dir}" "${iree_build_dir}/bin" "${iree_build_dir}/tools"; do
  if [ -f "${dir}/iree-compile" ]; then
    iree_compile="${dir}/iree-compile"
  fi
done



# These will be generated:
hal_ir_fn=${output_dir}/hal_ir.mlir
dispatch_dir=${output_dir}/executable_sources
intermediate_fn=${output_dir}/intermediate.mlir
output_fn=${output_dir}/output.mlir

# Run iree-opt and save the output to the temporary file
iree-opt --convert-torch-onnx-to-torch --torch-lower-to-backend-contract --canonicalize --cse --torch-backend-to-linalg-on-tensors-backend-pipeline "${torch_onnx_fn}" > "${intermediate_fn}"

# These probably need adjusting.
compilation_flags=" \
 --iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false \
 --mlir-print-ir-after=iree-flow-form-dispatch-regions \
 --iree-flow-inline-constants-max-byte-length=0 \
 --mlir-elide-elementsattrs-if-larger=16 \
 --iree-llvmcpu-fail-on-large-vector=0 \
 --iree-flow-enable-aggressive-fusion \
 --iree-hal-target-backends=llvm-cpu \
 --iree-input-demote-i64-to-i32 \
 --mlir-disable-threading \
 --compile-to=hal \
 --iree-hal-dump-executable-sources-to=${dispatch_dir}"

${iree_compile} ${intermediate_fn} ${compilation_flags}  > ${hal_ir_fn} 2>${output_fn}

# # Run iree-compile with the temporary file as input
# iree-compile "$intermediate_fn" --iree-input-demote-i64-to-i32 --iree-hal-target-backends=llvm-cpu --iree-llvmcpu-fail-on-large-vector=0 --mlir-print-ir-after=iree-flow-form-dispatch-regions --iree-llvmcpu-stack-allocation-limit=100000 -o deeplabv3_cpu.mvfb > "$output_fn" 2>&1

