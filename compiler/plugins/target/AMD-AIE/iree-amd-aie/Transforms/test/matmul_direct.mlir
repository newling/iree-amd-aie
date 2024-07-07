// RUN: iree-opt --pass-pipeline="builtin.module(func.func(iree-amdaie-matmul-direct))"

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>
#executable_target_amdaie_xclbin_fb = #hal.executable.target<"amd-aie", "amdaie-xclbin-fb", {target_arch = "chip-tbd", ukernels = "none"}>
#packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#pipeline_layout = #hal.pipeline.layout<push_constants = 0, sets = [<0, bindings = [<0, storage_buffer, ReadOnly>, <1, storage_buffer, ReadOnly>, <2, storage_buffer>]>]>
#translation = #iree_codegen.translation_info<Custom>
#device_target_amd_aie = #hal.device.target<"amd-aie", [#executable_target_amdaie_xclbin_fb]>

builtin.module {
 func.func @matmul_i32_dispatch_0_matmul_128x128x256_i32() attributes {translation_info = #translation} {
   %c0_i32 = arith.constant 0 : i32
   %c0 = arith.constant 0 : index
   %0 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<128x256xi32, #hal.descriptor_type<storage_buffer>>
   memref.assume_alignment %0, 64 : memref<128x256xi32, #hal.descriptor_type<storage_buffer>>
   %1 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<256x128xi32, #hal.descriptor_type<storage_buffer>>
   memref.assume_alignment %1, 64 : memref<256x128xi32, #hal.descriptor_type<storage_buffer>>
   %2 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<128x128xi32, #hal.descriptor_type<storage_buffer>>
   memref.assume_alignment %2, 64 : memref<128x128xi32, #hal.descriptor_type<storage_buffer>>
   linalg.fill ins(%c0_i32 : i32) outs(%2 : memref<128x128xi32, #hal.descriptor_type<storage_buffer>>)
   linalg.matmul {lowering_config = #config, packing_config = #packingConfig} ins(%0, %1 : memref<128x256xi32, #hal.descriptor_type<storage_buffer>>, memref<256x128xi32, #hal.descriptor_type<storage_buffer>>) outs(%2 : memref<128
2, #hal.descriptor_type<storage_buffer>>)
   return
  }
}
