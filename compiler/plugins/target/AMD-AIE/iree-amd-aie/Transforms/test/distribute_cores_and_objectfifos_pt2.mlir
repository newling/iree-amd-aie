// RUN: iree-opt --pass-pipeline="builtin.module(iree-amdaie-distribute-cores-and-objectfifos,cse)" --split-input-file --verify-diagnostics %s | FileCheck %s


// -----

// expected-error @+below {{Expected only one user of the compute op}}

#config = #iree_codegen.lowering_config<tile_sizes = [[64, 64], [0, 0, 1], [1, 1, 0, 0, 0, 0]]>
#map = affine_map<(d0) -> (d0 * 64)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
#map4 = affine_map<(d0) -> (d0 * 32 + 32)>
#packingConfig = #amdaie.packing_config<packing_config = [{packedSizes = [32, 32, 32], transposePackIndices = [1], unpackEmpty = [false], innerPerm = [[1, 0]], outerPerm = [[0, 1]]}, {packedSizes = [0, 0, 0, 4, 4, 8], transposePackIndices = [0, 1, 2], unpackEmpty = [false, false, true], innerPerm = [[0, 1], [1, 0], [0, 1]], outerPerm = [[0, 1, 3, 2], [0, 1, 3, 2], [0, 1, 3, 2]]}]>
#translation = #iree_codegen.translation_info<Custom>
module {
  func.func @matmul_int32_dispatch_0_matmul_128x128x256_i32() attributes {translation_info = #translation} {
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c224 = arith.constant 224 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index
    %c128 = arith.constant 128 : index
    %c4096 = arith.constant 4096 : index
    %c2048 = arith.constant 2048 : index
    %c256 = arith.constant 256 : index
    %c8192 = arith.constant 8192 : index
    %c1024 = arith.constant 1024 : index
    %c32 = arith.constant 32 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
    %alloc_0 = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
    %alloc_1 = memref.alloc() : memref<1x2x32x32xi32, 1 : i32>
    %0 = amdaie.logicalobjectfifo.from_memref %alloc_1, {} : memref<1x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>
    %alloc_2 = memref.alloc() : memref<2x1x32x32xi32, 1 : i32>
    %1 = amdaie.logicalobjectfifo.from_memref %alloc_2, {} : memref<2x1x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>
    %alloc_3 = memref.alloc() : memref<2x2x8x8x4x4xi32, 2 : i32>
    %alloc_4 = memref.alloc() : memref<2x2x32x32xi32, 1 : i32>
    %2 = amdaie.logicalobjectfifo.from_memref %alloc_4, {} : memref<2x2x32x32xi32, 1 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>
    %3 = hal.interface.binding.subspan set(0) binding(0) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<128x256xi32>
    %4 = amdaie.logicalobjectfifo.from_memref %3, {} : memref<128x256xi32> -> !amdaie.logicalobjectfifo<memref<128x256xi32>>
    memref.assume_alignment %3, 64 : memref<128x256xi32>
    %5 = hal.interface.binding.subspan set(0) binding(1) type(storage_buffer) alignment(64) offset(%c0) flags(ReadOnly) : memref<256x128xi32>
    %6 = amdaie.logicalobjectfifo.from_memref %5, {} : memref<256x128xi32> -> !amdaie.logicalobjectfifo<memref<256x128xi32>>
    memref.assume_alignment %5, 64 : memref<256x128xi32>
    %7 = hal.interface.binding.subspan set(0) binding(2) type(storage_buffer) alignment(64) offset(%c0) : memref<128x128xi32>
    %8 = amdaie.logicalobjectfifo.from_memref %7, {} : memref<128x128xi32> -> !amdaie.logicalobjectfifo<memref<128x128xi32>>
    memref.assume_alignment %7, 64 : memref<128x128xi32>
    scf.forall (%arg0, %arg1) in (2, 2) {
      %9 = affine.apply #map(%arg1)
      %10 = affine.apply #map(%arg0)
      %11 = amdaie.dma_cpy_nd(%1[%c0, %c0, %c0, %c0] [%c2, %c1, %c32, %c32] [%c1024, %c1024, %c32, %c1], %4[%c0, %c0, %10, %c0] [%c2, %c1, %c32, %c32] [%c8192, %c32, %c256, %c1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x256xi32>>)
      %12 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c2, %c32, %c32] [%c2048, %c1024, %c32, %c1], %6[%c0, %c0, %c0, %9] [%c1, %c2, %c32, %c32] [%c4096, %c32, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
      %13 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
      %14 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
      scf.forall (%arg2, %arg3) in (2, 2) {
        %19 = amdaie.dma_cpy_nd(%14[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c256, %c32, %c8, %c1], %1[%arg2, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c8, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
        %20 = amdaie.dma_cpy_nd(%13[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c128, %c32, %c4, %c1], %0[%c0, %arg3, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c4, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
        %subview = memref.subview %alloc_3[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
        %21 = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %21)
        %22 = amdaie.core(%tile) {
          amdaie.logicalobjectfifo.consume(%19)
          amdaie.logicalobjectfifo.consume(%20)
          linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>)
         //  linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) attrs =  {lowering_config = #config, packing_config = #packingConfig} {
         //  ^bb0(%in: i32, %in_5: i32, %out: i32):
         //    %23 = arith.muli %in, %in_5 : i32
         //    %24 = arith.addi %out, %23 : i32
         //    linalg.yield %24 : i32
         //  }
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      // scf.for %arg2 = %c0 to %c6 step %c1 {
      scf.for %arg2 = %c0 to %c7 step %c1 {
        %19 = affine.apply #map4(%arg2)
        %20 = amdaie.dma_cpy_nd(%1[%c0, %c0, %c0, %c0] [%c2, %c1, %c32, %c32] [%c1024, %c1024, %c32, %c1], %4[%c0, %c0, %10, %19] [%c2, %c1, %c32, %c32] [%c8192, %c32, %c256, %c1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x256xi32>>)
        %21 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c2, %c32, %c32] [%c2048, %c1024, %c32, %c1], %6[%c0, %c0, %19, %9] [%c1, %c2, %c32, %c32] [%c4096, %c32, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
        scf.forall (%arg3, %arg4) in (2, 2) {
          %22 = amdaie.dma_cpy_nd(%14[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c256, %c32, %c8, %c1], %1[%arg3, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c8, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
          %23 = amdaie.dma_cpy_nd(%13[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c128, %c32, %c4, %c1], %0[%c0, %arg4, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c4, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
          %subview = memref.subview %alloc_3[%arg3, %arg4, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
          %24 = arith.addi %arg3, %c2 : index
          %tile = amdaie.tile(%arg4, %24)
          %25 = amdaie.core(%tile) {
            amdaie.logicalobjectfifo.consume(%22)
            amdaie.logicalobjectfifo.consume(%23)
            linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) attrs =  {lowering_config = #config, packing_config = #packingConfig} {
            ^bb0(%in: i32, %in_5: i32, %out: i32):
              %26 = arith.muli %in, %in_5 : i32
              %27 = arith.addi %out, %26 : i32
              linalg.yield %27 : i32
            }
            amdaie.end
          }
        } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      }
      %15 = amdaie.dma_cpy_nd(%1[%c0, %c0, %c0, %c0] [%c2, %c1, %c32, %c32] [%c1024, %c1024, %c32, %c1], %4[%c0, %c0, %10, %c224] [%c2, %c1, %c32, %c32] [%c8192, %c32, %c256, %c1]) : (!amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<128x256xi32>>)
      %16 = amdaie.dma_cpy_nd(%0[%c0, %c0, %c0, %c0] [%c1, %c2, %c32, %c32] [%c2048, %c1024, %c32, %c1], %6[%c0, %c0, %c224, %9] [%c1, %c2, %c32, %c32] [%c4096, %c32, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<256x128xi32>>)
      %17 = amdaie.logicalobjectfifo.from_memref %alloc_3, {} : memref<2x2x8x8x4x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<2x2x8x8x4x4xi32, 2 : i32>>
      scf.forall (%arg2, %arg3) in (2, 2) {
        %19 = amdaie.dma_cpy_nd(%14[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c256, %c32, %c8, %c1], %1[%arg2, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c4, %c8, %c4, %c8] [%c1024, %c1024, %c8, %c128, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<2x1x32x32xi32, 1 : i32>>)
        %20 = amdaie.dma_cpy_nd(%13[%c0, %c0, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c1024, %c1024, %c128, %c32, %c4, %c1], %0[%c0, %arg3, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c4, %c256, %c32, %c1]) : (!amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>, !amdaie.logicalobjectfifo<memref<1x2x32x32xi32, 1 : i32>>)
        %subview = memref.subview %alloc_3[%arg2, %arg3, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
        %21 = amdaie.dma_cpy_nd(%2[%arg2, %arg3, %c0, %c0] [%c1, %c1, %c32, %c32] [%c2048, %c1024, %c32, %c1], %17[%arg2, %arg3, %c0, %c0, %c0, %c0] [%c1, %c1, %c8, %c4, %c8, %c4] [%c2048, %c1024, %c16, %c4, %c128, %c1]) : (!amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>, !amdaie.logicalobjectfifo<memref<2x2x8x8x4x4xi32, 2 : i32>>)
        %22 = arith.addi %arg2, %c2 : index
        %tile = amdaie.tile(%arg3, %22)
        %23 = amdaie.core(%tile) {
          amdaie.logicalobjectfifo.consume(%19)
          amdaie.logicalobjectfifo.consume(%20)
          linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%alloc_0, %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>) outs(%subview : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) attrs =  {lowering_config = #config, packing_config = #packingConfig} {
          ^bb0(%in: i32, %in_5: i32, %out: i32):
            %24 = arith.muli %in, %in_5 : i32
            %25 = arith.addi %out, %24 : i32
            linalg.yield %25 : i32
          }
          amdaie.logicalobjectfifo.produce(%21)
          amdaie.end
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      %18 = amdaie.dma_cpy_nd(%8[%10, %9] [%c64, %c64] [%c128, %c1], %2[%c0, %c0, %c0, %c0] [%c2, %c32, %c2, %c32] [%c2048, %c32, %c1024, %c1]) : (!amdaie.logicalobjectfifo<memref<128x128xi32>>, !amdaie.logicalobjectfifo<memref<2x2x32x32xi32, 1 : i32>>)
    } {mapping = [#gpu.block<y>, #gpu.block<x>]}
    memref.dealloc %alloc_4 : memref<2x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_3 : memref<2x2x8x8x4x4xi32, 2 : i32>
    memref.dealloc %alloc_2 : memref<2x1x32x32xi32, 1 : i32>
    memref.dealloc %alloc_1 : memref<1x2x32x32xi32, 1 : i32>
    memref.dealloc %alloc_0 : memref<1x1x4x8x4x8xi32, 2 : i32>
    memref.dealloc %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
    return
  }
}

// Command Output (stderr):
// --
// + iree-opt '--pass-pipeline=builtin.module(iree-amdaie-distribute-cores-and-objectfifos,cse)' --split-input-file --verify-diagnostics /proj/gdba/jamesn/workspace/iree-amd-aie/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test/distribute_cores_and_objectfifos_pt2.mlir
// + FileCheck /proj/gdba/jamesn/workspace/iree-amd-aie/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test/distribute_cores_and_objectfifos_pt2.mlir
// error: no check strings found with prefix 'CHECK:'
// within split at /proj/gdba/jamesn/workspace/iree-amd-aie/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test/distribute_cores_and_objectfifos_pt2.mlir:4 offset :57:13: error: unexpected error: 'amdaie.logicalobjectfifo.from_memref' op found logical objectfifo on local memory space with no tiles assigned.
//       %14 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
//             ^
// within split at /proj/gdba/jamesn/workspace/iree-amd-aie/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test/distribute_cores_and_objectfifos_pt2.mlir:4 offset :56:13: error: unexpected error: 'amdaie.logicalobjectfifo.from_memref' op found logical objectfifo on local memory space with no tiles assigned.
//       %13 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
//             ^
// within split at /proj/gdba/jamesn/workspace/iree-amd-aie/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test/distribute_cores_and_objectfifos_pt2.mlir:4 offset :57:13: error: unexpected error: 'amdaie.logicalobjectfifo.from_memref' op found logical objectfifo on local memory space with no tiles assigned.
//       %14 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
//             ^
// within split at /proj/gdba/jamesn/workspace/iree-amd-aie/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test/distribute_cores_and_objectfifos_pt2.mlir:4 offset :56:13: error: unexpected error: 'amdaie.logicalobjectfifo.from_memref' op found logical objectfifo on local memory space with no tiles assigned.
//       %13 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
//             ^
// within split at /proj/gdba/jamesn/workspace/iree-amd-aie/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test/distribute_cores_and_objectfifos_pt2.mlir:4 offset :57:13: error: unexpected error: 'amdaie.logicalobjectfifo.from_memref' op found logical objectfifo on local memory space with no tiles assigned.
//       %14 = amdaie.logicalobjectfifo.from_memref %alloc_0, {} : memref<1x1x4x8x4x8xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x4x8x4x8xi32, 2 : i32>>
//             ^
// within split at /proj/gdba/jamesn/workspace/iree-amd-aie/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test/distribute_cores_and_objectfifos_pt2.mlir:4 offset :56:13: error: unexpected error: 'amdaie.logicalobjectfifo.from_memref' op found logical objectfifo on local memory space with no tiles assigned.
//       %13 = amdaie.logicalobjectfifo.from_memref %alloc, {} : memref<1x1x8x4x8x4xi32, 2 : i32> -> !amdaie.logicalobjectfifo<memref<1x1x8x4x8x4xi32, 2 : i32>>
//             ^
// iree-opt: iree/third_party/llvm-project/llvm/include/llvm/ADT/SmallVector.h:304: reference llvm::SmallVectorTemplateCommon<mlir::iree_compiler::AMDAIE::TileOp>::operator[](size_type) [T = mlir::iree_compiler::AMDAIE::TileOp]: Assertion `idx < size()' failed.
// Please report issues to https://github.com/iree-org/iree/issues and include the crash backtrace.
// Stack dump:
// 0.      Program arguments: iree-opt --pass-pipeline=builtin.module(iree-amdaie-distribute-cores-and-objectfifos,cse) --split-input-file --verify-diagnostics /proj/gdba/jamesn/workspace/iree-amd-aie/compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test/distribute_cores_and_objectfifos_pt2.mlir
// Stack dump without symbol names (ensure you have llvm-symbolizer in your PATH or set the environment var `LLVM_SYMBOLIZER_PATH` to point to it):
// 0  libIREECompiler.so 0x00007f219bd93c88 llvm::sys::PrintStackTrace(llvm::raw_ostream&, int) + 40
// 1  libIREECompiler.so 0x00007f219bd91ea0 llvm::sys::RunSignalHandlers() + 80
// 2  libIREECompiler.so 0x00007f219bd94328
// 3  libc.so.6          0x00007f2196c42520
// 4  libc.so.6          0x00007f2196c969fc pthread_kill + 300
// 5  libc.so.6          0x00007f2196c42476 raise + 22
// 6  libc.so.6          0x00007f2196c287f3 abort + 211
// 7  libc.so.6          0x00007f2196c2871b
// 8  libc.so.6          0x00007f2196c39e96
// 9  libIREECompiler.so 0x00007f219cb6a5fb
// 10 libIREECompiler.so 0x00007f219bce0f4e
// 11 libIREECompiler.so 0x00007f219bce0f4e
// 12 libIREECompiler.so 0x00007f219cb63e5d
// 13 libIREECompiler.so 0x00007f219cb636af
// 14 libIREECompiler.so 0x00007f219bf316af mlir::detail::OpToOpPassAdaptor::run(mlir::Pass*, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int) + 639
// 15 libIREECompiler.so 0x00007f219bf31f77 mlir::detail::OpToOpPassAdaptor::runPipeline(mlir::OpPassManager&, mlir::Operation*, mlir::AnalysisManager, bool, unsigned int, mlir::PassInstrumentor*, mlir::PassInstrumentation::PipelineParentInfo const*) + 311
// 16 libIREECompiler.so 0x00007f219bf3444d mlir::PassManager::run(mlir::Operation*) + 973
// 17 libIREECompiler.so 0x00007f219bf228d5
// 18 libIREECompiler.so 0x00007f219bf224d0
// 19 libIREECompiler.so 0x00007f219bf2ace6
// 20 libIREECompiler.so 0x00007f219bf2ab0e mlir::splitAndProcessBuffer(std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::function_ref<mlir::LogicalResult (std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, llvm::raw_ostream&)>, llvm::raw_ostream&, llvm::StringRef, llvm::StringRef) + 1358
// 21 libIREECompiler.so 0x00007f219bf1e85d mlir::MlirOptMain(llvm::raw_ostream&, std::unique_ptr<llvm::MemoryBuffer, std::default_delete<llvm::MemoryBuffer>>, mlir::DialectRegistry&, mlir::MlirOptMainConfig const&) + 205
// 22 libIREECompiler.so 0x00007f219bcf7df2 ireeOptRunMain + 1538
// 23 libc.so.6          0x00007f2196c29d90
// 24 libc.so.6          0x00007f2196c29e40 __libc_start_main + 128
// 25 iree-opt           0x000055e761e486a5
// 
// --
// 
// ********************
// ********************
// Failed Tests (1):
//   IREE :: compiler/plugins/target/AMD-AIE/iree-amd-aie/Transforms/test/distribute_cores_and_objectfifos_pt2.mlir



// -----
