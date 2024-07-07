// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/AMDAIEUtils.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Iterators.h"

#define DEBUG_TYPE "iree-amdaie-matmul-direct"

namespace mlir::iree_compiler::AMDAIE {

namespace {

class AMDAIEMatmulDirectPass
    : public impl::AMDAIEMatmulDirectBase<AMDAIEMatmulDirectPass> {
 private:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect, AMDAIEDialect>();
  }

 public:
  void runOnOperation() final {

    mlir::FunctionOpInterface funcOp = getOperation();
    auto context = funcOp->getContext();
    IRRewriter rewriter(context);

    // 1) Create workgroup, and move function body into it.
    Block *funcBlock = &funcOp.getFunctionBody().front();
    auto *newBlock = rewriter.createBlock(funcOp.getCallableRegion());
    rewriter.setInsertionPointToStart(newBlock);
    auto workgroupOp = rewriter.create<AMDAIE::WorkgroupOp>(funcOp->getLoc());
    rewriter.moveOpAfter(funcBlock->getTerminator(), workgroupOp);
    auto body = workgroupOp.getBody();
    rewriter.inlineBlockBefore(funcBlock, body, body->begin());

    // 2) what great progress!
  }
};

}  // namespace

std::unique_ptr<Pass> createAMDAIEMatmulDirectPass() {
  return std::make_unique<AMDAIEMatmulDirectPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
