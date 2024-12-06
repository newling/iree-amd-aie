// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-amd-aie/IR/AMDAIEOps.h"
#include "iree-amd-aie/Transforms/Passes.h"
#include "iree-amd-aie/Transforms/Utils/AMDAIEUtils.h"
#include "iree-amd-aie/aie_runtime/iree_aie_runtime.h"
#include "mlir/IR/Iterators.h"
#define DEBUG_TYPE "iree-amdaie-fold-dma-waits"

namespace mlir::iree_compiler::AMDAIE {

namespace {

FailureOr<bool> canFoldBasedOnHalfDmaCpyNdOp(
    AMDAIE::NpuHalfDmaCpyNdOp npuHalfDmaCpyNdOp,
    const AMDAIE::AMDAIEDeviceModel &deviceModel,
    DenseMap<std::pair<AMDAIE::TileOp, AMDAIE::ConnectionOp>,
             SmallVector<uint32_t>> &tileConnectionToBdIdQueueMap) {
  // Retrieve the connection op.
  std::optional<AMDAIE::ConnectionOp> maybeConnectionOp =
      npuHalfDmaCpyNdOp.getConnectionOp();
  if (!maybeConnectionOp) {
    return npuHalfDmaCpyNdOp.emitOpError()
           << "expected to operate on an `amdaie.connection`";
  }
  AMDAIE::ConnectionOp connectionOp = maybeConnectionOp.value();

  // Retrieve the flow op.
  std::optional<AMDAIE::FlowOp> maybeFlowOp = connectionOp.getFlowOp();
  if (!maybeFlowOp) {
    return maybeConnectionOp->emitOpError()
           << "expected to operate on an `amdaie.flow`";
  }
  bool isPacketFlow = maybeFlowOp->getIsPacketFlow();

  // Retrieve the BD ID op.
  std::optional<AMDAIE::BdIdOp> maybeBdIdOp = npuHalfDmaCpyNdOp.getBdIdOp();
  if (!maybeBdIdOp) {
    return npuHalfDmaCpyNdOp.emitOpError()
           << "must have a BD ID op to lower to `amdaie.npu.write_bd`";
  }
  AMDAIE::BdIdOp bdIdOp = maybeBdIdOp.value();

  // Retrieve the tile op.
  AMDAIE::TileOp tileOp =
      dyn_cast_if_present<AMDAIE::TileOp>(bdIdOp.getTile().getDefiningOp());
  if (!tileOp) {
    return bdIdOp.emitOpError() << "must operate on an `amdaie.tile`";
  }

  // Get the maximum queue size.
  uint32_t col = getConstantIndexOrAssert(tileOp.getCol());
  uint32_t row = getConstantIndexOrAssert(tileOp.getRow());
  uint32_t maxQueueSize = deviceModel.getDmaMaxQueueSize(col, row);

  // Keep wait op if any of:
  //
  // 1) reaches the maximum queue size, or
  // 2) the queue is currently empty
  // 3) there is a duplicate BD ID in the same tile
  // 4) the flow is packet flow
  uint32_t bdId = getConstantIndexOrAssert(bdIdOp.getValue());
  bool isDuplicateBdId =
      llvm::any_of(tileConnectionToBdIdQueueMap, [&](const auto &entry) {
        return entry.first.first == tileOp &&
               llvm::is_contained(entry.second, bdId);
      });
  SmallVector<uint32_t> &bdIdQueue =
      tileConnectionToBdIdQueueMap[std::make_pair(tileOp, connectionOp)];
  if (isDuplicateBdId || isPacketFlow || bdIdQueue.size() >= maxQueueSize) {
    bdIdQueue.clear();
  }

  bool queueIsEmpty = bdIdQueue.empty();
  bdIdQueue.push_back(bdId);

  return !queueIsEmpty;
}

FailureOr<bool> canFoldWaitOp(
    AMDAIE::NpuDmaWaitOp waitOp, const AMDAIE::AMDAIEDeviceModel &deviceModel,
    DenseMap<std::pair<AMDAIE::TileOp, AMDAIE::ConnectionOp>,
             SmallVector<uint32_t>> &tileConnectionToBdIdQueueMap) {
  for (Value token : waitOp.getAsyncTokens()) {
    if (auto npuHalfDmaCpyNdOp = dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
            token.getDefiningOp())) {
      auto maybeCanFold = canFoldBasedOnHalfDmaCpyNdOp(
          npuHalfDmaCpyNdOp, deviceModel, tileConnectionToBdIdQueueMap);
      if (failed(maybeCanFold)) return failure();
      if (maybeCanFold.value() == false) return false;
    }
  }

  return true;
}

/// Traverses the control code in reverse, ensuring that for each connection,
/// only one DMA wait op is retained for every maximum queue size.
LogicalResult foldDmaWaits(const AMDAIE::AMDAIEDeviceModel &deviceModel,
                           AMDAIE::ControlCodeOp controlCodeOp) {
  IRRewriter rewriter(controlCodeOp->getContext());

  std::vector<AMDAIE::NpuDmaWaitOp> waitOpsToErase;

  DenseMap<std::pair<AMDAIE::TileOp, AMDAIE::ConnectionOp>,
           SmallVector<uint32_t>>
      tileConnectionToBdIdQueueMap;

  // Traverse the control code in reverse.
  WalkResult res = controlCodeOp->walk<WalkOrder::PostOrder, ReverseIterator>(
      [&](AMDAIE::NpuDmaWaitOp waitOp) {
        FailureOr<bool> maybeToErase =
            canFoldWaitOp(waitOp, deviceModel, tileConnectionToBdIdQueueMap);
        if (failed(maybeToErase)) return WalkResult::interrupt();
        bool canFold = maybeToErase.value();
        // Erase later to avoid invalidating the iterator.
        if (canFold) waitOpsToErase.push_back(waitOp);
        return WalkResult::advance();
      });
  if (res.wasInterrupted()) return failure();

  for (AMDAIE::NpuDmaWaitOp waitOp : waitOpsToErase) {
    SmallVector<Value> asyncTokens(waitOp.getAsyncTokens());
    // Erase the wait op.
    rewriter.eraseOp(waitOp);
    for (Value token : asyncTokens) {
      if (auto op = dyn_cast_if_present<AMDAIE::NpuHalfDmaCpyNdOp>(
              token.getDefiningOp())) {
        if (op.use_empty()) {
          rewriter.setInsertionPoint(op);
          TypeRange resultTypeRange = TypeRange{};
          // Nullify the result to avoid issuing a token.
          rewriter.create<AMDAIE::NpuHalfDmaCpyNdOp>(
              op.getLoc(), resultTypeRange, op.getConnection(), op.getInput(),
              op.getMixedOffsets(), op.getMixedSizes(), op.getMixedStrides(),
              op.getBdId(), op.getChannel());
          rewriter.eraseOp(op);
        }
      }
    }
  }

  return success();
}

class AMDAIEFoldDmaWaitsPass
    : public impl::AMDAIEFoldDmaWaitsBase<AMDAIEFoldDmaWaitsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AMDAIEDialect>();
  }

  AMDAIEFoldDmaWaitsPass() = default;
  AMDAIEFoldDmaWaitsPass(const AMDAIEFoldDmaWaitsPass &pass){};
  void runOnOperation() override;
};

void AMDAIEFoldDmaWaitsPass::runOnOperation() {
  Operation *parentOp = getOperation();

  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(parentOp);
  std::optional<AMDAIEDevice> maybeDevice = getConfigAMDAIEDevice(targetAttr);
  if (!maybeDevice) {
    parentOp->emitOpError()
        << "has no AMDAIEDevice in the target attribute configuration. This "
           "device-specific information is required to fold DMA wait ops.";
    return signalPassFailure();
  }
  AMDAIE::AMDAIEDeviceModel deviceModel =
      AMDAIE::getDeviceModel(maybeDevice.value());

  DenseSet<Block *> blocksWithDmaWaits;
  parentOp->walk([&](AMDAIE::NpuDmaWaitOp waitOp) {
    blocksWithDmaWaits.insert(waitOp->getBlock());
  });

  // Collect all control code ops:
  SmallVector<AMDAIE::ControlCodeOp> controlCodeOps;
  parentOp->walk([&](AMDAIE::ControlCodeOp controlCodeOp) {
    controlCodeOps.push_back(controlCodeOp);
  });
  for (AMDAIE::ControlCodeOp controlCodeOp : controlCodeOps) {
    if (failed(foldDmaWaits(deviceModel, controlCodeOp))) {
      return signalPassFailure();
    }
  }
}

}  // namespace

std::unique_ptr<Pass> createAMDAIEFoldDmaWaitsPass() {
  return std::make_unique<AMDAIEFoldDmaWaitsPass>();
}

}  // namespace mlir::iree_compiler::AMDAIE
