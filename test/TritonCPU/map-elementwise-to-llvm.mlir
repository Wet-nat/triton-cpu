// RUN: triton-opt %s -split-input-file -triton-cpu-map-elementwise-to-llvm -canonicalize | FileCheck %s --implicit-check-not=tt.map_elementwise

// A scalar map body can return early from different control-flow blocks.
// CHECK-LABEL: @map_conditional_returns
// CHECK: llvm.extractelement
// CHECK: llvm.cond_br
// CHECK: llvm.br
// CHECK: llvm.insertelement
tt.func @map_conditional_returns(%input: tensor<8xi32>) -> tensor<8xi32> {
  %result = "tt.map_elementwise"(%input) <{pack = 1 : i32}> ({
  ^bb0(%x: i32):
    %zero = arith.constant 0 : i32
    %negative = arith.cmpi slt, %x, %zero : i32
    cf.cond_br %negative, ^bb1, ^bb2
  ^bb1:
    %negated = arith.subi %zero, %x : i32
    tt.map_elementwise.return %negated : i32
  ^bb2:
    tt.map_elementwise.return %x : i32
  }) : (tensor<8xi32>) -> tensor<8xi32>
  tt.return %result : tensor<8xi32>
}

// -----

// Packs are grouped by operand and results retain their individual types.
// CHECK-LABEL: @map_packed_mixed_outputs
// CHECK: llvm.extractvalue
// CHECK: llvm.extractelement {{.*}} : vector<4xi32>
// CHECK: llvm.extractelement {{.*}} : vector<4xf32>
// CHECK: llvm.insertelement {{.*}} : vector<4xi32>
// CHECK: llvm.insertelement {{.*}} : vector<4xf32>
// CHECK: llvm.insertvalue
tt.func @map_packed_mixed_outputs(%input: tensor<2x4xi32>, %other: tensor<2x4xf32>) -> (tensor<2x4xi32>, tensor<2x4xf32>) {
  %result:2 = "tt.map_elementwise"(%input, %other) <{pack = 2 : i32}> ({
  ^bb0(%a0: i32, %a1: i32, %b0: f32, %b1: f32):
    %sum = arith.addi %a0, %a1 : i32
    %difference = arith.subf %b0, %b1 : f32
    tt.map_elementwise.return %sum, %a0, %difference, %b0 : i32, i32, f32, f32
  }) : (tensor<2x4xi32>, tensor<2x4xf32>) -> (tensor<2x4xi32>, tensor<2x4xf32>)
  tt.return %result#0, %result#1 : tensor<2x4xi32>, tensor<2x4xf32>
}
