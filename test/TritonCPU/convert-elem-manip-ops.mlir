// RUN: triton-opt %s -split-input-file -triton-cpu-convert-elem-manip-ops -canonicalize | FileCheck %s --implicit-check-not=tt.gather --implicit-check-not=memref.load --implicit-check-not=memref.store

// Gather through the middle axis, with a larger result and 64-bit indices.
// Materialize only the source and retain vector indices in the result shape.
// CHECK-LABEL: @gather_middle_axis
// CHECK-DAG: %[[POISON:.+]] = ub.poison : vector<2x16x8xf32>
// CHECK-DAG: %[[ALLTRUE:.+]] = arith.constant dense<true> : vector<2x16x8xi1>
// CHECK: %[[SOURCE:.+]] = memref.alloca() {{.*}} : memref<2x4x8xf32>
// CHECK: vector.transfer_write {{.*}}, %[[SOURCE]][{{.*}}]
// CHECK: %[[FLAT:.+]] = memref.collapse_shape %[[SOURCE]] {{.*}} : memref<2x4x8xf32> into memref<64xf32>
// CHECK: %[[MASK:.+]] = arith.cmpi ult, {{.*}} : vector<2x16x8xi64>
// CHECK: %[[SAFE:.+]] = arith.select %[[MASK]], {{.*}} : vector<2x16x8xi1>, vector<2x16x8xi64>
// CHECK: %[[SCALED:.+]] = arith.muli %[[SAFE]], {{.*}} : vector<2x16x8xi64>
// CHECK: %[[INDICES:.+]] = arith.addi %[[SCALED]], {{.*}} : vector<2x16x8xi64>
// CHECK: vector.gather %[[FLAT]][{{.*}}] [%[[INDICES]]], %[[ALLTRUE]], %[[POISON]] : memref<64xf32>, vector<2x16x8xi64>, vector<2x16x8xi1>, vector<2x16x8xf32> into vector<2x16x8xf32>
tt.func @gather_middle_axis(%src: tensor<2x4x8xf32>, %indices: tensor<2x16x8xi64>) -> tensor<2x16x8xf32> {
  %result = tt.gather %src[%indices] {axis = 1 : i32} : (tensor<2x4x8xf32>, tensor<2x16x8xi64>) -> tensor<2x16x8xf32>
  tt.return %result : tensor<2x16x8xf32>
}

// -----

// Narrow indices must be zero-extended: i8 values 128..255 can be valid indices.
// CHECK-LABEL: @gather_unsigned_indices
// CHECK: arith.extui {{.*}} : vector<64xi8> to vector<64xi64>
// CHECK: vector.gather {{.*}} : memref<256xi32>, vector<64xi64>, vector<64xi1>, vector<64xi32> into vector<64xi32>
tt.func @gather_unsigned_indices(%src: tensor<256xi32>, %indices: tensor<64xi8>) -> tensor<64xi32> {
  %result = tt.gather %src[%indices] {axis = 0 : i32} : (tensor<256xi32>, tensor<64xi8>) -> tensor<64xi32>
  tt.return %result : tensor<64xi32>
}

// -----

// Fixed scratch allocations must be outside the enclosing source loop, whose
// automatic allocation scope disappears when SCF is lowered to control flow.
// CHECK-LABEL: @gather_in_loop
// CHECK: memref.alloca() {{.*}} : memref<4xf32>
// CHECK: scf.for
// CHECK-NOT: memref.alloca
// CHECK: vector.transfer_write
// CHECK-NOT: scf.for
// CHECK-NOT: memref.alloca
// CHECK: vector.gather
// CHECK-NOT: scf.for
// CHECK: tt.return
tt.func @gather_in_loop(%src: tensor<4xf32>, %indices: tensor<4xi32>, %out: tensor<4x!tt.ptr<f32>>, %count: i32) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  scf.for %i = %zero to %count step %one : i32 {
    %result = tt.gather %src[%indices] {axis = 0 : i32} : (tensor<4xf32>, tensor<4xi32>) -> tensor<4xf32>
    tt.store %out, %result : tensor<4x!tt.ptr<f32>>
  }
  tt.return
}
