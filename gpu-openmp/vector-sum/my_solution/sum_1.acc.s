; ModuleID = 'The Accel Module'
source_filename = "The Accel Module"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: noinline
define amdgpu_kernel void @"vectorsum_$ck_L19_1"(i64 %"$$arg_ptr_acc_vecb_t15_t371", i64 %"$$arg_ptr_acc_veca_t17_t382", i64 %"$$arg_ptr_acc_vecc_t19_t393") #0 !dbg !5 !scalarlevel !9 !cachelevel !10 !fplevel !9 {
", bb80":
  %r = tail call i64 @__ockl_get_local_size(i32 0), !dbg !11 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:20
  %r4 = trunc i64 %r to i32, !dbg !11 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:20
  %r5 = tail call i32 @llvm.amdgcn.workgroup.id.x(), !dbg !11 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:20
  %r6 = tail call i32 @llvm.amdgcn.workitem.id.x(), !dbg !11 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:20
  %r7 = mul i32 %r5, %r4, !dbg !11 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:20
  %r8 = add i32 %r6, %r7, !dbg !11 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:20
  %r10 = icmp slt i32 %r8, 102400, !dbg !11 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:20
  br i1 %r10, label %"file sum.F90, line 20, bb13", label %"file sum.F90, line 23, bb93", !dbg !11 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:20

"file sum.F90, line 20, bb13":                    ; preds = %", bb80"
  %r15 = inttoptr i64 %"$$arg_ptr_acc_veca_t17_t382" to double*, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %cic-gep-idxcast = sext i32 %r8 to i64, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r16 = getelementptr double, double* %r15, i64 %cic-gep-idxcast, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r17 = addrspacecast double* %r16 to double addrspace(1)*, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r18 = load double, double addrspace(1)* %r17, align 8, !dbg !12, !CrayMri !13 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r21 = inttoptr i64 %"$$arg_ptr_acc_vecb_t15_t371" to double*, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %cic-gep-idxcast31 = sext i32 %r8 to i64, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r22 = getelementptr double, double* %r21, i64 %cic-gep-idxcast31, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r23 = addrspacecast double* %r22 to double addrspace(1)*, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r24 = load double, double addrspace(1)* %r23, align 8, !dbg !12, !CrayMri !14 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r25 = fadd double %r18, %r24, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r28 = inttoptr i64 %"$$arg_ptr_acc_vecc_t19_t393" to double*, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %cic-gep-idxcast32 = sext i32 %r8 to i64, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r29 = getelementptr double, double* %r28, i64 %cic-gep-idxcast32, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  %r30 = addrspacecast double* %r29 to double addrspace(1)*, !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  store double %r25, double addrspace(1)* %r30, align 8, !dbg !12, !CrayMri !15 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21
  br label %"file sum.F90, line 23, bb93", !dbg !12 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:21

"file sum.F90, line 23, bb93":                    ; preds = %"file sum.F90, line 20, bb13", %", bb80"
  ret void, !dbg !16 ; /pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90:23
}

declare i64 @__ockl_get_local_size(i32)

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workgroup.id.x() #1

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workitem.id.x() #1

attributes #0 = { noinline "amdgpu-flat-work-group-size"="1,1024" "amdgpu-implicitarg-num-bytes"="56" "uniform-work-group-size"="true" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}
!PDGFunctionMap = !{!3}
!nvvm.annotations = !{!4}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !2, producer: "Cray Fortran : Version 15.0.1", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!2 = !DIFile(filename: "/pfs/lustrep1/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution/sum.F90", directory: "/scratch/project_465000536/mizhang/summerschool/gpu-openmp/vector-sum/my_solution")
!3 = !{i32 3, !"vectorsum_$ck_L19_1"}
!4 = !{void (i64, i64, i64)* @"vectorsum_$ck_L19_1", !"kernel", i32 1}
!5 = distinct !DISubprogram(name: "vectorsum_$ck_L19_1", linkageName: "vectorsum_$ck_L19_1", scope: !2, file: !2, line: 19, type: !6, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !8)
!6 = !DISubroutineType(types: !7)
!7 = !{null}
!8 = !{}
!9 = !{i64 2}
!10 = !{i64 0}
!11 = !DILocation(line: 20, scope: !5)
!12 = !DILocation(line: 21, scope: !5)
!13 = !{i64 1911260446797}
!14 = !{i64 1915555414093}
!15 = !{i64 18016305474961485}
!16 = !DILocation(line: 23, scope: !5)
