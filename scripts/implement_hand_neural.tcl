# Implement the isolated hand-neural clone.  The AVS1 project/output is never opened.
# Reports and the bitstream are written only below output/hand_neural.
#
# Usage:
#   vivado -mode batch -source D:/FPGA/ZynqAudioLab/scripts/implement_hand_neural.tcl -tclargs --prepare-only
#   vivado -mode batch -source D:/FPGA/ZynqAudioLab/scripts/implement_hand_neural.tcl -tclargs --resume

set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/hand_neural_build"
set out "$root/output/hand_neural"
set build_jobs 1
if {[info exists ::env(HAND_NEURAL_FAST)] && $::env(HAND_NEURAL_FAST) eq "1"} {
    set build "$root/diagnostics/hand_neural_fast_build"
    set out "$root/output/hand_neural_fast"
    set build_jobs 2
}
set prepare_only 0
if {[info exists ::env(HAND_NEURAL_BUFFERED)] && $::env(HAND_NEURAL_BUFFERED) eq "1"} {
    set build "$root/diagnostics/hand_neural_buffered_build"
    set out "$root/output/hand_neural_buffered"
    set build_jobs 2
}
set resume 0
if {[info exists ::env(HAND_NEURAL_Q16)] && $::env(HAND_NEURAL_Q16) eq "1"} {
    set build "$root/diagnostics/hand_neural_q16_build"
    set out "$root/output/hand_neural_q16"
    set build_jobs 2
}
if {[info exists ::env(HAND_NEURAL_REALTIME)] && $::env(HAND_NEURAL_REALTIME) eq "1"} {
    set build "$root/diagnostics/hand_neural_realtime_v3_build"
    set out "$root/output/hand_neural_realtime"
    set build_jobs 2
}
if {[info exists ::env(HAND_NEURAL_BURST)] && $::env(HAND_NEURAL_BURST) eq "1"} {
    set build "$root/diagnostics/hand_neural_burst_build"
    set out "$root/output/hand_neural_burst"
    set build_jobs 2
}
if {[info exists ::env(HAND_NEURAL_COMPACT)] && $::env(HAND_NEURAL_COMPACT) eq "1"} {
    set build "$root/diagnostics/hand_neural_compact_build"
    set out "$root/output/hand_neural_compact"
    set build_jobs 2
}
if {[info exists ::env(HAND_NEURAL_BANKED)] && $::env(HAND_NEURAL_BANKED) eq "1"} {
    set build "$root/diagnostics/hand_neural_banked_build"
    set out "$root/output/hand_neural_banked"
    set build_jobs 2
}
if {[info exists ::env(HAND_NEURAL_LEAN)] && $::env(HAND_NEURAL_LEAN) eq "1"} {
    set build "$root/diagnostics/hand_neural_lean_build"
    set out "$root/output/hand_neural_lean"
    set build_jobs 2
}
if {[info exists ::env(HAND_NEURAL_LEAN_IO)] && $::env(HAND_NEURAL_LEAN_IO) eq "1"} {
    set build "$root/diagnostics/hand_neural_lean_io_build"
    set out "$root/output/hand_neural_lean_io"
    set build_jobs 2
}
if {[info exists ::env(HAND_NEURAL_DW_BOUNDED)] && $::env(HAND_NEURAL_DW_BOUNDED) eq "1"} {
    set build "$root/diagnostics/hand_neural_dw_bounded_build"
    set out "$root/output/hand_neural_dw_bounded"
    set build_jobs 2
}
if {[info exists ::env(HAND_NEURAL_PW_CACHED)] && $::env(HAND_NEURAL_PW_CACHED) eq "1"} {
    set build "$root/diagnostics/hand_neural_pw_cached_build"
    set out "$root/output/hand_neural_pw_cached"
    set build_jobs 2
}
if {[info exists ::env(HAND_NEURAL_FUSED)] && $::env(HAND_NEURAL_FUSED) eq "1"} {
    set build "$root/diagnostics/hand_neural_fused_build"
    set out "$root/output/hand_neural_fused"
    set build_jobs 2
}
if {[info exists ::env(HAND_NEURAL_DW_WIDE)] && $::env(HAND_NEURAL_DW_WIDE) eq "1"} {
    set build "$root/diagnostics/hand_neural_dw_wide_build"
    set out "$root/output/hand_neural_dw_wide"
    set build_jobs 2
}
foreach argument $argv {
    switch -exact -- $argument {
        --prepare-only { set prepare_only 1 }
        --resume { set resume 1 }
        default { error "Unknown argument: $argument (use --prepare-only or --resume)" }
    }
}
if {![file isfile "$build/vision_pl.xpr"]} {
    error "Prepared hand-neural project is missing: $build/vision_pl.xpr; run build_hand_neural.tcl first"
}
file mkdir $out
cd $build
set_param general.maxThreads 1
if {([info exists ::env(HAND_NEURAL_LEAN_IO)] && $::env(HAND_NEURAL_LEAN_IO) eq "1") ||
    ([info exists ::env(HAND_NEURAL_DW_BOUNDED)] && $::env(HAND_NEURAL_DW_BOUNDED) eq "1") ||
    ([info exists ::env(HAND_NEURAL_PW_CACHED)] && $::env(HAND_NEURAL_PW_CACHED) eq "1")} {
    set_param general.maxThreads 2
}
open_project "$build/vision_pl.xpr"
open_bd_design [get_files vision_pl.bd]
validate_bd_design
generate_target all [get_files vision_pl.bd]
save_bd_design

if {$prepare_only} {
    puts "HAND_NEURAL_PREPARED_ONLY: $build/vision_pl.xpr"
    close_project
    return
}

if {!$resume} {
    reset_run synth_1
    launch_runs synth_1 -jobs $build_jobs
}
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} { error "Hand-neural synthesis is not complete" }
launch_runs impl_1 -to_step route_design -jobs $build_jobs
wait_on_run impl_1
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} { error "Hand-neural implementation failed" }
open_run impl_1

set setup_paths [get_timing_paths -delay_type max -max_paths 1]
if {[llength $setup_paths] == 0} { error "Missing constrained setup path" }
set setup [get_property SLACK [lindex $setup_paths 0]]
if {$setup < 0} {
    phys_opt_design -directive AggressiveExplore
    route_design -directive Explore
}

report_timing_summary -delay_type min_max -file "$out/hand_neural_timing.rpt"
report_utilization -hierarchical -hierarchical_depth 2 -file "$out/hand_neural_utilization.rpt"
report_drc -file "$out/hand_neural_drc.rpt"
report_route_status -file "$out/hand_neural_route.rpt"
report_bus_skew -file "$out/hand_neural_bus_skew.rpt" -warn_on_violation
set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup_paths] == 0 || [llength $hold_paths] == 0} { error "Missing constrained setup/hold paths" }
set setup [get_property SLACK [lindex $setup_paths 0]]
set hold [get_property SLACK [lindex $hold_paths 0]]
if {$setup < 0 || $hold < 0} { error "Hand-neural timing failed: setup=$setup hold=$hold" }
set drc_failures [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]
if {[llength $drc_failures] != 0} { error "Hand-neural DRC failed: $drc_failures" }

set_property BITSTREAM.CONFIG.USERID 32'h41565331 [current_design]
write_checkpoint -force "$out/av_system_routed.dcp"
write_bitstream -force "$out/av_system.bit"
set hwh "$build/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh"
if {![file isfile $hwh]} { error "Missing hand-neural hardware handoff: $hwh" }
file copy -force $hwh "$out/av_system.hwh"
puts "HAND_NEURAL_BUILD_PASS: setup=$setup hold=$hold"
puts "HAND_NEURAL_OUTPUT: $out/av_system.bit $out/av_system.hwh"
close_project
