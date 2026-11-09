# Implement an already validated vision-flow project without rebuilding its BD.
set root [file dirname [file dirname [file normalize [info script]]]]
set build_dir "$root/diagnostics/vision_flow_build"
set export_dir "$root/output/vision_flow"
if {[info exists ::env(VISION_FLOW_BUILD_DIR)]} { set build_dir [file normalize $::env(VISION_FLOW_BUILD_DIR)] }
if {[info exists ::env(VISION_FLOW_EXPORT_DIR)]} { set export_dir [file normalize $::env(VISION_FLOW_EXPORT_DIR)] }
file mkdir $export_dir
cd $build_dir
set_param general.maxThreads 4
open_project "$build_dir/vision_pl.xpr"
open_bd_design [get_files vision_pl.bd]
foreach core {lk_pyramid roi_projection} {
    if {[llength [get_bd_cells -quiet $core]] != 1} { error "Missing prepared core $core" }
}
validate_bd_design
save_bd_design
update_compile_order -fileset sources_1
if {[lsearch -exact $argv --resume-synthesis] < 0} { launch_runs synth_1 -jobs 2 }
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
    error "Vision synthesis failed: [get_property STATUS [get_runs synth_1]]"
}
launch_runs impl_1 -to_step route_design -jobs 2
wait_on_run impl_1
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} {
    error "Vision implementation failed: [get_property STATUS [get_runs impl_1]]"
}
open_run impl_1
set_property BITSTREAM.CONFIG.USERID 32'h56464C31 [current_design]
set initial_setup [get_timing_paths -delay_type max -max_paths 1]
if {[llength $initial_setup] && [get_property SLACK [lindex $initial_setup 0]] < 0} {
    phys_opt_design -directive AggressiveExplore
    route_design -directive Explore
}
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file "$export_dir/vision_pl_timing.rpt"
report_utilization -hierarchical -hierarchical_depth 2 -file "$export_dir/vision_pl_utilization.rpt"
report_drc -file "$export_dir/vision_pl_drc.rpt"
report_route_status -file "$export_dir/vision_pl_route.rpt"
set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
if {[llength $setup_paths] == 0 || [llength $hold_paths] == 0} { error "Missing constrained paths" }
set setup_slack [get_property SLACK [lindex $setup_paths 0]]
set hold_slack [get_property SLACK [lindex $hold_paths 0]]
if {$setup_slack < 0 || $hold_slack < 0} { error "Timing failed setup=$setup_slack hold=$hold_slack" }
set violations [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]
if {[llength $violations]} { error "DRC failed: $violations" }
write_checkpoint -force "$export_dir/vision_pl_routed.dcp"
write_bitstream -force "$export_dir/vision_pl.bit"
file copy -force "$build_dir/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh" "$export_dir/vision_pl.hwh"
puts "VISION_FLOW_BUILD_PASS: setup=$setup_slack ns hold=$hold_slack ns"
close_project
