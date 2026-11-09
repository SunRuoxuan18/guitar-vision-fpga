# Repair the inactive level of proc_sys_reset's active-low auxiliary input.
set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/lk_hls_overlay"
set output "$root/output/lk_hls"
set_param general.maxThreads 4
open_project "$build/lk_hls.xpr"
open_bd_design [get_files lk_hls.bd]
disconnect_bd_net [get_bd_nets -of_objects [get_bd_pins reset_control/aux_reset_in]] [get_bd_pins reset_control/aux_reset_in]
connect_bd_net [get_bd_pins clock_locked/dout] [get_bd_pins reset_control/aux_reset_in]
validate_bd_design
save_bd_design
generate_target all [get_files lk_hls.bd]
reset_run synth_1
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {error "Synthesis failed"}
launch_runs impl_1 -to_step route_design -jobs 4
wait_on_run impl_1
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} {error "Route failed"}
open_run impl_1
report_timing_summary -file "$output/timing.rpt"
report_utilization -file "$output/utilization.rpt"
report_drc -file "$output/drc.rpt"
foreach type {min max} {
    set paths [get_timing_paths -delay_type $type -max_paths 1]
    if {[llength $paths] == 0 || [get_property SLACK [lindex $paths 0]] < 0} {error "Timing failed"}
}
if {[llength [get_drc_violations -quiet -filter {SEVERITY == Error}]] > 0} {error "DRC failed"}
set_property BITSTREAM.CONFIG.USERID 32'h4C4B4831 [current_design]
write_bitstream -force "$output/lk_hls.bit"
file copy -force "$build/lk_hls.gen/sources_1/bd/lk_hls/hw_handoff/lk_hls.hwh" "$output/lk_hls.hwh"
puts "LK_HLS_RESET_REPAIR_PASS"
close_project
