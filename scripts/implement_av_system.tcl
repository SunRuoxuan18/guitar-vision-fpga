set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/av_system_build"
set out "$root/output/av_system"
cd $build
set_param general.maxThreads 1
open_project "$build/vision_pl.xpr"
if {[lsearch -exact $argv --resume] < 0} { launch_runs synth_1 -jobs 1 }
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} { error "AV synthesis failed" }
launch_runs impl_1 -to_step route_design -jobs 1
wait_on_run impl_1
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} { error "AV implementation failed" }
open_run impl_1
if {[get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]] < 0} {
    phys_opt_design -directive AggressiveExplore
    route_design -directive Explore
}
report_timing_summary -delay_type min_max -file "$out/av_system_timing.rpt"
report_utilization -hierarchical -hierarchical_depth 2 -file "$out/av_system_utilization.rpt"
report_drc -file "$out/av_system_drc.rpt"
report_cdc -details -file "$out/av_system_cdc.rpt"
report_route_status -file "$out/av_system_route.rpt"
set setup [get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]]
set hold [get_property SLACK [lindex [get_timing_paths -delay_type min -max_paths 1] 0]]
if {$setup < 0 || $hold < 0} { error "AV timing failed setup=$setup hold=$hold" }
if {[llength [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]]} { error "AV DRC failed" }
foreach pad {audio_i2c_scl audio_i2c_sda} {
    set nets [get_nets -segments -of_objects [get_ports $pad]]
    set cells [get_cells -of_objects [get_pins -leaf -of_objects $nets]]
    set buffers [filter $cells {REF_NAME == OBUFT || REF_NAME == IOBUF}]
    if {[llength $buffers] != 1} { error "Invalid I2C buffer $pad" }
    set enable [get_nets -segments -of_objects [get_pins [lindex $buffers 0]/T]]
    set drivers [get_cells -of_objects [get_pins -leaf -of_objects $enable -filter {DIRECTION == OUT}]]
    if {![llength $drivers] || [llength [filter $drivers {REF_NAME == GND || REF_NAME == VCC}]]} { error "Constant I2C enable $pad" }
}
set_property BITSTREAM.CONFIG.USERID 32'h41565331 [current_design]
write_checkpoint -force "$out/av_system_routed.dcp"
write_bitstream -force "$out/av_system.bit"
file copy -force "$build/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh" "$out/av_system.hwh"
puts "AV_SYSTEM_BUILD_PASS: setup=$setup hold=$hold"
close_project
