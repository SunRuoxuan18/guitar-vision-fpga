set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/hand_stream_overlay_build"
set out "$root/output/hand_stream"
open_project "$build/vision_pl.xpr"
cd $build
set_param general.maxThreads 2
reset_run synth_1
launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} { error "Stream system synthesis failed" }
launch_runs impl_1 -to_step route_design -jobs 2
wait_on_run impl_1
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} { error "Stream routing failed" }
open_run impl_1
report_timing_summary -delay_type min_max -file "$out/timing.rpt"
report_utilization -hierarchical -hierarchical_depth 2 -file "$out/utilization.rpt"
report_drc -file "$out/drc.rpt"
set setup_paths [get_timing_paths -delay_type max -max_paths 1]
set hold_paths [get_timing_paths -delay_type min -max_paths 1]
if {![llength $setup_paths] || ![llength $hold_paths]} { error "Missing timing paths" }
set setup [get_property SLACK [lindex $setup_paths 0]]
set hold [get_property SLACK [lindex $hold_paths 0]]
if {$setup < 0 || $hold < 0} { error "Stream timing failed setup=$setup hold=$hold" }
if {[llength [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]]} { error "Stream DRC failed" }
set_property BITSTREAM.CONFIG.USERID 32'h41565331 [current_design]
write_checkpoint -force "$out/av_system_routed.dcp"
write_bitstream -force "$out/av_system.bit"
file copy -force "$build/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh" "$out/av_system.hwh"
puts "HAND_STREAM_IMPLEMENTATION_PASS setup=$setup hold=$hold"
close_project
