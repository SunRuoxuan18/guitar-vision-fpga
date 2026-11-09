set root [file dirname [file dirname [file normalize [info script]]]]
set build "$root/diagnostics/hand_requant_overlay_build"
set out "$root/output/hand_requant"
open_project "$build/vision_pl.xpr"
cd $build
set_param general.maxThreads 2
reset_run synth_1
launch_runs synth_1 -jobs 2
wait_on_run synth_1
if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} { error "Requant system synthesis failed" }
launch_runs impl_1 -to_step route_design -jobs 2
wait_on_run impl_1
if {![string match "route_design Complete*" [get_property STATUS [get_runs impl_1]]]} { error "Requant routing failed" }
open_run impl_1
report_timing_summary -delay_type min_max -file "$out/timing.rpt"
report_utilization -hierarchical -hierarchical_depth 2 -file "$out/utilization.rpt"
report_drc -file "$out/drc.rpt"
report_methodology -json "$out/methodology.json"
write_xdc -force -exclude_physical "$out/constraints.xdc"
report_timing -delay_type max -max_paths 20 -file "$out/critical_paths.rpt"
set setup [get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]]
set hold [get_property SLACK [lindex [get_timing_paths -delay_type min -max_paths 1] 0]]
if {$setup < 0 || $hold < 0} { error "Requant timing failed setup=$setup hold=$hold" }
set violations [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]
if {[llength $violations]} { error "Requant DRC failed: $violations" }
set_property BITSTREAM.CONFIG.USERID 32'h41565331 [current_design]
write_checkpoint -force "$out/av_system_routed.dcp"
write_bitstream -force "$out/av_system.bit"
file copy -force "$build/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh" "$out/av_system.hwh"
puts "HAND_REQUANT_IMPLEMENTATION_PASS setup=$setup hold=$hold"
close_project
