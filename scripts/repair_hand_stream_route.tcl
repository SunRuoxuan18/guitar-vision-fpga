set root [file dirname [file dirname [file normalize [info script]]]]
set out "$root/output/hand_stream/route_repair"
file mkdir $out
open_checkpoint "$root/diagnostics/hand_stream_overlay_build/vision_pl.runs/impl_1/vision_pl_wrapper_routed.dcp"
set_param general.maxThreads 2
report_methodology -json "$out/methodology_before.json"
write_xdc -force -exclude_physical "$out/constraints_before.xdc"
report_timing -max_paths 20 -file "$out/paths_before.rpt"
phys_opt_design -directive AggressiveExplore
route_design -directive Explore
report_timing_summary -delay_type min_max -file "$out/timing.rpt"
report_timing -max_paths 20 -file "$out/paths_after.rpt"
report_utilization -file "$out/utilization.rpt"
report_route_status -file "$out/route_status.rpt"
report_drc -file "$out/drc.rpt"
report_methodology -json "$out/methodology_after.json"
write_checkpoint -force "$out/repaired_routed.dcp"
set setup [get_property SLACK [get_timing_paths -delay_type max -max_paths 1]]
set hold [get_property SLACK [get_timing_paths -delay_type min -max_paths 1]]
if {$setup < 0 || $hold < 0} {error "HAND_STREAM_REPAIR_FAILED setup=$setup hold=$hold"}
if {[llength [get_drc_violations -quiet -filter {SEVERITY == Error || SEVERITY == {Critical Warning}}]]} {error "HAND_STREAM_REPAIR_DRC_FAILED"}
set_property BITSTREAM.CONFIG.USERID 32'h41565331 [current_design]
write_bitstream -force "$root/output/hand_stream/av_system.bit"
file copy -force "$root/diagnostics/hand_stream_overlay_build/vision_pl.gen/sources_1/bd/vision_pl/hw_handoff/vision_pl.hwh" "$root/output/hand_stream/av_system.hwh"
puts "HAND_STREAM_REPAIR_PASS setup=$setup hold=$hold"
close_design
