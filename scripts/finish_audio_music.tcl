set root [file dirname [file dirname [file normalize [info script]]]]
set export_dir "$root/output/audio_music"
open_project "$root/diagnostics/audio_music_build/audio_music.xpr"
open_run impl_1
phys_opt_design -directive AggressiveExplore
report_timing_summary -file "$export_dir/audio_music_timing.rpt"
report_utilization -file "$export_dir/audio_music_utilization.rpt"
report_drc -file "$export_dir/audio_music_drc.rpt"
report_cdc -details -file "$export_dir/audio_music_cdc.rpt"
report_clock_interaction -file "$export_dir/audio_music_clock_interaction.rpt"
if {[get_property SLACK [lindex [get_timing_paths -delay_type max -max_paths 1] 0]] < 0} { error "Setup timing failed" }
if {[get_property SLACK [lindex [get_timing_paths -delay_type min -max_paths 1] 0]] < 0} { error "Hold timing failed" }
foreach pad {audio_i2c_scl audio_i2c_sda} {
    set nets [get_nets -segments -of_objects [get_ports $pad]]
    set cells [get_cells -of_objects [get_pins -leaf -of_objects $nets]]
    set buffers [filter $cells {REF_NAME == OBUFT || REF_NAME == IOBUF}]
    if {[llength $buffers] != 1} { error "Invalid I2C buffer: $pad" }
    set enable [get_nets -segments -of_objects [get_pins [lindex $buffers 0]/T]]
    set drivers [get_cells -of_objects [get_pins -leaf -of_objects $enable -filter {DIRECTION == OUT}]]
    if {[llength $drivers] == 0 || [llength [filter $drivers {REF_NAME == GND || REF_NAME == VCC}]]} { error "Constant I2C enable: $pad" }
}
set banks [get_cells -hier -filter {NAME =~ *capture_inst/bank*_reg* && REF_NAME =~ RAMB*}]
if {[llength $banks] != 4} { error "Expected four RAMB36 capture primitives" }
set_property BITSTREAM.CONFIG.USERID 32'h41554434 [current_design]
write_checkpoint -force "$export_dir/audio_music_routed.dcp"
write_bitstream -force "$export_dir/audio_music.bit"
file copy -force "$root/diagnostics/audio_music_build/audio_music.gen/sources_1/bd/audio_music/hw_handoff/audio_music.hwh" "$export_dir/audio_music.hwh"
puts "AUDIO_MUSIC_FINAL_PASS"
close_project
