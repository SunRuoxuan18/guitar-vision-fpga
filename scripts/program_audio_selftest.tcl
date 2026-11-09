set bitstream_file "D:/FPGA/ZynqAudioLab/output/pynq_audio_board_top.bit"
if {[llength $argv] > 0} {
    set bitstream_file [lindex $argv 0]
}

if {![file exists $bitstream_file]} {
    error "Bitstream not found: $bitstream_file"
}

open_hw_manager
connect_hw_server -allow_non_jtag
open_hw_target

set devices [get_hw_devices -quiet xc7z020*]
if {[llength $devices] == 0} {
    error "No xc7z020 device found. Check PYNQ-Z2 power and PROG/JTAG USB."
}

set device [lindex $devices 0]
current_hw_device $device
refresh_hw_device -update_hw_probes false $device
set_property PROGRAM.FILE $bitstream_file $device
program_hw_devices $device
refresh_hw_device -update_hw_probes false $device

set design_id [get_property REGISTER.USERCODE $device]
puts "DESIGN_ID: $design_id"
if {[llength $argv] > 1 && ![string equal -nocase $design_id [lindex $argv 1]]} {
    error "Programmed design ID does not match the requested audio revision."
}

puts "PROGRAM_SUCCESS: $device"
puts "BITSTREAM: $bitstream_file"

close_hw_target
disconnect_hw_server
close_hw_manager
