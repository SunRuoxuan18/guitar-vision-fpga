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
puts "DEVICE_STATUS: $device"
set design_id [get_property REGISTER.USERCODE $device]
puts "DESIGN_ID: $design_id"
puts "FPGA_DONE: [get_property REGISTER.CONFIG_STATUS.BIT13_DONE_INTERNAL_SIGNAL_STATUS $device]"
puts "FPGA_CRC_ERROR: [get_property REGISTER.CONFIG_STATUS.BIT00_CRC_ERROR $device]"
if {[llength $argv] > 0 && ![string equal -nocase $design_id [lindex $argv 0]]} {
    error "The expected audio design is no longer loaded."
}

close_hw_target
disconnect_hw_server
close_hw_manager
