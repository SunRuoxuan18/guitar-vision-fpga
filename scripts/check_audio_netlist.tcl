set root "D:/FPGA/ZynqAudioLab"
open_checkpoint "$root/diagnostics/ps_clock_test/audio_ps_clock.runs/impl_1/audio_ps_clock_wrapper_placed.dcp"
foreach port_name {audio_i2c_scl audio_i2c_sda} {
    set port [get_ports $port_name]
    puts "AUDIO_PORT: $port_name direction=[get_property DIRECTION $port]"
    set pins [get_pins -leaf -of_objects [get_nets -segments -of_objects $port]]
    foreach pin $pins {
        set cell [get_cells -of_objects $pin]
        puts "AUDIO_PAD_CELL: $cell type=[get_property REF_NAME $cell] pin=$pin"
    }
}
puts "IOBUFS: [get_cells -hier -filter {REF_NAME == IOBUF}]"
puts "OBUFTS: [get_cells -hier -filter {REF_NAME == OBUFT}]"
close_design
