set project_file "D:/FPGA/ZynqAudioLab/ZynqAudioLab.xpr"
open_project $project_file

set rtl_files [list \
    "D:/FPGA/ZynqAudioLab/rtl/i2c_write_master.v" \
    "D:/FPGA/ZynqAudioLab/rtl/adau1761_init.v" \
    "D:/FPGA/ZynqAudioLab/rtl/audio_mclk_gen.v" \
    "D:/FPGA/ZynqAudioLab/rtl/pynq_audio_board_top.v"]

foreach rtl_file $rtl_files {
    if {[llength [get_files -quiet $rtl_file]] == 0} {
        add_files -norecurse $rtl_file
    }
}

set tb_file "D:/FPGA/ZynqAudioLab/sim/tb_adau1761_init.v"
if {[llength [get_files -quiet $tb_file]] == 0} {
    add_files -fileset sim_1 -norecurse $tb_file
}

set xdc_file "D:/FPGA/ZynqAudioLab/constraints/pynq_z2_audio.xdc"
if {[llength [get_files -quiet $xdc_file]] == 0} {
    add_files -fileset constrs_1 -norecurse $xdc_file
}

set_property top pynq_audio_board_top [get_filesets sources_1]
set_property top tb_pynq_audio_stream_top [get_filesets sim_1]
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
close_project
