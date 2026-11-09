connect -url tcp:127.0.0.1:3121
puts "PS_TARGETS:"
targets
targets -set -filter {name =~ "APU"}
puts "BOOT_MODE: [mrd 0xF800025C]"
puts "IO_PLL_CTRL: [mrd 0xF8000108]"
puts "PLL_STATUS: [mrd 0xF800010C]"
puts "FPGA0_CLK_CTRL: [mrd 0xF8000170]"
puts "FPGA0_THR_CTRL: [mrd 0xF8000178]"
puts "FPGA_RST_CTRL: [mrd 0xF8000240]"
puts "LVL_SHFTR_EN: [mrd 0xF8000900]"
disconnect
