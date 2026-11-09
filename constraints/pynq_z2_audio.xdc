# PYNQ-Z2 audio self-test constraints
# Device: xc7z020clg400-1

# 125 MHz Ethernet-PHY clock. Keep an active Ethernet link for this PL-only
# design; the PHY clock can become unstable without a link.
set_property -dict {PACKAGE_PIN H16 IOSTANDARD LVCMOS33} [get_ports sys_clk]
create_clock -period 8.000 -name sys_clk [get_ports sys_clk]

# BTN0 resets the clock generator, codec configuration, and audio stream.
set_property -dict {PACKAGE_PIN D19 IOSTANDARD LVCMOS33} [get_ports rst_btn]

# SW0/SW1: 00=440 Hz tone, 01=raw I2S loopback,
#          10=decoded I2S loopback, 11=complete PCM processing path.
set_property -dict {PACKAGE_PIN M20 IOSTANDARD LVCMOS33} [get_ports source_sw]
set_property -dict {PACKAGE_PIN M19 IOSTANDARD LVCMOS33} [get_ports gain_sw]

# User LEDs: MCLK lock, codec ready, signal active, I2C error.
set_property -dict {PACKAGE_PIN R14 IOSTANDARD LVCMOS33} [get_ports {led[0]}]
set_property -dict {PACKAGE_PIN P14 IOSTANDARD LVCMOS33} [get_ports {led[1]}]
set_property -dict {PACKAGE_PIN N16 IOSTANDARD LVCMOS33} [get_ports {led[2]}]
set_property -dict {PACKAGE_PIN M14 IOSTANDARD LVCMOS33} [get_ports {led[3]}]

# ADAU1761 master clock and standard I2S interface.
set_property -dict {PACKAGE_PIN U5 IOSTANDARD LVCMOS33 DRIVE 8 SLEW FAST} [get_ports codec_mclk]
set_property -dict {PACKAGE_PIN R18 IOSTANDARD LVCMOS33} [get_ports audio_bclk]
set_property -dict {PACKAGE_PIN T17 IOSTANDARD LVCMOS33} [get_ports audio_lrclk]
set_property -dict {PACKAGE_PIN F17 IOSTANDARD LVCMOS33} [get_ports audio_sdata_in]
set_property -dict {PACKAGE_PIN G18 IOSTANDARD LVCMOS33 DRIVE 8 SLEW FAST} [get_ports audio_sdata_out]

# ADAU1761 control interface. Both lines are open drain.
set_property -dict {PACKAGE_PIN U9 IOSTANDARD LVCMOS33 PULLUP TRUE} [get_ports audio_i2c_scl]
set_property -dict {PACKAGE_PIN T9 IOSTANDARD LVCMOS33 PULLUP TRUE} [get_ports audio_i2c_sda]
set_property -dict {PACKAGE_PIN M17 IOSTANDARD LVCMOS33} [get_ports {codec_addr[0]}]
set_property -dict {PACKAGE_PIN M18 IOSTANDARD LVCMOS33} [get_ports {codec_addr[1]}]

# Codec master mode produces 3.072 MHz BCLK for 48 kHz stereo, 32-bit slots.
create_clock -add -period 325.521 -name audio_bclk [get_ports audio_bclk]
set_property CLOCK_DEDICATED_ROUTE FALSE [get_nets audio_bclk_IBUF]

# Samples cross these domains through explicit toggle synchronizers.
set_clock_groups -asynchronous \
    -group [get_clocks sys_clk] \
    -group [get_clocks audio_bclk]
