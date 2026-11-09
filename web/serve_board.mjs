process.env.FPGA_API_URL ||= "http://169.254.165.100:8081";
process.env.FPGA_VISION_URL ||= "http://169.254.165.100:8082";
// PC address on the physical board link; override when using another adapter.
process.env.FPGA_BIND_ADDRESS ||= "169.254.165.99";
process.env.PORT ||= "8001";
await import("./server.mjs");
