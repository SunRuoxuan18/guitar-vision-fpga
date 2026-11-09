#define main full_reference_main
#include "test_hand_conv_q16.cpp"
#undef main

int main() {
    bool ok = true;
    ok &= check(3,5,3,9,1,1,0,0,0,0,1);
    ok &= check(7,9,3,11,3,2,0,1,1,0,1);
    ok &= check(7,9,11,11,3,1,1,1,1,1,11);
    ok &= check(3,17,8,9,1,1,0,0,0,0,1);
    ok &= check(5,19,9,9,5,2,2,1,1,2,9);
    return ok ? 0 : 1;
}
