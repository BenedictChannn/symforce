#include <symforce/examples/residual/unit_test.cc>
#include <spdlog/spdlog.h>

int main() {
    for (int i = 0; i < 6; i++) {
        spdlog::info("Iteration {}: ", i);
        event_res::unit_test();
    }
}