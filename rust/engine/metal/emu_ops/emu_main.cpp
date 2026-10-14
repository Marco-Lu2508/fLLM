// emu_main.cpp - runner of the MSL emulator (package p06b). Reads fixture cases written by the unit test `emu_fixtures` of the planning layer
// (src/backend/metal/ops/mirror.rs), runs the kernels named in a case on the CPU with the grid, threadgroup size and parameter bytes of the
// plans, and compares the output buffers with what the engine's CPU backend computed. Build and run: emu_ops/run_emu.sh.
//
// case.txt (one directive per line; a case is a sequence of dispatches over named buffers):
//   buf NAME SOURCE                       SOURCE: a file name (initial bytes) or zero:BYTES
//   run KERNEL GX GY GZ TX TY TZ CONSTANTS BUF1 BUF2 ...  (CONSTANTS: hex bytes of the constant struct, bound at buffer 0; BUFn: buffers 1, 2, ...)
//   expect NAME FILE DTYPE ABS REL        DTYPE f32 | i32 | u8 (bytes, exact); f32: |a - b| <= ABS + REL |b|
#include "msl_emu.h"
// clang-format off
#include "unit.inc"
#include "glue.inc"
// clang-format on

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

static std::vector<unsigned char> read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", path.c_str());
        std::exit(2);
    }
    return std::vector<unsigned char>(std::istreambuf_iterator<char>(f), {});
}

struct Buffer {
    std::vector<unsigned char> storage;  // guard | data | guard
    size_t size = 0;
    static constexpr size_t kGuard = 256;
    unsigned char *data() { return storage.data() + kGuard; }
    void init(const std::vector<unsigned char> &bytes) {
        size = bytes.size();
        storage.assign(size + 2 * kGuard + 64, 0xCD);
        std::memcpy(data(), bytes.data(), size);
    }
    bool guards_intact() {
        for (size_t i = 0; i < kGuard; i++)
            if (storage[i] != 0xCD) return false;
        for (size_t i = kGuard + size; i < storage.size(); i++)
            if (storage[i] != 0xCD) return false;
        return true;
    }
};

static int run_case(const std::string &dir, const std::string &name) {
    std::ifstream f(dir + "/case.txt");
    if (!f) {
        std::printf("FAIL %s: no case.txt\n", name.c_str());
        return 1;
    }
    std::map<std::string, Buffer> bufs;
    struct Expect {
        std::string buf, file, dtype;
        double abs_tol, rel_tol;
    };
    std::vector<Expect> expects;
    std::vector<std::string> kernels;
    std::ostringstream report;
    int bad = 0;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string key;
        ls >> key;
        if (key == "buf") {
            std::string bname, source;
            ls >> bname >> source;
            if (source.rfind("zero:", 0) == 0) bufs[bname].init(std::vector<unsigned char>(std::stoul(source.substr(5)), 0));
            else bufs[bname].init(read_file(dir + "/" + source));
        } else if (key == "run") {
            std::string kname, params_hex, bname;
            unsigned grid[3], tpg[3];
            ls >> kname >> grid[0] >> grid[1] >> grid[2] >> tpg[0] >> tpg[1] >> tpg[2] >> params_hex;
            std::vector<unsigned char> params;
            if (params_hex != "-")
                for (size_t i = 0; i + 1 < params_hex.size(); i += 2) params.push_back(static_cast<unsigned char>(std::stoul(params_hex.substr(i, 2), nullptr, 16)));
            params.resize(params.size() + 64, 0);
            emu::Env env;
            env.bufs.push_back(nullptr);  // buffer 0 is the constants struct (not a buffer of the case)
            while (ls >> bname) {
                if (!bufs.count(bname)) {
                    report << " unknown buffer " << bname << ";";
                    bad++;
                    env.bufs.push_back(nullptr);
                } else {
                    env.bufs.push_back(bufs[bname].data());
                }
            }
            const KernelEntry *entry = nullptr;
            for (const auto &k : kKernels)
                if (kname == k.name) entry = &k;
            if (!entry) {
                report << " kernel " << kname << " is not in the emulator table;";
                bad++;
                continue;
            }
            kernels.push_back(kname);
            const unsigned char *pp = params.data();
            emu::run_grid(grid, tpg, [&](const emu::Attrs &a) { entry->run(env, pp, a); });
        } else if (key == "expect") {
            Expect e;
            ls >> e.buf >> e.file >> e.dtype >> e.abs_tol >> e.rel_tol;
            expects.push_back(e);
        }
    }

    for (auto &[n, b] : bufs)
        if (!b.guards_intact()) {
            report << " buffer " << n << " guard bytes overwritten;";
            bad++;
        }
    double worst_abs = 0, worst_rel = 0;
    for (const auto &e : expects) {
        const auto want = read_file(dir + "/" + e.file);
        Buffer &got = bufs[e.buf];
        if (want.size() != got.size) {
            report << " buffer " << e.buf << " size " << got.size << " != expected " << want.size() << ";";
            bad++;
            continue;
        }
        size_t mismatches = 0, first = 0;
        if (e.dtype == "f32") {
            for (size_t k = 0; k + 4 <= want.size(); k += 4) {
                float a, b;
                std::memcpy(&a, got.data() + k, 4);
                std::memcpy(&b, want.data() + k, 4);
                const double d = std::fabs(static_cast<double>(a) - static_cast<double>(b));
                const bool ok = d <= e.abs_tol + e.rel_tol * std::fabs(static_cast<double>(b)) && !(std::isnan(a) != std::isnan(b));
                worst_abs = std::max(worst_abs, d);
                if (std::fabs(b) > 1e-6) worst_rel = std::max(worst_rel, d / std::fabs(static_cast<double>(b)));
                if (!ok) {
                    if (!mismatches) first = k / 4;
                    mismatches++;
                }
            }
        } else {
            const size_t w = e.dtype == "i32" ? 4 : 1;
            for (size_t k = 0; k + w <= want.size(); k += w)
                if (std::memcmp(got.data() + k, want.data() + k, w) != 0) {
                    if (!mismatches) first = k / w;
                    mismatches++;
                }
        }
        if (mismatches) {
            report << " buffer " << e.buf << " (" << e.dtype << "): " << mismatches << " mismatches, first at element " << first << ";";
            bad++;
        }
    }
    std::string ks = kernels.empty() ? "-" : kernels[0] + (kernels.size() > 1 ? "+" + std::to_string(kernels.size() - 1) : "");
    if (bad) {
        std::printf("FAIL %s [%s]:%s\n", name.c_str(), ks.c_str(), report.str().c_str());
        return 1;
    }
    std::printf("ok   %s [%s] max abs %.3g max rel %.3g\n", name.c_str(), ks.c_str(), worst_abs, worst_rel);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <fixture dir> [case-name-filter]\n", argv[0]);
        return 2;
    }
    const std::string root = argv[1];
    const std::string filter = argc > 2 ? argv[2] : "";
    std::ifstream list(root + "/cases.txt");
    std::string name;
    int failed = 0, total = 0;
    while (std::getline(list, name)) {
        if (name.empty() || (!filter.empty() && name.find(filter) == std::string::npos)) continue;
        total++;
        failed += run_case(root + "/" + name, name);
    }
    std::printf("%d cases, %d failed\n", total, failed);
    return failed ? 1 : 0;
}
