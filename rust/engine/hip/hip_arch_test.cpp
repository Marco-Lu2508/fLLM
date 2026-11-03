// Host-only test: hip_arch.h against hip/arch_table.csv.   g++ -std=c++17 hip_arch_test.cpp -o t && ./t arch_table.csv
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include "hip_arch.h"

using namespace fllm_hip_arch;

static std::vector<std::string> split(const std::string &s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string part;
    while (std::getline(ss, part, ',')) out.push_back(part);
    return out;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s arch_table.csv\n", argv[0]); return 2; }
    std::ifstream in(argv[1]);
    std::string line;
    std::getline(in, line);  // header
    int rows = 0, bad = 0;
    std::map<std::string, bool> seen;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto c = split(line);  // gfx,family,wave,matrix,use,dot4,dot2,generic
        if (c.size() != 8) { fprintf(stderr, "bad row: %s\n", line.c_str()); return 2; }
        const Info i = classify(c[0] + ":sramecc+:xnack-");
        const Use use = c[4] == "cdna" ? USE_CDNA : c[4] == "rdna3" ? USE_RDNA3 : c[4] == "rdna4" ? USE_RDNA4 : USE_OTHER;
        const Dot4 dot4 = c[5] == "sdot4" ? DOT4_SDOT4 : c[5] == "sudot4" ? DOT4_SUDOT4 : DOT4_NONE;
        rows++;
        seen[c[0]] = true;
        if (!i.known || i.use != use || i.wave != std::stoi(c[2]) || i.dot4 != dot4 || i.dot2 != (c[6] == "1")) {
            fprintf(stderr, "MISMATCH %s: header says known=%d use=%d wave=%d dot4=%d dot2=%d\n", c[0].c_str(), i.known, i.use, i.wave, i.dot4, i.dot2);
            bad++;
        }
    }
    int n = 0;
    const Row *table_rows = table(&n);
    for (int k = 0; k < n; k++)
        if (!seen[table_rows[k].gfx]) { fprintf(stderr, "header row %s is not in the csv\n", table_rows[k].gfx); bad++; }
    if (n != rows) { fprintf(stderr, "header has %d rows, csv %d\n", n, rows); bad++; }
    // unknown names get the conservative class
    const Info u = classify("gfx9999");
    if (u.known || u.use != USE_OTHER || u.dot4 != DOT4_NONE || u.dot2 || u.wave != 64) { fprintf(stderr, "gfx9999 is not conservative\n"); bad++; }
    const Info v = classify("gfx1999");
    if (v.known || v.use != USE_OTHER || v.wave != 32) { fprintf(stderr, "gfx1999 is not conservative\n"); bad++; }
    const Info w = classify("something");
    if (w.known || w.use != USE_OTHER || w.wave != 0) { fprintf(stderr, "unknown name is not conservative\n"); bad++; }
    printf("%d rows checked, %d problems\n", rows, bad);
    return bad ? 1 : 0;
}
