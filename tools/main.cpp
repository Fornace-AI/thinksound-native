// tools/main.cpp — CLI frontend for native ThinkSound Metal generator.
#include "models.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

static const char * opt(int argc, char ** argv, const char * key, const char * def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == key) return argv[i + 1];
    }
    return def;
}

int main(int argc, char ** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        printf("usage: %s [options]\n", argv[0]);
        printf("  --caption <text>      caption for CLIP text conditioning (default: 'a dog barking')\n");
        printf("  --cot <text>          Chain-of-thought description for T5 (default: 'A dog barks...')\n");
        printf("  --dir <path>          GGUF weights and metallib directory (default: /tmp/ts-weights)\n");
        printf("  --out <path>          output WAV path (default: out.wav)\n");
        printf("  --dur <sec>           duration in seconds [1..30] (default: 8)\n");
        printf("  --steps <int>         Euler sampling steps (default: 24)\n");
        printf("  --cfg <float>         Classifier-free guidance scale (default: 5.0)\n");
        printf("  --seed <uint>         RNG seed (default: 4242)\n");
        return 0;
    }

    tn::GenOpts opts;
    opts.caption = opt(argc, argv, "--caption", "a dog barking");
    opts.cot     = opt(argc, argv, "--cot", "A dog barks several times, sharp and clear, in a quiet room.");
    opts.dir     = opt(argc, argv, "--dir", "/tmp/ts-weights");
    opts.dur     = std::atoi(opt(argc, argv, "--dur", "8"));
    opts.steps   = std::atoi(opt(argc, argv, "--steps", "24"));
    opts.cfg     = std::atof(opt(argc, argv, "--cfg", "5.0"));
    opts.seed    = (unsigned) std::strtoul(opt(argc, argv, "--seed", "4242"), nullptr, 10);
    std::string out_path = opt(argc, argv, "--out", "out.wav");

    printf("=== ThinkSound Native Metal Generator ===\n");
    printf("caption: \"%s\"\n", opts.caption.c_str());
    printf("cot:     \"%s\"\n", opts.cot.c_str());
    printf("dir:     %s\n", opts.dir.c_str());
    printf("dur:     %d s, steps: %d, cfg: %.2f, seed: %u\n", opts.dur, opts.steps, opts.cfg, opts.seed);
    printf("out:     %s\n", out_path.c_str());

    auto t0 = std::chrono::steady_clock::now();
    std::string err;
    int frames = tn::generate_wav(opts, out_path, err);
    auto t1 = std::chrono::steady_clock::now();

    if (frames <= 0) {
        fprintf(stderr, "generation failed: %s\n", err.c_str());
        return 1;
    }

    double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
    double audio_s = (double) frames / tn::SAMPLE_RATE;
    double rtf = elapsed_s / audio_s;
    printf("\nGenerated %d frames (%.2f s audio) in %.3f s (RTF: %.3fx)\n",
           frames, audio_s, elapsed_s, rtf);
    printf("Saved to %s\n", out_path.c_str());
    return 0;
}
