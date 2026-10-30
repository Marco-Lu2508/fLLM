use super::*;
use crate::rust_decoder::RustDecoder;
use crate::tiny_models::{build_gguf, decode, scratch, Arch, Encoding, PROMPT};
use crate::GgufModel;

fn inline(budget: Budget) -> Options {
    Options { budget, prefetch: true, demand: true, evict: true, random_access: true, inline: true }
}

#[test]
fn budgets_parse_with_units() {
    assert_eq!(parse_budget("auto"), Ok(Budget::Auto));
    assert_eq!(parse_budget("off"), Ok(Budget::Off));
    assert_eq!(parse_budget("4096"), Ok(Budget::Bytes(4096 << 20)));
    assert_eq!(parse_budget("512M"), Ok(Budget::Bytes(512 << 20)));
    assert_eq!(parse_budget("8GiB"), Ok(Budget::Bytes(8 << 30)));
    assert_eq!(parse_budget("1.5g"), Ok(Budget::Bytes(3 << 29)));
    assert!(parse_budget("lots").is_err());
    assert!(parse_budget("-3").is_err());
    assert!(parse_budget("4X").is_err());
}

#[test]
fn speed_classes_follow_the_share_that_fits() {
    assert_eq!(speed_class(false, 1.0), "in-memory");
    assert_eq!(speed_class(true, 0.9), "mostly cached");
    assert_eq!(speed_class(true, 0.5), "streaming");
    assert_eq!(speed_class(true, 0.1), "disk-bound");
    let plan = Plan { expert_bytes: 1000, budget: 250, automatic: false, streaming: true, enforce: true, layers: 4, active_bytes_per_token: 400, class: "streaming" };
    assert_eq!(plan.coverage(), 0.25);
    assert_eq!(plan.cold_bytes_per_token(), 300);
    assert!(plan.describe().contains("25 % of the experts"));
}

/// Four experts cut out of the stacked down projection of a tiny quantized MoE model (real file-backed bytes), and the size of one.
fn four_experts(label: &str) -> (Vec<Vec<TensorBytes>>, u64, std::path::PathBuf) {
    let dir = scratch(label);
    let gguf = build_gguf(&dir, Arch::Qwen3Moe, Encoding::Quant("q8_0"), 7);
    let model = GgufModel::load(gguf.to_str().unwrap(), 64, 1, 2).unwrap();
    let whole = model.mapping().owned_tensor_bytes("blk.0.ffn_down_exps.weight").unwrap();
    let each = whole.len() / 4;
    let parts = (0..4).map(|e| vec![whole.slice(e * each, each).unwrap()]).collect();
    (parts, each as u64, dir)
}

fn cache_with(options: Options, name: &str, parts: &[Vec<TensorBytes>], layers: usize) -> (ExpertCache, Vec<Arc<LayerTable>>) {
    let cache = with_options(options, || ExpertCache::new(name)).unwrap();
    let tables = (0..layers).map(|l| cache.register_layer(l, 2, parts).unwrap()).collect();
    (cache, tables)
}

#[test]
fn hottest_experts_are_pinned_and_the_oldest_are_dropped_first() {
    let (parts, each, dir) = four_experts("pin");
    // 12 experts, room for four: two may be pinned (half of the budget), the rest is the window of recent ones
    let (cache, tables) = cache_with(inline(Budget::Bytes(4 * each)), "sized", &parts, 3);
    let plan = cache.finalize(0);
    assert!(plan.streaming && plan.layers == 3, "{plan:?}");
    // layer 0 uses experts 0 and 1 on every pass; the other layers rotate through theirs
    for pass in 0..40usize {
        tables[0].on_route(&[0, 1], 1);
        tables[1].on_route(&[pass % 4, (pass + 1) % 4], 1);
        tables[2].on_route(&[(pass + 2) % 4, (pass + 3) % 4], 1);
    }
    assert!(tables[0].entries[0].hot.load(Relaxed) && tables[0].entries[1].hot.load(Relaxed), "the experts used on every pass are pinned");
    let s = cache.snapshot();
    assert!(s.evicted_bytes > 0, "the budget forces experts out");
    assert!(s.resident_bytes <= plan.budget + 6 * each, "resident {} budget {}", s.resident_bytes, plan.budget);
    assert!(tables[0].entries[0].resident.load(Relaxed), "a pinned expert stays");
    assert_eq!(s.passes, 40);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn a_budget_that_holds_everything_never_evicts_and_reports_in_memory() {
    let (parts, _, dir) = four_experts("fits");
    let (cache, tables) = cache_with(inline(Budget::Bytes(u64::MAX / 2)), "fits", &parts, 1);
    let plan = cache.finalize(0);
    assert!(!plan.streaming);
    assert_eq!(plan.class, "in-memory");
    for pass in 0..20usize {
        tables[0].on_route(&[pass % 4, (pass + 1) % 4], 1);
    }
    cache.flush();
    let s = cache.snapshot();
    assert_eq!((s.evicted_bytes, s.prefetch_issued), (0, 0));
    assert_eq!(s.activations, 40);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn device_resident_layers_do_not_count_as_host_experts() {
    let (parts, each, dir) = four_experts("device");
    let (cache, tables) = cache_with(inline(Budget::Bytes(1 << 20)), "device", &parts, 3);
    tables[1].mark_device_resident();
    let plan = cache.finalize(0);
    assert_eq!(plan.layers, 2);
    assert_eq!(plan.expert_bytes, 8 * each);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn the_next_layer_is_predicted_from_its_previous_pass() {
    let (parts, _, dir) = four_experts("predict");
    let (cache, tables) = cache_with(inline(Budget::Bytes(1)), "predict", &parts, 2);
    cache.finalize(0);
    for _ in 0..3 {
        tables[0].on_route(&[1, 3], 1);
        tables[1].on_route(&[0, 2], 1);
    }
    assert_eq!(tables[1].predict(), vec![0, 2]);
    // the wrap-around: the first layer of the next pass follows the last layer of this one
    assert!(tables[1].next.get().and_then(Weak::upgrade).is_some_and(|t| t.layer() == 0));
    assert_eq!(tables[0].predict(), vec![1, 3]);
    let _ = std::fs::remove_dir_all(dir);
}

#[test]
fn heap_weights_are_not_registered() {
    let cache = with_options(inline(Budget::Auto), || ExpertCache::new("heap")).unwrap();
    assert!(cache.register_layer(0, 2, &[vec![TensorBytes::from_vec(vec![1, 2, 3])]]).is_none());
}

fn logits_with(gguf: &std::path::Path, options: Options, prompt: &[i64]) -> (Vec<Vec<u32>>, Option<Snapshot>) {
    let model = GgufModel::load(gguf.to_str().unwrap(), 64, 1, 2).unwrap();
    let dec = with_options(options, || RustDecoder::load(&model, "cpu")).unwrap();
    let run = decode(&dec, &model, prompt, 12, None).unwrap();
    let snapshot = dec.expert_cache().map(|c| {
        c.flush();
        c.snapshot()
    });
    (run.logits.iter().map(|l| l.iter().map(|v| v.to_bits()).collect()).collect(), snapshot)
}

/// The cache only hints the operating system: the logits must not depend on the budget.
fn same_logits_for_every_budget(label: &str, arch: Arch, encoding: Encoding) {
    let dir = scratch(label);
    let gguf = build_gguf(&dir, arch, encoding, 0x1234_5678_9abc_def1);
    let long: Vec<i64> = (0..40).map(|i| (i * 7 + 3) % 120).collect();
    for prompt in [&PROMPT[..], &long[..]] {
        let (unlimited, free) = logits_with(&gguf, inline(Budget::Bytes(1 << 40)), prompt);
        let (tiny, small) = logits_with(&gguf, inline(Budget::Bytes(1)), prompt);
        let (threaded, _) = logits_with(&gguf, Options { inline: false, ..inline(Budget::Bytes(1)) }, prompt);
        let (off, none) = logits_with(&gguf, inline(Budget::Off), prompt);
        assert_eq!(unlimited, tiny, "{label}: a budget of one byte changed the logits");
        assert_eq!(unlimited, threaded, "{label}: the background thread changed the logits");
        assert_eq!(unlimited, off, "{label}: switching the cache off changed the logits");
        assert!(none.is_none());
        let (free, small) = (free.unwrap(), small.unwrap());
        assert!(!free.streaming && small.streaming, "{label}: modes {} / {}", free.class, small.class);
        assert!(small.activations > 0 && small.evicted_bytes > 0 && small.expert_bytes > 0, "{label}: {small:?}");
        assert_eq!(free.evicted_bytes, 0);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn qwen3_moe_logits_do_not_depend_on_the_expert_budget() {
    same_logits_for_every_budget("budget-qwen3moe", Arch::Qwen3Moe, Encoding::Quant("q4_k_m"));
}

#[test]
fn deepseek_logits_do_not_depend_on_the_expert_budget() {
    same_logits_for_every_budget("budget-deepseek", Arch::Deepseek, Encoding::Quant("q8_0"));
}

#[test]
fn mixtral_logits_do_not_depend_on_the_expert_budget() {
    same_logits_for_every_budget("budget-mixtral", Arch::Mixtral, Encoding::Quant("q4_0"));
}
