//! Settings > Model details and *Save as default* against a real engine (only on request):
//!
//! ```text
//! fllm-rs studio --model a.gguf --port 18090 &
//! FLLM_LIVE_ENGINE=http://127.0.0.1:18090 cargo test --test details_live -- --ignored --nocapture
//! ```

use fllm_app::modeldetails;
use fllm_app::store::Prefs;

#[tokio::test]
#[ignore = "needs a running engine with a model: see the top of the file"]
async fn the_facts_the_names_and_the_defaults_work_against_the_engine() {
    let base = std::env::var("FLLM_LIVE_ENGINE").expect("FLLM_LIVE_ENGINE is not set (see the top of this file)");
    let facts = modeldetails::fetch(base.clone()).await.expect("the status is read").expect("a model is loaded");
    println!("{} | {} | {}", facts.model_line(), facts.length_line(), facts.quantization);
    println!("chat format: {} characters, aliases {:?}", facts.chat_template.len(), facts.aliases);
    assert!(!facts.arch.is_empty() && facts.parameters > 0 && facts.size_bytes > 0 && facts.vocab_size > 0, "the facts are there: {facts:?}");

    // another name for the loaded model, and away again
    let added = modeldetails::add_alias(base.clone(), facts.name.clone(), "gpt-4o-mini".into()).await.expect("the name is added");
    println!("{added}");
    let after = modeldetails::fetch(base.clone()).await.unwrap().unwrap();
    assert!(after.aliases.iter().any(|a| a == "gpt-4o-mini"), "the server lists it: {:?}", after.aliases);
    let removed = modeldetails::remove_alias(base.clone(), "gpt-4o-mini".into()).await.expect("the name goes");
    println!("{removed}");
    let gone = modeldetails::fetch(base.clone()).await.unwrap().unwrap();
    assert!(!gone.aliases.iter().any(|a| a == "gpt-4o-mini"), "and it is gone: {:?}", gone.aliases);

    // the defaults: manual values and a prompt are accepted
    let mut prefs = Prefs { manual_sampling: true, temperature: 0.4, top_p: 0.9, custom_system: true, system_prompt: "Answer in one sentence.".into(), ..Default::default() };
    fllm_app::models::save_defaults(base.clone(), fllm_app::settings::defaults_body(&prefs)).await.expect("the defaults are accepted");
    // a value the server refuses comes back as its reason
    prefs.temperature = -1.0;
    let refused = fllm_app::models::save_defaults(base.clone(), fllm_app::settings::defaults_body(&prefs)).await;
    println!("refused: {refused:?}");
    assert!(refused.is_err());
    // the state of the server is put back
    prefs.temperature = 0.7;
    prefs.custom_system = false;
    fllm_app::models::save_defaults(base, fllm_app::settings::defaults_body(&prefs)).await.unwrap();
}
