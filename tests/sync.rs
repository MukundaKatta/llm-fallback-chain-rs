use llm_fallback_chain::{AllProvidersFailed, DynError, FallbackChain};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Debug)]
struct RateLimited(&'static str);
impl std::fmt::Display for RateLimited {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "rate limited: {}", self.0)
    }
}
impl std::error::Error for RateLimited {}

#[derive(Debug)]
struct Validation(&'static str);
impl std::fmt::Display for Validation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "validation: {}", self.0)
    }
}
impl std::error::Error for Validation {}

// ---------- construction ----------

#[test]
fn empty_providers_rejected() {
    let providers: Vec<(&str, llm_fallback_chain::SyncProvider<(), i32>)> = vec![];
    let err = match FallbackChain::new(providers) {
        Ok(_) => panic!("expected Err"),
        Err(e) => e,
    };
    assert!(err.contains("non-empty"));
}

#[test]
fn names_property_preserves_order() {
    let chain = FallbackChain::<(), &'static str>::new(vec![
        ("anthropic", Box::new(|_: &()| Ok("a")) as _),
        ("openai", Box::new(|_: &()| Ok("b")) as _),
        ("gemini", Box::new(|_: &()| Ok("c")) as _),
    ])
    .unwrap();
    assert_eq!(chain.names(), vec!["anthropic", "openai", "gemini"]);
}

// ---------- sync: success paths ----------

#[test]
fn first_provider_succeeds() {
    let chain = FallbackChain::<&str, String>::new(vec![
        (
            "anthropic",
            Box::new(|p: &&str| -> Result<String, DynError> { Ok(format!("a:{}", p)) }) as _,
        ),
        (
            "openai",
            Box::new(|p: &&str| -> Result<String, DynError> { Ok(format!("o:{}", p)) }) as _,
        ),
    ])
    .unwrap();
    let result = chain.call(&"hi").unwrap();
    assert_eq!(result.value, "a:hi");
    assert_eq!(result.provider, "anthropic");
    assert!(result.attempts.is_empty());
}

#[test]
fn second_provider_succeeds_after_first_fails() {
    let chain = FallbackChain::<&str, String>::new(vec![
        (
            "anthropic",
            Box::new(|_: &&str| -> Result<String, DynError> {
                Err(Box::new(RateLimited("anthropic out of quota")))
            }) as _,
        ),
        (
            "openai",
            Box::new(|p: &&str| -> Result<String, DynError> { Ok(format!("o:{}", p)) }) as _,
        ),
    ])
    .unwrap();
    let result = chain.call(&"hi").unwrap();
    assert_eq!(result.value, "o:hi");
    assert_eq!(result.provider, "openai");
    assert_eq!(result.attempts.len(), 1);
    assert_eq!(result.attempts[0].name, "anthropic");
    assert!(result.attempts[0].error.is_some());
    let err_string = result.attempts[0].error.as_ref().unwrap().to_string();
    assert!(err_string.contains("rate limited"));
    assert!(result.attempts[0].duration_ms >= 0.0);
}

#[test]
fn third_provider_succeeds_after_two_failures() {
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            Box::new(|_: &()| -> Result<&'static str, DynError> { Err("a".into()) }) as _,
        ),
        (
            "b",
            Box::new(|_: &()| -> Result<&'static str, DynError> { Err("b".into()) }) as _,
        ),
        ("c", Box::new(|_: &()| Ok("ok")) as _),
    ])
    .unwrap();
    let result = chain.call(&()).unwrap();
    assert_eq!(result.value, "ok");
    assert_eq!(result.provider, "c");
    let names: Vec<&str> = result.attempts.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(names, vec!["a", "b"]);
}

#[test]
fn input_is_forwarded() {
    #[derive(Debug)]
    struct In {
        a: i32,
        b: i32,
        mode: &'static str,
    }
    let chain = FallbackChain::<In, String>::new(vec![(
        "p",
        Box::new(|i: &In| -> Result<String, DynError> {
            Ok(format!("{}+{}={}@{}", i.a, i.b, i.a + i.b, i.mode))
        }) as _,
    )])
    .unwrap();
    let result = chain
        .call(&In {
            a: 1,
            b: 2,
            mode: "x",
        })
        .unwrap();
    assert_eq!(result.value, "1+2=3@x");
}

// ---------- sync: failure paths ----------

#[test]
fn all_providers_fail_raises_aggregate() {
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            Box::new(|_: &()| -> Result<&'static str, DynError> {
                Err(Box::new(RateLimited("a down")))
            }) as _,
        ),
        (
            "b",
            Box::new(|_: &()| -> Result<&'static str, DynError> {
                Err(Box::new(RateLimited("b down")))
            }) as _,
        ),
    ])
    .unwrap();
    let err = chain.call(&()).unwrap_err();
    let agg = err
        .downcast::<AllProvidersFailed>()
        .expect("AllProvidersFailed");
    assert_eq!(agg.attempts.len(), 2);
    assert_eq!(agg.attempts[0].name, "a");
    assert_eq!(agg.attempts[1].name, "b");
    for a in &agg.attempts {
        assert!(a
            .error
            .as_ref()
            .unwrap()
            .to_string()
            .contains("rate limited"));
    }
}

#[test]
fn custom_predicate_skips_non_retryable() {
    let openai_calls = Arc::new(AtomicUsize::new(0));
    let oc = openai_calls.clone();
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "anthropic",
            Box::new(|_: &()| -> Result<&'static str, DynError> {
                Err(Box::new(Validation("bad input")))
            }) as _,
        ),
        (
            "openai",
            Box::new(move |_: &()| -> Result<&'static str, DynError> {
                oc.fetch_add(1, Ordering::SeqCst);
                Ok("should not happen")
            }) as _,
        ),
    ])
    .unwrap()
    .with_should_fall_back(|err| err.downcast_ref::<RateLimited>().is_some());

    let err = chain.call(&()).unwrap_err();
    // re-raised the original Validation error, not AllProvidersFailed
    assert!(err.downcast_ref::<Validation>().is_some());
    assert!(err.downcast_ref::<AllProvidersFailed>().is_none());
    assert_eq!(openai_calls.load(Ordering::SeqCst), 0);
}

#[test]
fn custom_predicate_allows_fallback_on_whitelisted() {
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            Box::new(|_: &()| -> Result<&'static str, DynError> {
                Err(Box::new(RateLimited("retry me")))
            }) as _,
        ),
        ("b", Box::new(|_: &()| Ok("ok")) as _),
    ])
    .unwrap()
    .with_should_fall_back(|err| err.downcast_ref::<RateLimited>().is_some());
    let result = chain.call(&()).unwrap();
    assert_eq!(result.provider, "b");
}

// ---------- callback ----------

#[test]
fn on_fallback_callback_fires_for_each_fallback() {
    let calls: Arc<std::sync::Mutex<Vec<(String, String)>>> = Arc::new(Default::default());
    let cc = calls.clone();
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            Box::new(|_: &()| -> Result<&'static str, DynError> { Err(Box::new(RateLimited("a"))) })
                as _,
        ),
        (
            "b",
            Box::new(|_: &()| -> Result<&'static str, DynError> { Err(Box::new(RateLimited("b"))) })
                as _,
        ),
        ("c", Box::new(|_: &()| Ok("ok")) as _),
    ])
    .unwrap()
    .with_on_fallback(move |failed, _exc, next| {
        cc.lock()
            .unwrap()
            .push((failed.to_string(), next.to_string()));
    });
    let result = chain.call(&()).unwrap();
    assert_eq!(result.provider, "c");
    let got = calls.lock().unwrap().clone();
    assert_eq!(
        got,
        vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
        ]
    );
}

#[test]
fn on_fallback_not_called_when_first_succeeds() {
    let calls: Arc<std::sync::Mutex<Vec<String>>> = Arc::new(Default::default());
    let cc = calls.clone();
    let chain = FallbackChain::<(), &'static str>::new(vec![
        ("a", Box::new(|_: &()| Ok("ok")) as _),
        ("b", Box::new(|_: &()| Ok("x")) as _),
    ])
    .unwrap()
    .with_on_fallback(move |n, _e, _nxt| cc.lock().unwrap().push(n.to_string()));
    chain.call(&()).unwrap();
    assert!(calls.lock().unwrap().is_empty());
}

#[test]
fn on_fallback_not_called_after_last_provider() {
    let calls: Arc<std::sync::Mutex<Vec<(String, String)>>> = Arc::new(Default::default());
    let cc = calls.clone();
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            Box::new(|_: &()| -> Result<&'static str, DynError> { Err("a".into()) }) as _,
        ),
        (
            "b",
            Box::new(|_: &()| -> Result<&'static str, DynError> { Err("b".into()) }) as _,
        ),
    ])
    .unwrap()
    .with_on_fallback(move |f, _e, nxt| cc.lock().unwrap().push((f.to_string(), nxt.to_string())));
    let err = chain.call(&()).unwrap_err();
    assert!(err.downcast_ref::<AllProvidersFailed>().is_some());
    let got = calls.lock().unwrap().clone();
    assert_eq!(got, vec![("a".to_string(), "b".to_string())]);
}

// ---------- introspection / wiring ----------

#[test]
fn attempt_records_duration() {
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "slow",
            Box::new(|_: &()| -> Result<&'static str, DynError> {
                std::thread::sleep(std::time::Duration::from_millis(10));
                Err(Box::new(RateLimited("slow then fail")))
            }) as _,
        ),
        ("fast", Box::new(|_: &()| Ok("ok")) as _),
    ])
    .unwrap();
    let result = chain.call(&()).unwrap();
    assert!(result.attempts[0].duration_ms >= 5.0);
}

#[test]
fn all_providers_failed_error_message_lists_names() {
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "anthropic",
            Box::new(|_: &()| -> Result<&'static str, DynError> { Err("x".into()) }) as _,
        ),
        (
            "openai",
            Box::new(|_: &()| -> Result<&'static str, DynError> { Err("y".into()) }) as _,
        ),
    ])
    .unwrap();
    let err = chain.call(&()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("anthropic"));
    assert!(msg.contains("openai"));
}

#[test]
fn chain_is_reusable() {
    let counter = Arc::new(AtomicUsize::new(0));
    let cc = counter.clone();
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            Box::new(move |_: &()| -> Result<&'static str, DynError> {
                let n = cc.fetch_add(1, Ordering::SeqCst) + 1;
                if n == 1 {
                    Err(Box::new(RateLimited("first call only")))
                } else {
                    Ok("a")
                }
            }) as _,
        ),
        ("b", Box::new(|_: &()| Ok("b")) as _),
    ])
    .unwrap();
    let r1 = chain.call(&()).unwrap();
    assert_eq!(r1.provider, "b");
    assert_eq!(r1.attempts.len(), 1);
    let r2 = chain.call(&()).unwrap();
    assert_eq!(r2.provider, "a");
    assert!(r2.attempts.is_empty());
}

// ---------- with_skip ----------

#[test]
fn with_skip_removes_named_provider() {
    let chain = FallbackChain::<(), &'static str>::new(vec![
        ("a", Box::new(|_: &()| Ok("a")) as _),
        ("b", Box::new(|_: &()| Ok("b")) as _),
        ("c", Box::new(|_: &()| Ok("c")) as _),
    ])
    .unwrap()
    .with_skip(|name| name == "b")
    .unwrap();
    assert_eq!(chain.names(), vec!["a", "c"]);
    let result = chain.call(&()).unwrap();
    assert_eq!(result.provider, "a");
}

#[test]
fn with_skip_makes_b_chain_fall_to_c_when_a_fails() {
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            Box::new(|_: &()| -> Result<&'static str, DynError> { Err("down".into()) }) as _,
        ),
        ("b", Box::new(|_: &()| Ok("b")) as _),
        ("c", Box::new(|_: &()| Ok("c")) as _),
    ])
    .unwrap()
    .with_skip(|name| name == "b")
    .unwrap();
    let result = chain.call(&()).unwrap();
    assert_eq!(result.provider, "c");
    let attempt_names: Vec<&str> = result.attempts.iter().map(|a| a.name.as_str()).collect();
    assert_eq!(attempt_names, vec!["a"]);
}

#[test]
fn with_skip_rejects_when_all_filtered_out() {
    let chain = FallbackChain::<(), &'static str>::new(vec![
        ("a", Box::new(|_: &()| Ok("a")) as _),
        ("b", Box::new(|_: &()| Ok("b")) as _),
    ])
    .unwrap();
    let err = match chain.with_skip(|_| true) {
        Ok(_) => panic!("expected Err"),
        Err(e) => e,
    };
    assert!(err.contains("removed all"));
}

// ---------- error propagation through on_fallback ----------

#[test]
fn on_fallback_sees_error_text() {
    let last_err: Arc<std::sync::Mutex<Option<String>>> = Arc::new(Default::default());
    let lc = last_err.clone();
    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            Box::new(|_: &()| -> Result<&'static str, DynError> {
                Err(Box::new(RateLimited("a is down")))
            }) as _,
        ),
        ("b", Box::new(|_: &()| Ok("b")) as _),
    ])
    .unwrap()
    .with_on_fallback(move |_f, exc, _nxt| {
        *lc.lock().unwrap() = Some(exc.to_string());
    });
    chain.call(&()).unwrap();
    let got = last_err.lock().unwrap().clone().unwrap();
    assert!(got.contains("a is down"));
}
