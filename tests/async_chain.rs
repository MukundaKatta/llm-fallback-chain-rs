#![cfg(feature = "tokio")]

use llm_fallback_chain::{async_provider, AllProvidersFailed, AsyncFallbackChain, DynError};
use std::sync::Arc;
use std::sync::Mutex;

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

#[tokio::test]
async fn async_first_succeeds() {
    let chain = AsyncFallbackChain::<(), &'static str>::new(vec![(
        "anthropic",
        async_provider(|_: &()| async { Ok::<_, DynError>("a") }),
    )])
    .unwrap();
    let result = chain.call(&()).await.unwrap();
    assert_eq!(result.value, "a");
    assert_eq!(result.provider, "anthropic");
}

#[tokio::test]
async fn async_second_succeeds_after_first_fails() {
    let chain = AsyncFallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            async_provider(|_: &()| async {
                Err::<&'static str, DynError>(Box::new(RateLimited("nope")))
            }),
        ),
        (
            "b",
            async_provider(|_: &()| async { Ok::<_, DynError>("b") }),
        ),
    ])
    .unwrap();
    let result = chain.call(&()).await.unwrap();
    assert_eq!(result.value, "b");
    assert_eq!(result.provider, "b");
    assert_eq!(result.attempts.len(), 1);
}

#[tokio::test]
async fn async_all_fail_raises() {
    let chain = AsyncFallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            async_provider(|_: &()| async {
                Err::<&'static str, DynError>(Box::new(RateLimited("a")))
            }),
        ),
        (
            "b",
            async_provider(|_: &()| async {
                Err::<&'static str, DynError>(Box::new(RateLimited("b")))
            }),
        ),
    ])
    .unwrap();
    let err = chain.call(&()).await.unwrap_err();
    assert!(err.downcast_ref::<AllProvidersFailed>().is_some());
}

#[tokio::test]
async fn async_callback_fires() {
    let calls: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Default::default());
    let cc = calls.clone();
    let chain = AsyncFallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            async_provider(|_: &()| async {
                Err::<&'static str, DynError>(Box::new(RateLimited("a")))
            }),
        ),
        (
            "b",
            async_provider(|_: &()| async { Ok::<_, DynError>("b") }),
        ),
    ])
    .unwrap()
    .with_on_fallback(move |f, _e, n| {
        cc.lock().unwrap().push((f.to_string(), n.to_string()));
    });
    chain.call(&()).await.unwrap();
    let got = calls.lock().unwrap().clone();
    assert_eq!(got, vec![("a".to_string(), "b".to_string())]);
}

#[tokio::test]
async fn async_custom_predicate_skips_non_retryable() {
    let chain = AsyncFallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            async_provider(|_: &()| async {
                Err::<&'static str, DynError>(Box::new(Validation("bad")))
            }),
        ),
        (
            "b",
            async_provider(|_: &()| async { Ok::<_, DynError>("b") }),
        ),
    ])
    .unwrap()
    .with_should_fall_back(|err| err.downcast_ref::<RateLimited>().is_some());
    let err = chain.call(&()).await.unwrap_err();
    assert!(err.downcast_ref::<Validation>().is_some());
    assert!(err.downcast_ref::<AllProvidersFailed>().is_none());
}

#[tokio::test]
async fn async_with_skip_removes_named_provider() {
    let chain = AsyncFallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            async_provider(|_: &()| async {
                Err::<&'static str, DynError>(Box::new(RateLimited("a")))
            }),
        ),
        (
            "b",
            async_provider(|_: &()| async { Ok::<_, DynError>("b") }),
        ),
        (
            "c",
            async_provider(|_: &()| async { Ok::<_, DynError>("c") }),
        ),
    ])
    .unwrap()
    .with_skip(|n| n == "b")
    .unwrap();
    assert_eq!(chain.names(), vec!["a", "c"]);
    let result = chain.call(&()).await.unwrap();
    assert_eq!(result.provider, "c");
}
