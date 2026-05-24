#![cfg(feature = "serde")]

use llm_fallback_chain::{Attempt, AttemptView, DynError, FallbackChain};

#[test]
fn attempt_view_serializes_failed_attempt() {
    #[derive(Debug)]
    struct E;
    impl std::fmt::Display for E {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "boom")
        }
    }
    impl std::error::Error for E {}

    let chain = FallbackChain::<(), &'static str>::new(vec![
        (
            "a",
            Box::new(|_: &()| -> Result<&'static str, DynError> { Err(Box::new(E)) }) as _,
        ),
        ("b", Box::new(|_: &()| Ok("b")) as _),
    ])
    .unwrap();
    let result = chain.call(&()).unwrap();
    let view: AttemptView = (&result.attempts[0]).into();
    let json = serde_json::to_string(&view).unwrap();
    assert!(json.contains("\"name\":\"a\""));
    assert!(json.contains("\"error\":\"boom\""));
    assert!(json.contains("\"duration_ms\""));
}

#[test]
fn attempt_view_handles_none_error() {
    let view = AttemptView::from(&Attempt {
        name: "p".to_string(),
        error: None,
        duration_ms: 0.0,
    });
    let json = serde_json::to_string(&view).unwrap();
    assert!(json.contains("\"name\":\"p\""));
    assert!(json.contains("\"error\":null"));
}
