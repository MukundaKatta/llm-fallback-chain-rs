# llm-fallback-chain

Multi-provider failover for LLM calls. Try provider A; on failure, try B; on B failure, try C. Stop on the first one that returns `Ok`.

This is the cross-provider failover piece. Not retry-with-backoff against one provider (use [`llm-retry`](https://crates.io/crates/llm-retry) for that) and not a circuit breaker (use [`llm-circuit-breaker`](https://crates.io/crates/llm-circuit-breaker)). All three compose.

## Install

```toml
[dependencies]
llm-fallback-chain = "0.1"
```

Async variant gated behind a feature:

```toml
[dependencies]
llm-fallback-chain = { version = "0.1", features = ["tokio"] }
```

## Sync

```rust
use llm_fallback_chain::{FallbackChain, DynError};

let chain = FallbackChain::<&str, String>::new(vec![
    ("anthropic", Box::new(|p: &&str| -> Result<String, DynError> {
        Err("rate limited".into())
    }) as _),
    ("openai", Box::new(|p: &&str| -> Result<String, DynError> {
        Ok(format!("o:{}", p))
    }) as _),
])?;

let result = chain.call(&"hello")?;
assert_eq!(result.provider, "openai");
assert_eq!(result.value, "o:hello");
assert_eq!(result.attempts.len(), 1); // anthropic failed before openai won
```

## Skip predicate

Skip a provider entirely when something else (an open circuit breaker, a feature flag, an outage signal) tells you not to try it:

```rust
let chain = FallbackChain::<(), &'static str>::new(providers)?
    .with_skip(|name| breaker_is_open(name))?;
```

## Custom should-fall-back predicate

Default: any error causes fallback. Restrict to specific error types:

```rust
let chain = FallbackChain::<(), String>::new(providers)?
    .with_should_fall_back(|err| err.downcast_ref::<RateLimited>().is_some());
```

A non-matching error re-raises as-is instead of triggering fallback.

## Audit hook

```rust
let chain = chain.with_on_fallback(|failed, err, next| {
    log::warn!("{} failed ({}), trying {}", failed, err, next);
});
```

The hook fires after each failure and before the next provider is tried. It does not fire for the last provider because there is no next.

## Async (feature = "tokio")

```rust
use llm_fallback_chain::{async_provider, AsyncFallbackChain, DynError};

let chain = AsyncFallbackChain::<(), String>::new(vec![
    ("anthropic", async_provider(|_: &()| async {
        Err::<String, DynError>("rate limited".into())
    })),
    ("openai", async_provider(|_: &()| async {
        Ok::<_, DynError>("hi".to_string())
    })),
])?;

let result = chain.call(&()).await?;
```

## All-failed error

When every provider in the chain fails, `call` returns a `DynError` containing an `AllProvidersFailed` with one `Attempt` per provider:

```rust
let err = chain.call(&()).unwrap_err();
let failed = err.downcast_ref::<AllProvidersFailed>().unwrap();
for attempt in &failed.attempts {
    println!("{}: {}", attempt.name, attempt.error.as_ref().unwrap());
}
```

## Optional serde

Enable `serde` to get `AttemptView`, a lossy view of `Attempt` that serializes (the boxed `dyn Error` becomes its `Display` string):

```toml
llm-fallback-chain = { version = "0.1", features = ["serde"] }
```

## License

MIT.
