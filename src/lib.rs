//! Multi-provider failover for LLM calls.
//!
//! Wrap an ordered list of `(name, callable)` provider pairs. Each call tries
//! them in order. If a provider returns `Err`, the chain decides whether to fall
//! back or re-raise, then moves on. You get back a [`ChainResult`] with the
//! return value, the winning provider name, and a trace of every failed attempt.
//!
//! ```
//! use llm_fallback_chain::{FallbackChain, DynError};
//!
//! let chain = FallbackChain::<&str, String>::new(vec![
//!     ("anthropic", Box::new(|_p: &&str| -> Result<String, DynError> {
//!         Err("rate limited".into())
//!     }) as _),
//!     ("openai", Box::new(|p: &&str| -> Result<String, DynError> {
//!         Ok(format!("o:{}", p))
//!     }) as _),
//! ]).unwrap();
//!
//! let result = chain.call(&"hi").unwrap();
//! assert_eq!(result.value, "o:hi");
//! assert_eq!(result.provider, "openai");
//! assert_eq!(result.attempts.len(), 1);
//! ```
//!
//! Pluggable predicate to whitelist only certain errors:
//!
//! ```
//! use llm_fallback_chain::{FallbackChain, DynError};
//!
//! let chain = FallbackChain::<(), i32>::new(vec![
//!     ("a", Box::new(|_: &()| -> Result<i32, DynError> {
//!         Err("validation error".into())
//!     }) as _),
//!     ("b", Box::new(|_: &()| -> Result<i32, DynError> { Ok(1) }) as _),
//! ])
//! .unwrap()
//! .with_should_fall_back(|err| err.to_string().contains("rate"));
//!
//! // validation error is not "rate", so we do not fall back; the chain re-raises.
//! assert!(chain.call(&()).is_err());
//! ```

use std::error::Error as StdError;
use std::fmt;
use std::time::Instant;

/// Boxed dynamic error used throughout the chain. Providers return this so
/// every provider in the chain can fail in its own way.
pub type DynError = Box<dyn StdError + Send + Sync>;

/// One provider attempt within a chain call.
#[derive(Debug)]
pub struct Attempt {
    /// Provider name as passed to [`FallbackChain::new`].
    pub name: String,
    /// The error the provider returned (`None` on success).
    pub error: Option<DynError>,
    /// Wall time the provider took, in milliseconds.
    pub duration_ms: f64,
}

/// Outcome of a successful [`FallbackChain::call`].
#[derive(Debug)]
pub struct ChainResult<O> {
    /// Whatever the winning provider returned.
    pub value: O,
    /// Name of the provider that succeeded.
    pub provider: String,
    /// Failed attempts that came before the success. Empty when the first
    /// provider worked.
    pub attempts: Vec<Attempt>,
}

/// Raised when every provider in the chain failed.
#[derive(Debug)]
pub struct AllProvidersFailed {
    /// One [`Attempt`] per provider tried, in order.
    pub attempts: Vec<Attempt>,
}

impl fmt::Display for AllProvidersFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.attempts.iter().map(|a| a.name.as_str()).collect();
        write!(f, "all providers failed: {}", names.join(", "))
    }
}

impl StdError for AllProvidersFailed {}

/// Sync provider callable: `(input) -> Result<O, DynError>`.
pub type SyncProvider<I, O> = Box<dyn Fn(&I) -> Result<O, DynError> + Send + Sync>;

/// Predicate deciding whether a given error should cause fallback.
/// Default: any error causes fallback.
pub type ShouldFallBack = Box<dyn Fn(&DynError) -> bool + Send + Sync>;

/// Audit callback fired after a provider fails, before the next is tried.
/// `(failed_name, error, next_name)`.
pub type OnFallback = Box<dyn Fn(&str, &DynError, &str) + Send + Sync>;

fn default_should_fall_back(_err: &DynError) -> bool {
    true
}

/// Ordered list of LLM providers to try in sequence.
///
/// Each provider is a `(name, callable)` pair. [`call`](Self::call) tries them
/// in order until one returns `Ok`. Returns a [`ChainResult`] describing what
/// won and what failed. If every provider returns `Err`, an
/// [`AllProvidersFailed`] error is returned.
pub struct FallbackChain<I, O> {
    providers: Vec<(String, SyncProvider<I, O>)>,
    should_fall_back: ShouldFallBack,
    on_fallback: Option<OnFallback>,
}

impl<I, O> FallbackChain<I, O> {
    /// Build a new chain from an ordered list of `(name, provider)` pairs.
    ///
    /// Returns `Err` if `providers` is empty.
    pub fn new<S: Into<String>>(
        providers: Vec<(S, SyncProvider<I, O>)>,
    ) -> Result<Self, &'static str> {
        if providers.is_empty() {
            return Err("providers must be a non-empty list");
        }
        let providers = providers
            .into_iter()
            .map(|(name, fn_)| (name.into(), fn_))
            .collect();
        Ok(Self {
            providers,
            should_fall_back: Box::new(default_should_fall_back),
            on_fallback: None,
        })
    }

    /// Set the predicate that decides whether a given error should cause
    /// fallback. Default falls back on any error.
    #[must_use]
    pub fn with_should_fall_back<F>(mut self, f: F) -> Self
    where
        F: Fn(&DynError) -> bool + Send + Sync + 'static,
    {
        self.should_fall_back = Box::new(f);
        self
    }

    /// Set the audit callback fired after each fallback. Called with
    /// `(failed_name, error, next_name)` before the next provider is tried.
    #[must_use]
    pub fn with_on_fallback<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &DynError, &str) + Send + Sync + 'static,
    {
        self.on_fallback = Some(Box::new(f));
        self
    }

    /// Skip a provider entirely when `predicate(&name)` returns `true`.
    /// Useful when an upstream circuit breaker says a provider is open.
    /// Returns `Err` if the filter removes every provider.
    pub fn with_skip<P>(mut self, predicate: P) -> Result<Self, &'static str>
    where
        P: Fn(&str) -> bool,
    {
        self.providers.retain(|(name, _)| !predicate(name));
        if self.providers.is_empty() {
            return Err("with_skip removed all providers");
        }
        Ok(self)
    }

    /// Provider names in order.
    #[must_use]
    pub fn names(&self) -> Vec<&str> {
        self.providers.iter().map(|(n, _)| n.as_str()).collect()
    }

    /// Try each provider in order until one returns `Ok`.
    pub fn call(&self, input: &I) -> Result<ChainResult<O>, DynError> {
        let mut failures: Vec<Attempt> = Vec::new();
        let last = self.providers.len() - 1;
        for (i, (name, fn_)) in self.providers.iter().enumerate() {
            let start = Instant::now();
            match fn_(input) {
                Ok(value) => {
                    return Ok(ChainResult {
                        value,
                        provider: name.clone(),
                        attempts: failures,
                    });
                }
                Err(err) => {
                    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                    if !(self.should_fall_back)(&err) {
                        return Err(err);
                    }
                    if i < last {
                        if let Some(cb) = &self.on_fallback {
                            let next_name = &self.providers[i + 1].0;
                            cb(name, &err, next_name);
                        }
                    }
                    failures.push(Attempt {
                        name: name.clone(),
                        error: Some(err),
                        duration_ms: elapsed,
                    });
                }
            }
        }
        Err(Box::new(AllProvidersFailed { attempts: failures }))
    }
}

#[cfg(feature = "tokio")]
mod async_chain {
    use super::{
        default_should_fall_back, AllProvidersFailed, Attempt, ChainResult, DynError, OnFallback,
        ShouldFallBack,
    };
    use futures::future::BoxFuture;
    use std::time::Instant;

    /// Async provider callable: `(input) -> Future<Result<O, DynError>>`.
    /// The returned future borrows from the input for lifetime `'a`.
    pub type AsyncProvider<I, O> =
        Box<dyn for<'a> Fn(&'a I) -> BoxFuture<'a, Result<O, DynError>> + Send + Sync>;

    /// Helper to construct an [`AsyncProvider`] from a closure. Avoids the
    /// HRTB inference rough edge: the explicit function signature pins the
    /// `for<'a>` bound so callers can pass a plain `|input| async { ... }`
    /// without manual lifetime annotations.
    pub fn async_provider<I, O, F, Fut>(f: F) -> AsyncProvider<I, O>
    where
        F: for<'a> Fn(&'a I) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<O, DynError>> + Send + 'static,
        I: 'static,
    {
        Box::new(move |i: &I| {
            let fut = f(i);
            Box::pin(fut) as BoxFuture<'_, _>
        })
    }

    /// Async variant of [`crate::FallbackChain`]. Behaves the same way but
    /// awaits provider futures.
    pub struct AsyncFallbackChain<I, O> {
        providers: Vec<(String, AsyncProvider<I, O>)>,
        should_fall_back: ShouldFallBack,
        on_fallback: Option<OnFallback>,
    }

    impl<I: Send + Sync, O: Send> AsyncFallbackChain<I, O> {
        pub fn new<S: Into<String>>(
            providers: Vec<(S, AsyncProvider<I, O>)>,
        ) -> Result<Self, &'static str> {
            if providers.is_empty() {
                return Err("providers must be a non-empty list");
            }
            let providers = providers
                .into_iter()
                .map(|(name, fn_)| (name.into(), fn_))
                .collect();
            Ok(Self {
                providers,
                should_fall_back: Box::new(default_should_fall_back),
                on_fallback: None,
            })
        }

        #[must_use]
        pub fn with_should_fall_back<F>(mut self, f: F) -> Self
        where
            F: Fn(&DynError) -> bool + Send + Sync + 'static,
        {
            self.should_fall_back = Box::new(f);
            self
        }

        #[must_use]
        pub fn with_on_fallback<F>(mut self, f: F) -> Self
        where
            F: Fn(&str, &DynError, &str) + Send + Sync + 'static,
        {
            self.on_fallback = Some(Box::new(f));
            self
        }

        pub fn with_skip<P>(mut self, predicate: P) -> Result<Self, &'static str>
        where
            P: Fn(&str) -> bool,
        {
            self.providers.retain(|(name, _)| !predicate(name));
            if self.providers.is_empty() {
                return Err("with_skip removed all providers");
            }
            Ok(self)
        }

        #[must_use]
        pub fn names(&self) -> Vec<&str> {
            self.providers.iter().map(|(n, _)| n.as_str()).collect()
        }

        pub async fn call(&self, input: &I) -> Result<ChainResult<O>, DynError> {
            let mut failures: Vec<Attempt> = Vec::new();
            let last = self.providers.len() - 1;
            for (i, (name, fn_)) in self.providers.iter().enumerate() {
                let start = Instant::now();
                match fn_(input).await {
                    Ok(value) => {
                        return Ok(ChainResult {
                            value,
                            provider: name.clone(),
                            attempts: failures,
                        });
                    }
                    Err(err) => {
                        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                        if !(self.should_fall_back)(&err) {
                            return Err(err);
                        }
                        if i < last {
                            if let Some(cb) = &self.on_fallback {
                                let next_name = &self.providers[i + 1].0;
                                cb(name, &err, next_name);
                            }
                        }
                        failures.push(Attempt {
                            name: name.clone(),
                            error: Some(err),
                            duration_ms: elapsed,
                        });
                    }
                }
            }
            Err(Box::new(AllProvidersFailed { attempts: failures }))
        }
    }
}

#[cfg(feature = "tokio")]
pub use async_chain::{async_provider, AsyncFallbackChain, AsyncProvider};

#[cfg(feature = "serde")]
mod serde_impls {
    use super::Attempt;
    use serde::Serialize;

    /// Lossy serializable view of an [`Attempt`]. The error is recorded as its
    /// `Display` string because boxed `dyn Error` is not naturally `Serialize`.
    #[derive(Debug, Serialize)]
    pub struct AttemptView {
        pub name: String,
        pub error: Option<String>,
        pub duration_ms: f64,
    }

    impl From<&Attempt> for AttemptView {
        fn from(a: &Attempt) -> Self {
            Self {
                name: a.name.clone(),
                error: a.error.as_ref().map(|e| e.to_string()),
                duration_ms: a.duration_ms,
            }
        }
    }
}

#[cfg(feature = "serde")]
pub use serde_impls::AttemptView;
