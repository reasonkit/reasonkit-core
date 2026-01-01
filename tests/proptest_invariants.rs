//! Comprehensive invariant checking templates for property-based testing.
//!
//! This module provides reusable invariant checking patterns that can be
//! applied across all RK-PROJECT crates for property-based testing.

use proptest::prelude::*;
use std::collections::HashMap;

// ============================================================================
// INVARIANT CHECKING PATTERNS
// ============================================================================

/// Trait for types that have defined invariants
pub trait HasInvariants {
    /// Check all invariants for this type
    /// Returns Ok(()) if all invariants hold, Err(message) otherwise
    fn check_invariants(&self) -> Result<(), String>;
}

/// Trait for types that support serialization roundtrips
pub trait RoundtripTestable: Sized {
    /// Serialize and deserialize, returning the result
    fn roundtrip(&self) -> Result<Self, String>;

    /// Check if two instances are equivalent
    fn equivalent(&self, other: &Self) -> bool;
}

// ============================================================================
// NUMERIC INVARIANT PATTERNS
// ============================================================================

/// Check that a value is within a valid range
pub fn check_range<T: PartialOrd + std::fmt::Debug>(
    value: T,
    min: T,
    max: T,
    name: &str,
) -> Result<(), String> {
    if value >= min && value <= max {
        Ok(())
    } else {
        Err(format!(
            "{} value {:?} is outside valid range [{:?}, {:?}]",
            name, value, min, max
        ))
    }
}

/// Check that a confidence score is valid (0.0 to 1.0)
pub fn check_confidence(value: f64, name: &str) -> Result<(), String> {
    check_range(value, 0.0, 1.0, name)
}

/// Check that a score is non-negative
pub fn check_non_negative<T: PartialOrd + Default + std::fmt::Debug>(
    value: T,
    name: &str,
) -> Result<(), String> {
    if value >= T::default() {
        Ok(())
    } else {
        Err(format!("{} value {:?} must be non-negative", name, value))
    }
}

/// Check that a value is positive
pub fn check_positive<T: PartialOrd + Default + std::fmt::Debug + Copy>(
    value: T,
    name: &str,
) -> Result<(), String>
where
    T: std::ops::Add<Output = T> + From<u8>,
{
    let one: T = T::from(1);
    let min = T::default() + one;
    if value >= min {
        Ok(())
    } else {
        Err(format!("{} value {:?} must be positive", name, value))
    }
}

// ============================================================================
// STRING INVARIANT PATTERNS
// ============================================================================

/// Check that a string is non-empty
pub fn check_non_empty(value: &str, name: &str) -> Result<(), String> {
    if !value.is_empty() {
        Ok(())
    } else {
        Err(format!("{} cannot be empty", name))
    }
}

/// Check that a string has minimum length
pub fn check_min_length(value: &str, min: usize, name: &str) -> Result<(), String> {
    if value.len() >= min {
        Ok(())
    } else {
        Err(format!(
            "{} length {} is less than minimum {}",
            name,
            value.len(),
            min
        ))
    }
}

/// Check that a string matches a pattern (simplified check)
pub fn check_pattern(
    value: &str,
    allowed_chars: fn(char) -> bool,
    name: &str,
) -> Result<(), String> {
    if value.chars().all(allowed_chars) {
        Ok(())
    } else {
        Err(format!("{} contains invalid characters", name))
    }
}

/// Check identifier format (alphanumeric + underscore)
pub fn check_identifier(value: &str, name: &str) -> Result<(), String> {
    check_non_empty(value, name)?;
    check_pattern(value, |c| c.is_ascii_alphanumeric() || c == '_', name)
}

// ============================================================================
// COLLECTION INVARIANT PATTERNS
// ============================================================================

/// Check that a collection is non-empty
pub fn check_non_empty_collection<T>(items: &[T], name: &str) -> Result<(), String> {
    if !items.is_empty() {
        Ok(())
    } else {
        Err(format!("{} collection cannot be empty", name))
    }
}

/// Check that a collection has at least N items
pub fn check_min_count<T>(items: &[T], min: usize, name: &str) -> Result<(), String> {
    if items.len() >= min {
        Ok(())
    } else {
        Err(format!(
            "{} has {} items, minimum is {}",
            name,
            items.len(),
            min
        ))
    }
}

/// Check that a collection has at most N items
pub fn check_max_count<T>(items: &[T], max: usize, name: &str) -> Result<(), String> {
    if items.len() <= max {
        Ok(())
    } else {
        Err(format!(
            "{} has {} items, maximum is {}",
            name,
            items.len(),
            max
        ))
    }
}

/// Check that all items in a collection satisfy a predicate
pub fn check_all<T, F>(items: &[T], predicate: F, name: &str) -> Result<(), String>
where
    F: Fn(&T) -> bool,
{
    if items.iter().all(&predicate) {
        Ok(())
    } else {
        Err(format!("Not all items in {} satisfy the invariant", name))
    }
}

/// Check that at least one item in a collection satisfies a predicate
pub fn check_any<T, F>(items: &[T], predicate: F, name: &str) -> Result<(), String>
where
    F: Fn(&T) -> bool,
{
    if items.iter().any(&predicate) {
        Ok(())
    } else {
        Err(format!("No items in {} satisfy the invariant", name))
    }
}

// ============================================================================
// ORDERING INVARIANT PATTERNS
// ============================================================================

/// Check that a is less than or equal to b
pub fn check_le<T: PartialOrd + std::fmt::Debug>(
    a: T,
    b: T,
    name_a: &str,
    name_b: &str,
) -> Result<(), String> {
    if a <= b {
        Ok(())
    } else {
        Err(format!(
            "{} ({:?}) must be <= {} ({:?})",
            name_a, a, name_b, b
        ))
    }
}

/// Check that a is less than b
pub fn check_lt<T: PartialOrd + std::fmt::Debug>(
    a: T,
    b: T,
    name_a: &str,
    name_b: &str,
) -> Result<(), String> {
    if a < b {
        Ok(())
    } else {
        Err(format!(
            "{} ({:?}) must be < {} ({:?})",
            name_a, a, name_b, b
        ))
    }
}

/// Check that a is equal to b
pub fn check_eq<T: PartialEq + std::fmt::Debug>(a: T, b: T, name: &str) -> Result<(), String> {
    if a == b {
        Ok(())
    } else {
        Err(format!("{} values {:?} and {:?} must be equal", name, a, b))
    }
}

// ============================================================================
// TEMPORAL INVARIANT PATTERNS
// ============================================================================

/// Check that a timestamp is not in the future
pub fn check_not_future(
    timestamp: chrono::DateTime<chrono::Utc>,
    name: &str,
) -> Result<(), String> {
    let now = chrono::Utc::now();
    if timestamp <= now {
        Ok(())
    } else {
        Err(format!("{} timestamp {} is in the future", name, timestamp))
    }
}

/// Check that timestamp a is before or equal to timestamp b
pub fn check_temporal_order(
    before: chrono::DateTime<chrono::Utc>,
    after: chrono::DateTime<chrono::Utc>,
    name_before: &str,
    name_after: &str,
) -> Result<(), String> {
    if before <= after {
        Ok(())
    } else {
        Err(format!(
            "{} ({}) must be <= {} ({})",
            name_before, before, name_after, after
        ))
    }
}

// ============================================================================
// SEMANTIC INVARIANT PATTERNS
// ============================================================================

/// Check that if condition A is true, then condition B must also be true
pub fn check_implication(a: bool, b: bool, name_a: &str, name_b: &str) -> Result<(), String> {
    if !a || b {
        Ok(())
    } else {
        Err(format!(
            "If {} is true, then {} must be true",
            name_a, name_b
        ))
    }
}

/// Check that exactly one of the conditions is true (XOR)
pub fn check_exclusive(a: bool, b: bool, name_a: &str, name_b: &str) -> Result<(), String> {
    if a != b {
        Ok(())
    } else {
        Err(format!(
            "Exactly one of {} or {} must be true",
            name_a, name_b
        ))
    }
}

/// Check that all options are None or at least one is Some
pub fn check_all_or_none<T>(options: &[&Option<T>], name: &str) -> Result<(), String> {
    let none_count = options.iter().filter(|o| o.is_none()).count();
    if none_count == 0 || none_count == options.len() {
        Ok(())
    } else {
        Err(format!(
            "{}: all values must be Some or all must be None",
            name
        ))
    }
}

// ============================================================================
// COMPOSITE INVARIANT CHECKING
// ============================================================================

/// Check multiple invariants, collecting all failures
pub struct InvariantChecker {
    failures: Vec<String>,
}

impl InvariantChecker {
    pub fn new() -> Self {
        Self { failures: vec![] }
    }

    pub fn check(&mut self, result: Result<(), String>) -> &mut Self {
        if let Err(e) = result {
            self.failures.push(e);
        }
        self
    }

    pub fn finish(self) -> Result<(), Vec<String>> {
        if self.failures.is_empty() {
            Ok(())
        } else {
            Err(self.failures)
        }
    }

    pub fn is_ok(&self) -> bool {
        self.failures.is_empty()
    }
}

impl Default for InvariantChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PROPERTY TEST TEMPLATES
// ============================================================================

/// Template for testing numeric ranges
macro_rules! prop_test_range {
    ($name:ident, $type:ty, $min:expr, $max:expr) => {
        proptest! {
            #[test]
            fn $name(value in $min..$max) {
                prop_assert!(value >= $min && value <= $max);
            }
        }
    };
}

/// Template for testing non-empty strings
macro_rules! prop_test_non_empty {
    ($name:ident, $pattern:expr) => {
        proptest! {
            #[test]
            fn $name(s in $pattern) {
                prop_assert!(!s.is_empty());
            }
        }
    };
}

/// Template for testing serialization roundtrip
macro_rules! prop_test_roundtrip {
    ($name:ident, $type:ty, $strategy:expr) => {
        proptest! {
            #[test]
            fn $name(value in $strategy) {
                let serialized = serde_json::to_string(&value)
                    .expect("Failed to serialize");
                let deserialized: $type = serde_json::from_str(&serialized)
                    .expect("Failed to deserialize");
                // Type-specific equality check would go here
            }
        }
    };
}

// ============================================================================
// EXAMPLE USAGE TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_range() {
        assert!(check_range(0.5, 0.0, 1.0, "test").is_ok());
        assert!(check_range(0.0, 0.0, 1.0, "test").is_ok());
        assert!(check_range(1.0, 0.0, 1.0, "test").is_ok());
        assert!(check_range(1.5, 0.0, 1.0, "test").is_err());
        assert!(check_range(-0.1, 0.0, 1.0, "test").is_err());
    }

    #[test]
    fn test_check_confidence() {
        assert!(check_confidence(0.5, "conf").is_ok());
        assert!(check_confidence(0.0, "conf").is_ok());
        assert!(check_confidence(1.0, "conf").is_ok());
        assert!(check_confidence(1.1, "conf").is_err());
        assert!(check_confidence(-0.1, "conf").is_err());
    }

    #[test]
    fn test_check_non_empty() {
        assert!(check_non_empty("hello", "str").is_ok());
        assert!(check_non_empty("", "str").is_err());
    }

    #[test]
    fn test_check_min_length() {
        assert!(check_min_length("hello", 3, "str").is_ok());
        assert!(check_min_length("hi", 3, "str").is_err());
    }

    #[test]
    fn test_check_identifier() {
        assert!(check_identifier("valid_id", "id").is_ok());
        assert!(check_identifier("ValidId123", "id").is_ok());
        assert!(check_identifier("", "id").is_err());
        assert!(check_identifier("invalid-id", "id").is_err());
    }

    #[test]
    fn test_check_non_empty_collection() {
        assert!(check_non_empty_collection(&[1, 2, 3], "items").is_ok());
        assert!(check_non_empty_collection::<i32>(&[], "items").is_err());
    }

    #[test]
    fn test_check_min_max_count() {
        assert!(check_min_count(&[1, 2, 3], 2, "items").is_ok());
        assert!(check_min_count(&[1], 2, "items").is_err());
        assert!(check_max_count(&[1, 2, 3], 5, "items").is_ok());
        assert!(check_max_count(&[1, 2, 3, 4, 5, 6], 5, "items").is_err());
    }

    #[test]
    fn test_check_le_lt() {
        assert!(check_le(5, 10, "a", "b").is_ok());
        assert!(check_le(10, 10, "a", "b").is_ok());
        assert!(check_le(15, 10, "a", "b").is_err());
        assert!(check_lt(5, 10, "a", "b").is_ok());
        assert!(check_lt(10, 10, "a", "b").is_err());
    }

    #[test]
    fn test_check_implication() {
        // If A then B
        assert!(check_implication(true, true, "A", "B").is_ok()); // T -> T = T
        assert!(check_implication(true, false, "A", "B").is_err()); // T -> F = F
        assert!(check_implication(false, true, "A", "B").is_ok()); // F -> T = T
        assert!(check_implication(false, false, "A", "B").is_ok()); // F -> F = T
    }

    #[test]
    fn test_check_exclusive() {
        assert!(check_exclusive(true, false, "A", "B").is_ok());
        assert!(check_exclusive(false, true, "A", "B").is_ok());
        assert!(check_exclusive(true, true, "A", "B").is_err());
        assert!(check_exclusive(false, false, "A", "B").is_err());
    }

    #[test]
    fn test_invariant_checker() {
        let mut checker = InvariantChecker::new();
        checker
            .check(check_range(0.5, 0.0, 1.0, "a"))
            .check(check_non_empty("hello", "b"))
            .check(check_min_count(&[1, 2, 3], 2, "c"));

        assert!(checker.is_ok());
        assert!(checker.finish().is_ok());
    }

    #[test]
    fn test_invariant_checker_with_failures() {
        let mut checker = InvariantChecker::new();
        checker
            .check(check_range(1.5, 0.0, 1.0, "a")) // Fails
            .check(check_non_empty("", "b")) // Fails
            .check(check_min_count(&[1, 2, 3], 2, "c")); // Passes

        assert!(!checker.is_ok());
        let failures = checker.finish().unwrap_err();
        assert_eq!(failures.len(), 2);
    }
}

// ============================================================================
// PROPERTY TESTS FOR INVARIANT CHECKERS
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn prop_check_range_accepts_valid(value in 0.0f64..=1.0) {
        prop_assert!(check_range(value, 0.0, 1.0, "test").is_ok());
    }

    #[test]
    fn prop_check_range_rejects_below(value in -100.0f64..-0.001) {
        prop_assert!(check_range(value, 0.0, 1.0, "test").is_err());
    }

    #[test]
    fn prop_check_range_rejects_above(value in 1.001f64..100.0) {
        prop_assert!(check_range(value, 0.0, 1.0, "test").is_err());
    }

    #[test]
    fn prop_check_identifier_valid(id in "[a-zA-Z_][a-zA-Z0-9_]{2,20}") {
        prop_assert!(check_identifier(&id, "test").is_ok());
    }

    #[test]
    fn prop_check_le_reflexive(value in any::<i32>()) {
        prop_assert!(check_le(value, value, "a", "a").is_ok());
    }

    #[test]
    fn prop_check_le_transitive(
        a in 0i32..100,
        b in 100i32..200,
        c in 200i32..300
    ) {
        // If a <= b and b <= c, then a <= c
        if check_le(a, b, "a", "b").is_ok() && check_le(b, c, "b", "c").is_ok() {
            prop_assert!(check_le(a, c, "a", "c").is_ok());
        }
    }

    #[test]
    fn prop_implication_tautology(a in any::<bool>()) {
        // A -> A is always true
        prop_assert!(check_implication(a, a, "A", "A").is_ok());
    }

    #[test]
    fn prop_exclusive_complement(a in any::<bool>()) {
        // A XOR !A is always true
        prop_assert!(check_exclusive(a, !a, "A", "not_A").is_ok());
    }
}
