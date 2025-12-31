// ============================================================
// ReasonKit Brand Constants
// ============================================================
// CANONICAL SOURCE - Synchronized with reasonkit-site/main.css
// Reference: BRAND_IDENTITY.md
// Updated: 2025-12-30 (v1.0.0 - Launch)
// ============================================================

/// Brand color constants (Landing Page Standard)
pub mod colors {
    // Primary Palette
    pub const CYAN: &str = "#06b6d4";
    pub const CYAN_RGB: (u8, u8, u8) = (6, 182, 212);

    pub const GREEN: &str = "#10b981";
    pub const GREEN_RGB: (u8, u8, u8) = (16, 185, 129);

    pub const PURPLE: &str = "#a855f7";
    pub const PURPLE_RGB: (u8, u8, u8) = (168, 85, 247);

    pub const PINK: &str = "#ec4899";
    pub const PINK_RGB: (u8, u8, u8) = (236, 72, 153);

    pub const ORANGE: &str = "#f97316";
    pub const ORANGE_RGB: (u8, u8, u8) = (249, 115, 22);

    pub const YELLOW: &str = "#fbbf24";
    pub const YELLOW_RGB: (u8, u8, u8) = (251, 191, 36);

    // Backgrounds
    pub const BG_VOID: &str = "#030508";
    pub const BG_VOID_RGB: (u8, u8, u8) = (3, 5, 8);

    pub const BG_DEEP: &str = "#0a0d14";
    pub const BG_DEEP_RGB: (u8, u8, u8) = (10, 13, 20);

    pub const BG_SURFACE: &str = "#111827";
    pub const BG_SURFACE_RGB: (u8, u8, u8) = (17, 24, 39);

    pub const BG_ELEVATED: &str = "#1f2937";
    pub const BG_ELEVATED_RGB: (u8, u8, u8) = (31, 41, 55);

    // Text Colors
    pub const TEXT_PRIMARY: &str = "#f9fafb";
    pub const TEXT_PRIMARY_RGB: (u8, u8, u8) = (249, 250, 251);

    pub const TEXT_SECONDARY: &str = "#9ca3af";
    pub const TEXT_SECONDARY_RGB: (u8, u8, u8) = (156, 163, 175);

    pub const TEXT_MUTED: &str = "#6b7280";
    pub const TEXT_MUTED_RGB: (u8, u8, u8) = (107, 114, 128);

    pub const TEXT_DIM: &str = "#4b5563";
    pub const TEXT_DIM_RGB: (u8, u8, u8) = (75, 85, 99);

    // Semantic aliases (for backwards compatibility)
    pub const PRIMARY: &str = CYAN;
    pub const SECONDARY: &str = PURPLE;
    pub const BACKGROUND: &str = BG_VOID;
    pub const SURFACE: &str = BG_DEEP;
    pub const SUCCESS: &str = GREEN;
    pub const ERROR: &str = ORANGE;
    pub const WARNING: &str = YELLOW;
    pub const ACCENT: &str = PURPLE;

    // Border colors
    pub const BORDER: &str = "rgba(6, 182, 212, 0.2)";
    pub const BORDER_SUBTLE: &str = "rgba(255, 255, 255, 0.05)";
    pub const BORDER_MUTED: &str = "rgba(255, 255, 255, 0.1)";

    // Glow colors
    pub const GLOW_CYAN: &str = "rgba(6, 182, 212, 0.3)";
    pub const GLOW_PURPLE: &str = "rgba(168, 85, 247, 0.3)";
    pub const GLOW_PINK: &str = "rgba(236, 72, 153, 0.3)";
    pub const GLOW_GREEN: &str = "rgba(16, 185, 129, 0.3)";
    pub const GLOW_ORANGE: &str = "rgba(249, 115, 22, 0.3)";
}

/// Brand typography constants
pub mod typography {
    pub const FONT_HEADLINE: &str = "Inter";
    pub const FONT_BODY: &str = "Inter";
    pub const FONT_DISPLAY: &str = "Playfair Display";
    pub const FONT_CODE: &str = "JetBrains Mono";

    pub const FALLBACK_SANS: &str = "-apple-system, BlinkMacSystemFont, sans-serif";
    pub const FALLBACK_SERIF: &str = "Georgia, serif";
    pub const FALLBACK_MONO: &str = "SF Mono, Consolas, monospace";
}

/// Brand gradients
pub mod gradients {
    pub const HERO: &str = "linear-gradient(135deg, #06b6d4 0%, #a855f7 50%, #ec4899 100%)";
    pub const CYAN_PURPLE: &str = "linear-gradient(135deg, #06b6d4, #a855f7)";
    pub const PURPLE_PINK: &str = "linear-gradient(135deg, #a855f7, #ec4899)";
    pub const TEXT: &str = "linear-gradient(90deg, #06b6d4, #a855f7, #ec4899)";
}

/// Brand identity
pub mod identity {
    pub const NAME: &str = "ReasonKit";
    pub const TAGLINE: &str = "Turn Prompts into Protocols";
    pub const PHILOSOPHY: &str = "Designed, Not Dreamed";
    pub const POSITIONING: &str = "Structure Beats Intelligence";
    pub const WEBSITE: &str = "https://reasonkit.sh";
}

/// Version info
pub mod version {
    pub const BRAND_VERSION: &str = "1.0.0";
    pub const BRAND_UPDATED: &str = "2025-12-30";
}
