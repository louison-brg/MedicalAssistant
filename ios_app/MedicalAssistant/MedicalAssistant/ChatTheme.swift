import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

// MARK: - Design Tokens

enum ChatTheme {

    // MARK: Colors

    static let electricBlue = Color(red: 52 / 255, green: 136 / 255, blue: 255 / 255)
    static let sky = Color(red: 112 / 255, green: 192 / 255, blue: 255 / 255)
    static let auroraCyan = Color(red: 64 / 255, green: 212 / 255, blue: 245 / 255)
    static let royalBlue = Color(red: 24 / 255, green: 104 / 255, blue: 230 / 255)
    static let mint = Color(red: 83 / 255, green: 224 / 255, blue: 170 / 255)
    static let emerald = Color(red: 52 / 255, green: 211 / 255, blue: 153 / 255)
    static let teal = Color(red: 16 / 255, green: 122 / 255, blue: 137 / 255)
    static let navy = Color(red: 6 / 255, green: 12 / 255, blue: 29 / 255)
    static let indigo = Color(red: 24 / 255, green: 31 / 255, blue: 72 / 255)
    static let deepViolet = Color(red: 88 / 255, green: 86 / 255, blue: 214 / 255)
    static let coral = Color(red: 255 / 255, green: 115 / 255, blue: 125 / 255)
    static let amber = Color(red: 255 / 255, green: 185 / 255, blue: 70 / 255)

    static let glassStroke = Color.white.opacity(0.18)
    static let glassHighlight = Color.white.opacity(0.12)
    static let backgroundBase = Color(red: 0.032, green: 0.045, blue: 0.086)
    static let backgroundSecondary = Color(red: 0.08, green: 0.11, blue: 0.19)

    /// Surface for AI message cards (adapts to dark/light automatically via Material).
    static let aiCardBackground: some ShapeStyle = .ultraThinMaterial

    /// Subtle surface behind the input bar.
    static let inputBarBackground: some ShapeStyle = .thinMaterial

    /// Very subtle separator shade.
    static let separator = Color.primary.opacity(0.08)

    // MARK: Gradients

    static let userBubbleGradient = LinearGradient(
        colors: [
            Color(red: 45 / 255, green: 135 / 255, blue: 255 / 255),
            Color(red: 20 / 255, green: 95 / 255, blue: 225 / 255)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let heroCrossGradient = LinearGradient(
        colors: [auroraCyan, electricBlue, deepViolet],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    static let glassBorderGradient = LinearGradient(
        colors: [
            Color.white.opacity(0.28),
            Color.white.opacity(0.08),
            Color.white.opacity(0.03)
        ],
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    // MARK: Spacing & Radii

    static let bubbleRadius: CGFloat = 22
    static let bubbleRadiusSmall: CGFloat = 6
    static let inputRadius: CGFloat = 26
    static let messagePaddingH: CGFloat = 16
    static let messagePaddingV: CGFloat = 12
    static let avatarSize: CGFloat = 30
    static let glassLineWidth: CGFloat = 0.6

    // MARK: Fonts

    static let headerTitle = Font.system(.headline, design: .rounded).weight(.bold)
    static let headerSubtitle = Font.system(.caption2, design: .rounded).weight(.medium)
    static let messageBody = Font.system(.body, design: .rounded)
    static let messageCaption = Font.system(.caption2, design: .rounded)
    static let inputFont = Font.system(.body, design: .rounded)
    static let disclaimerFont = Font.system(.caption2, design: .rounded)
    static let emptyStateTitle = Font.system(.title2, design: .rounded).weight(.bold)
    static let emptyStateBody = Font.system(.subheadline, design: .rounded)

    // MARK: Shadows

    static let bubbleShadow = Color.black.opacity(0.12)
    static let bubbleShadowRadius: CGFloat = 8
    static let bubbleShadowY: CGFloat = 3

    static func accentColor(load: Double) -> Color {
        #if canImport(UIKit)
        let progress = min(max(load, 0), 1)
        return Color(blend(UIColor(electricBlue), with: UIColor(deepViolet), progress: progress))
        #else
        return load > 0.5 ? deepViolet : electricBlue
        #endif
    }

    static func accentGradient(load: Double) -> LinearGradient {
        let accent = accentColor(load: load)
        return LinearGradient(
            colors: [
                accent.opacity(0.96),
                accent.opacity(0.80),
                deepViolet.opacity(0.85)
            ],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    #if canImport(UIKit)
    private static func blend(_ from: UIColor, with to: UIColor, progress: Double) -> UIColor {
        let clampedProgress = CGFloat(min(max(progress, 0), 1))
        var fromRed: CGFloat = 0
        var fromGreen: CGFloat = 0
        var fromBlue: CGFloat = 0
        var fromAlpha: CGFloat = 0
        var toRed: CGFloat = 0
        var toGreen: CGFloat = 0
        var toBlue: CGFloat = 0
        var toAlpha: CGFloat = 0
        from.getRed(&fromRed, green: &fromGreen, blue: &fromBlue, alpha: &fromAlpha)
        to.getRed(&toRed, green: &toGreen, blue: &toBlue, alpha: &toAlpha)

        return UIColor(
            red: fromRed + (toRed - fromRed) * clampedProgress,
            green: fromGreen + (toGreen - fromGreen) * clampedProgress,
            blue: fromBlue + (toBlue - fromBlue) * clampedProgress,
            alpha: fromAlpha + (toAlpha - fromAlpha) * clampedProgress
        )
    }
    #endif
}
