import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

// MARK: - Design Tokens

enum ChatTheme {

    // MARK: Colors

    static let electricBlue = Color(red: 0 / 255, green: 122 / 255, blue: 255 / 255)
    static let deepViolet = Color(red: 88 / 255, green: 86 / 255, blue: 214 / 255)
    static let glassStroke = Color.white.opacity(0.20)
    static let glassHighlight = Color.white.opacity(0.10)
    static let backgroundBase = Color(red: 0.06, green: 0.06, blue: 0.10)
    static let backgroundSecondary = Color(red: 0.10, green: 0.12, blue: 0.20)

    /// Surface for AI message cards (adapts to dark/light automatically via Material).
    static let aiCardBackground: some ShapeStyle = .ultraThinMaterial

    /// Subtle surface behind the input bar.
    static let inputBarBackground: some ShapeStyle = .thinMaterial

    /// Very subtle separator shade.
    static let separator = Color.primary.opacity(0.08)

    // MARK: Spacing & Radii

    static let bubbleRadius: CGFloat = 22
    static let bubbleRadiusSmall: CGFloat = 6
    static let inputRadius: CGFloat = 24
    static let messagePaddingH: CGFloat = 14
    static let messagePaddingV: CGFloat = 10
    static let avatarSize: CGFloat = 28
    static let glassLineWidth: CGFloat = 0.5

    // MARK: Fonts

    static let headerTitle = Font.system(.title3, design: .rounded).weight(.bold)
    static let headerSubtitle = Font.system(.caption, design: .rounded)
    static let messageBody = Font.system(.body, design: .rounded)
    static let messageCaption = Font.system(.caption2, design: .rounded)
    static let inputFont = Font.system(.body, design: .rounded)
    static let disclaimerFont = Font.system(.caption2, design: .rounded)
    static let emptyStateTitle = Font.system(.title2, design: .rounded).weight(.semibold)
    static let emptyStateBody = Font.system(.subheadline, design: .rounded)

    // MARK: Shadows

    static let bubbleShadow = Color.black.opacity(0.08)
    static let bubbleShadowRadius: CGFloat = 6
    static let bubbleShadowY: CGFloat = 2

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
                accent.opacity(0.72),
                deepViolet.opacity(0.82)
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
