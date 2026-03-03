import SwiftUI

// MARK: - Design Tokens

enum ChatTheme {

    // MARK: Colors

    /// Gradient used for user message bubbles.
    static let userGradient = LinearGradient(
        colors: [Color(hue: 0.72, saturation: 0.65, brightness: 0.95),   // soft indigo
                 Color(hue: 0.60, saturation: 0.70, brightness: 0.98)],  // bright blue
        startPoint: .topLeading,
        endPoint: .bottomTrailing
    )

    /// Accent color matching the gradient's primary hue.
    static let accent = Color(hue: 0.66, saturation: 0.68, brightness: 0.96)

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
}
