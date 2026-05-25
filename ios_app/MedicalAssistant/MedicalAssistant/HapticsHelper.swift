import Foundation
#if canImport(UIKit)
import UIKit
#endif

// MARK: - Haptic Feedback Manager with Debouncing

/// Provides debounced haptic feedback to avoid Taptic Engine saturation
/// during high-frequency token generation.
enum HapticsHelper {
    #if canImport(UIKit)
    private static let heavyGenerator = UIImpactFeedbackGenerator(style: .heavy)
    private static let mediumGenerator = UIImpactFeedbackGenerator(style: .medium)
    #endif

    /// Minimum interval between token ticks (~6 per second max).
    private static let tokenTickInterval: TimeInterval = 0.15

    /// Tracks the last time a token haptic was fired.
    private static var lastTokenTickDate: Date = .distantPast

    /// Sentence-ending characters that trigger a stronger "thought complete" bump.
    private static let sentenceEndings: Set<Character> = [".", "!", "?", "\n"]

    // MARK: - Public API

    /// Light impact for discrete user actions (send message, etc.).
    static func playLightImpact() {
        #if canImport(UIKit)
        heavyGenerator.prepare()
        heavyGenerator.impactOccurred(intensity: 1.0)
        #endif
    }

    /// Debounced token tick — call this for every chunk during streaming.
    ///
    /// - **Time-based debounce**: Ignores calls arriving < 0.15 s after the last tick.
    /// - **Sentence-end detection**: If `chunk` contains a sentence-ending character
    ///   (`. ! ? \n`), a slightly stronger `.light` impact fires regardless of debounce,
    ///   giving a satisfying "thought complete" bump.
    static func playTokenTick(for chunk: String) {
        let now = Date()
        let elapsed = now.timeIntervalSince(lastTokenTickDate)

        // Check for sentence-end — fires a stronger bump immediately
        let hasSentenceEnd = chunk.contains(where: { sentenceEndings.contains($0) })

        if hasSentenceEnd {
            lastTokenTickDate = now
            #if canImport(UIKit)
            heavyGenerator.prepare()
            heavyGenerator.impactOccurred(intensity: 0.85)
            #endif
            return
        }

        // Time-based debounce for regular token ticks
        guard elapsed >= tokenTickInterval else { return }
        lastTokenTickDate = now

        #if canImport(UIKit)
        mediumGenerator.prepare()
        mediumGenerator.impactOccurred(intensity: 0.7)
        #endif
    }
}
