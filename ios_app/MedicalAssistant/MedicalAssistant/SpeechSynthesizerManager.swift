import AVFoundation
import Foundation

/// Lightweight on-device text-to-speech manager for accessibility and hands-free medical consultations.
@MainActor
final class SpeechSynthesizerManager: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    static let shared = SpeechSynthesizerManager()

    private let synthesizer = AVSpeechSynthesizer()
    @Published var speakingMessageId: UUID?

    override private init() {
        super.init()
        synthesizer.delegate = self
    }

    /// Speaks the given message text. If this message is already speaking, it toggles off.
    func speak(messageId: UUID, text: String, language: String = "fr-FR") {
        if speakingMessageId == messageId {
            stop()
            return
        }

        stop()

        let cleaned = cleanMarkdownForSpeech(text)
        guard !cleaned.isEmpty else { return }

        let utterance = AVSpeechUtterance(string: cleaned)
        // Prefer French voice if text has French characteristics or requested, fallback to user locale / English
        if let voice = AVSpeechSynthesisVoice(language: language) {
            utterance.voice = voice
        } else if let preferred = AVSpeechSynthesisVoice(language: Locale.current.identifier) {
            utterance.voice = preferred
        } else {
            utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        }

        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1.0

        speakingMessageId = messageId
        synthesizer.speak(utterance)
    }

    /// Stops any active speech.
    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        speakingMessageId = nil
    }

    func isSpeaking(messageId: UUID) -> Bool {
        speakingMessageId == messageId
    }

    // MARK: - AVSpeechSynthesizerDelegate

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.speakingMessageId = nil
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.speakingMessageId = nil
        }
    }

    // MARK: - Helpers

    /// Strips markdown artifacts (headers, bullet markers, backticks, bold/italics) for natural speech.
    private func cleanMarkdownForSpeech(_ markdown: String) -> String {
        var text = markdown
        // Remove code blocks
        text = text.replacingOccurrences(of: "```[a-zA-Z]*", with: "", options: .regularExpression)
        // Remove headers
        text = text.replacingOccurrences(of: "#{1,6}\\s*", with: "", options: .regularExpression)
        // Remove bold / italic symbols
        text = text.replacingOccurrences(of: "\\*\\*", with: "")
        text = text.replacingOccurrences(of: "\\*", with: "")
        text = text.replacingOccurrences(of: "__", with: "")
        text = text.replacingOccurrences(of: "`", with: "")
        // Remove list bullets
        text = text.replacingOccurrences(of: "(?m)^[\\*\\-\\+]\\s+", with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: "(?m)^\\d+\\.\\s+", with: "", options: .regularExpression)
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
