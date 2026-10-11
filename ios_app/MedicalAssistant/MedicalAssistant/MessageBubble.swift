import MarkdownUI
import SwiftUI
#if canImport(UIKit)
import UIKit
#endif

struct MessageBubble: View {
    let message: Message
    let accentColor: Color
    var onEdit: ((String) -> Void)?
    var onRegenerate: (() -> Void)?

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @ObservedObject private var speechManager = SpeechSynthesizerManager.shared
    @State private var appeared = false
    @State private var isEditing = false
    @State private var editText = ""
    @State private var isCopied = false
    @FocusState private var isEditFocused: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if message.isUser {
                Spacer(minLength: 44)
            } else {
                assistantAvatar
            }

            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 6) {
                if isEditing {
                    editor
                } else {
                    content
                        .padding(.horizontal, ChatTheme.messagePaddingH)
                        .padding(.vertical, ChatTheme.messagePaddingV)
                        .foregroundStyle(message.isUser ? Color.white : Color.white.opacity(0.92))
                        .background(bubbleBackground)
                        .clipShape(bubbleShape)
                }

                if !message.isPartial && !isEditing {
                    actions
                }
            }

            if !message.isUser {
                Spacer(minLength: 44)
            }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 3)
        .opacity(appeared ? 1 : 0)
        .offset(y: appeared ? 0 : (reduceMotion ? 0 : 8))
        .onAppear {
            withAnimation(reduceMotion ? nil : .spring(response: 0.35, dampingFraction: 0.82)) {
                appeared = true
            }
        }
    }

    private var assistantAvatar: some View {
        ZStack {
            Circle()
                .fill(
                    LinearGradient(
                        colors: [
                            ChatTheme.auroraCyan.opacity(0.25),
                            ChatTheme.electricBlue.opacity(0.12)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )

            Circle()
                .strokeBorder(
                    LinearGradient(
                        colors: [
                            ChatTheme.auroraCyan.opacity(0.45),
                            ChatTheme.electricBlue.opacity(0.15)
                        ],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    ),
                    lineWidth: 0.7
                )

            Image(systemName: "cross.case.fill")
                .font(.system(size: 13, weight: .semibold))
                .foregroundStyle(ChatTheme.auroraCyan)
        }
        .frame(width: ChatTheme.avatarSize, height: ChatTheme.avatarSize)
        .shadow(color: ChatTheme.electricBlue.opacity(0.25), radius: 6, y: 2)
        .padding(.top, 4)
        .accessibilityHidden(true)
    }

    @ViewBuilder
    private var content: some View {
        if !message.isUser && message.isPartial && message.text.isEmpty {
            TypingIndicator()
        } else if message.isUser || message.isPartial {
            Text(message.text.isEmpty ? "…" : message.text)
                .font(ChatTheme.messageBody)
                .lineSpacing(3)
                .textSelection(.enabled)
        } else {
            Markdown(message.text)
                .markdownTheme(.basic)
                .markdownImageProvider(.asset)
                .markdownInlineImageProvider(.asset)
                .environment(\.openURL, OpenURLAction { _ in .discarded })
                .font(ChatTheme.messageBody)
                .lineSpacing(3.5)
                .textSelection(.enabled)
        }
    }

    private var editor: some View {
        VStack(alignment: .trailing, spacing: 8) {
            TextField("Edit message", text: $editText, axis: .vertical)
                .font(ChatTheme.messageBody)
                .foregroundStyle(.white)
                .lineLimit(1...6)
                .padding(12)
                .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .strokeBorder(accentColor.opacity(0.5), lineWidth: 0.8)
                )
                .focused($isEditFocused)

            HStack(spacing: 8) {
                Button("Cancel") {
                    withAnimation(.snappy(duration: 0.2)) {
                        isEditing = false
                    }
                }
                .foregroundStyle(.white.opacity(0.6))
                .padding(.horizontal, 10)
                .padding(.vertical, 6)

                Button {
                    let edited = editText.trimmingCharacters(in: .whitespacesAndNewlines)
                    withAnimation(.snappy(duration: 0.2)) {
                        isEditing = false
                    }
                    if !edited.isEmpty, edited != message.text {
                        onEdit?(edited)
                    }
                } label: {
                    Text("Save & Send")
                        .fontWeight(.semibold)
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 6)
                        .background(ChatTheme.accentGradient(load: 0), in: Capsule())
                        .shadow(color: accentColor.opacity(0.3), radius: 6, y: 2)
                }
            }
            .font(.caption)
        }
        .onAppear { isEditFocused = true }
    }

    private var actions: some View {
        HStack(spacing: 10) {
            if message.isUser {
                actionButton("pencil", label: "Modifier le message") {
                    editText = message.text
                    withAnimation(.snappy(duration: 0.2)) {
                        isEditing = true
                    }
                }

                Button(action: copyMessage) {
                    Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(isCopied ? ChatTheme.mint : .white.opacity(0.42))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Copier la question")

                ShareLink(item: message.text) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.white.opacity(0.42))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Partager la question")
            } else {
                Button(action: copyMessage) {
                    HStack(spacing: 4) {
                        Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                            .font(.system(size: 11, weight: .medium))
                        if isCopied {
                            Text("Copié")
                                .font(.system(size: 11, weight: .medium, design: .rounded))
                        }
                    }
                    .foregroundStyle(isCopied ? ChatTheme.mint : .white.opacity(0.42))
                    .padding(.horizontal, isCopied ? 7 : 4)
                    .padding(.vertical, 3)
                    .background(
                        isCopied
                            ? ChatTheme.mint.opacity(0.12)
                            : Color.clear,
                        in: Capsule()
                    )
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Copier la réponse")

                // Text-to-Speech button
                Button {
                    HapticsHelper.playLightImpact()
                    speechManager.speak(messageId: message.id, text: message.text)
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: speechManager.isSpeaking(messageId: message.id) ? "speaker.wave.3.fill" : "speaker.wave.2")
                            .font(.system(size: 11, weight: .medium))
                        if speechManager.isSpeaking(messageId: message.id) {
                            Text("En écoute")
                                .font(.system(size: 11, weight: .medium, design: .rounded))
                        }
                    }
                    .foregroundStyle(speechManager.isSpeaking(messageId: message.id) ? ChatTheme.auroraCyan : .white.opacity(0.42))
                    .padding(.horizontal, speechManager.isSpeaking(messageId: message.id) ? 7 : 4)
                    .padding(.vertical, 3)
                    .background(
                        speechManager.isSpeaking(messageId: message.id)
                            ? ChatTheme.auroraCyan.opacity(0.14)
                            : Color.clear,
                        in: Capsule()
                    )
                }
                .buttonStyle(.plain)
                .accessibilityLabel(speechManager.isSpeaking(messageId: message.id) ? "Arrêter la lecture audio" : "Écouter la réponse")

                ShareLink(item: message.text) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.white.opacity(0.42))
                        .frame(width: 24, height: 24)
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Partager la réponse")

                actionButton("arrow.clockwise", label: "Régénérer la réponse") {
                    onRegenerate?()
                }
            }

            Text(Self.relativeFormatter.localizedString(for: message.createdAt, relativeTo: Date()))
                .font(ChatTheme.messageCaption)
                .foregroundStyle(.white.opacity(0.32))
        }
        .padding(.horizontal, 4)
    }

    private func actionButton(_ systemName: String, label: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: systemName)
                .font(.system(size: 11, weight: .medium))
                .foregroundStyle(.white.opacity(0.42))
                .frame(width: 24, height: 24)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(label)
    }

    private var bubbleShape: UnevenRoundedRectangle {
        UnevenRoundedRectangle(
            topLeadingRadius: message.isUser ? ChatTheme.bubbleRadius : ChatTheme.bubbleRadiusSmall,
            bottomLeadingRadius: ChatTheme.bubbleRadius,
            bottomTrailingRadius: ChatTheme.bubbleRadius,
            topTrailingRadius: message.isUser ? ChatTheme.bubbleRadiusSmall : ChatTheme.bubbleRadius
        )
    }

    private var bubbleBackground: some View {
        Group {
            if message.isUser {
                bubbleShape
                    .fill(ChatTheme.userBubbleGradient)
                    .overlay(
                        bubbleShape.strokeBorder(
                            LinearGradient(
                                colors: [
                                    Color.white.opacity(0.32),
                                    Color.white.opacity(0.08)
                                ],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 0.7
                        )
                    )
                    .shadow(
                        color: ChatTheme.royalBlue.opacity(0.35),
                        radius: 12,
                        y: 4
                    )
            } else {
                bubbleShape
                    .fill(.ultraThinMaterial)
                    .overlay(
                        bubbleShape.fill(Color.white.opacity(0.035))
                    )
                    .overlay(
                        bubbleShape.strokeBorder(
                            ChatTheme.glassBorderGradient,
                            lineWidth: 0.6
                        )
                    )
                    .shadow(
                        color: Color.black.opacity(0.22),
                        radius: 10,
                        y: 4
                    )
            }
        }
    }

    private func copyMessage() {
        #if canImport(UIKit)
        UIPasteboard.general.string = message.text
        HapticsHelper.playLightImpact()
        withAnimation(.spring(response: 0.25, dampingFraction: 0.7)) {
            isCopied = true
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.8) {
            withAnimation(.easeInOut(duration: 0.2)) {
                isCopied = false
            }
        }
        #endif
    }

    private static let relativeFormatter: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .abbreviated
        return formatter
    }()
}

#Preview {
    ZStack {
        ChatTheme.backgroundBase.ignoresSafeArea()
        VStack(spacing: 12) {
            MessageBubble(
                message: Message(text: "I have chest pain since this morning.", isUser: true),
                accentColor: ChatTheme.electricBlue
            )
            MessageBubble(
                message: Message(text: "Could you describe the pain and whether it radiates to your arm or jaw?", isUser: false),
                accentColor: ChatTheme.electricBlue
            )
        }
    }
}
