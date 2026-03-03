import SwiftUI
import MarkdownUI

struct MessageBubble: View {
    let message: Message
    var onEdit: ((String) -> Void)? = nil
    var onRegenerate: (() -> Void)? = nil
    
    @State private var appeared = false
    @State private var isEditing = false
    @State private var editText = ""
    @FocusState private var isEditFocused: Bool

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            if message.isUser { Spacer(minLength: 52) }

            // AI avatar
            if !message.isUser {
                ZStack {
                    Circle()
                        .fill(ChatTheme.accent.opacity(0.15))
                        .frame(width: ChatTheme.avatarSize, height: ChatTheme.avatarSize)
                    Image(systemName: "stethoscope")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(ChatTheme.accent.opacity(0.8))
                }
                .padding(.top, 4)
            }

            VStack(alignment: message.isUser ? .trailing : .leading, spacing: 5) {
                // Bubble
                if isEditing {
                    VStack(alignment: .trailing, spacing: 6) {
                        TextField("Edit message", text: $editText, axis: .vertical)
                            .font(ChatTheme.messageBody)
                            .foregroundStyle(.black)
                            .padding(8)
                            .background(Color.white)
                            .cornerRadius(8)
                            .focused($isEditFocused)
                        
                        HStack {
                            Button("Cancel") { isEditing = false }
                                .font(.caption).foregroundStyle(.white)
                            Button("Save & Send") {
                                isEditing = false
                                if editText != message.text && !editText.isEmpty {
                                    onEdit?(editText)
                                }
                            }
                            .font(.caption.bold())
                            .padding(.horizontal, 10).padding(.vertical, 4)
                            .background(ChatTheme.accent)
                            .clipShape(Capsule())
                            .foregroundStyle(.white)
                        }
                    }
                    .onAppear { isEditFocused = true }
                } else {
                    Group {
                        if !message.isUser && message.isPartial && message.text.isEmpty {
                            TypingIndicator()
                        } else {
                            Markdown(message.text.isEmpty ? "…" : message.text)
                                .markdownTheme(.basic)
                                .font(ChatTheme.messageBody)
                                .textSelection(.enabled)
                        }
                    }
                    .padding(.horizontal, ChatTheme.messagePaddingH)
                    .padding(.vertical, ChatTheme.messagePaddingV)
                    .background(bubbleBackground)
                    .foregroundStyle(message.isUser ? .white : .white.opacity(0.88))
                    .clipShape(bubbleShape)
                    .shadow(
                        color: message.isUser ? ChatTheme.accent.opacity(0.18) : Color.black.opacity(0.2),
                        radius: message.isUser ? 8 : 4,
                        y: message.isUser ? 3 : 2
                    )
                    .opacity(message.isPartial && !message.text.isEmpty ? 0.8 : 1.0)
                }

                // Actions & Timestamp
                if !message.isPartial && !isEditing {
                    HStack(spacing: 12) {
                        if message.isUser {
                            Button(action: {
                                editText = message.text
                                isEditing = true
                            }) { Image(systemName: "pencil").font(.system(size: 12)) }
                        } else {
                            Button(action: { onRegenerate?() }) {
                                Image(systemName: "arrow.clockwise").font(.system(size: 12))
                            }
                        }
                        
                        Text(relativeTime(message.createdAt))
                            .font(ChatTheme.messageCaption)
                    }
                    .foregroundStyle(.white.opacity(0.35))
                    .padding(.horizontal, 4)
                }
            }

            // User avatar
            if message.isUser {
                ZStack {
                    Circle()
                        .fill(
                            LinearGradient(
                                colors: [Color.white.opacity(0.2), Color.white.opacity(0.08)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            )
                        )
                        .frame(width: ChatTheme.avatarSize, height: ChatTheme.avatarSize)
                    Image(systemName: "person.fill")
                        .font(.system(size: 13, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.7))
                }
                .padding(.top, 4)
            }

            if !message.isUser { Spacer(minLength: 52) }
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 2)
        .opacity(appeared ? 1 : 0)
        .offset(y: appeared ? 0 : 14)
        .onAppear {
            withAnimation(.spring(response: 0.4, dampingFraction: 0.78)) {
                appeared = true
            }
        }
    }

    // MARK: - Bubble Shape

    private var bubbleShape: some Shape {
        UnevenRoundedRectangle(
            topLeadingRadius: message.isUser ? ChatTheme.bubbleRadius : ChatTheme.bubbleRadiusSmall,
            bottomLeadingRadius: ChatTheme.bubbleRadius,
            bottomTrailingRadius: ChatTheme.bubbleRadius,
            topTrailingRadius: message.isUser ? ChatTheme.bubbleRadiusSmall : ChatTheme.bubbleRadius
        )
    }

    // MARK: - Bubble Background

    @ViewBuilder
    private var bubbleBackground: some View {
        if message.isUser {
            ChatTheme.userGradient
        } else {
            Color.white.opacity(0.08)
                .background(.ultraThinMaterial.opacity(0.3))
        }
    }

    // MARK: - Relative time

    private func relativeTime(_ date: Date) -> String {
        let diff = Date().timeIntervalSince(date)
        if diff < 60 { return "now" }
        if diff < 3600 { return "\(Int(diff / 60))m ago" }
        if diff < 86400 { return "\(Int(diff / 3600))h ago" }
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}

#Preview {
    ZStack {
        Color(red: 0.06, green: 0.06, blue: 0.10).ignoresSafeArea()
        VStack(spacing: 12) {
            MessageBubble(message: Message(text: "I have chest pain since this morning", isUser: true))
            MessageBubble(message: Message(text: "Could you describe the pain? Is it sharp or dull? Does it radiate to your arm or jaw?", isUser: false))
            MessageBubble(message: Message(text: "", isUser: false, isPartial: true))
        }
        .padding()
    }
}
