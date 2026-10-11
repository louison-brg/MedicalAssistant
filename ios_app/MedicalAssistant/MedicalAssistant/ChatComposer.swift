import SwiftUI

/// Composer owns its draft locally. Keystrokes therefore invalidate only this
/// small view instead of the full message list, Markdown tree, and background.
struct ChatComposer: View {
    let isGenerating: Bool
    let onSend: (String) -> Void
    let onStop: () -> Void
    @FocusState.Binding var isFocused: Bool

    @State private var draft = ""

    private var canSend: Bool {
        !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(spacing: 6) {
            // Character count hint for long questions
            if draft.count > 300 {
                HStack {
                    Spacer()
                    Text("\(draft.count) caractères")
                        .font(.system(size: 10, design: .rounded))
                        .foregroundStyle(draft.count > 1200 ? ChatTheme.amber : .white.opacity(0.4))
                        .padding(.trailing, 24)
                }
                .transition(.opacity)
            }

            HStack(alignment: .bottom, spacing: 8) {
                // Text input
                TextField("Posez votre question (symptômes, ordonnance…)", text: $draft, axis: .vertical)
                    .font(ChatTheme.inputFont)
                    .foregroundStyle(.white.opacity(0.96))
                    .lineLimit(1...5)
                    .focused($isFocused)
                    .submitLabel(.send)
                    .onSubmit(sendDraft)
                    .tint(ChatTheme.auroraCyan)
                    .padding(.leading, 18)
                    .padding(.vertical, 12)

                // Quick dismiss keyboard button when typing
                if isFocused {
                    Button {
                        isFocused = false
                    } label: {
                        Image(systemName: "keyboard.chevron.compact.down")
                            .font(.system(size: 15, weight: .semibold))
                            .foregroundStyle(.white.opacity(0.45))
                    }
                    .buttonStyle(.plain)
                    .padding(.bottom, 12)
                    .transition(.opacity.combined(with: .scale))
                    .accessibilityLabel("Masquer le clavier")
                }

                // Clear button
                if !draft.isEmpty && !isGenerating {
                    Button {
                        draft = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.system(size: 15))
                            .foregroundStyle(.white.opacity(0.35))
                    }
                    .buttonStyle(.plain)
                    .padding(.bottom, 12)
                    .transition(.opacity.combined(with: .scale))
                }

                // Send / Stop button
                Button(action: primaryAction) {
                    ZStack {
                        if isGenerating {
                            Circle()
                                .fill(
                                    LinearGradient(
                                        colors: [Color.red.opacity(0.9), Color(red: 0.8, green: 0.1, blue: 0.2)],
                                        startPoint: .topLeading,
                                        endPoint: .bottomTrailing
                                    )
                                )
                                .frame(width: 36, height: 36)
                                .shadow(color: Color.red.opacity(0.4), radius: 8, y: 2)

                            RoundedRectangle(cornerRadius: 3, style: .continuous)
                                .fill(.white)
                                .frame(width: 11, height: 11)
                        } else {
                            Circle()
                                .fill(
                                    canSend
                                        ? AnyShapeStyle(ChatTheme.userBubbleGradient)
                                        : AnyShapeStyle(Color.white.opacity(0.08))
                                )
                                .frame(width: 36, height: 36)
                                .shadow(
                                    color: canSend ? ChatTheme.royalBlue.opacity(0.4) : Color.clear,
                                    radius: 8,
                                    y: 3
                                )

                            Image(systemName: "arrow.up")
                                .font(.system(size: 14, weight: .bold))
                                .foregroundStyle(canSend ? .white : .white.opacity(0.35))
                        }
                    }
                }
                .buttonStyle(.plain)
                .disabled(!isGenerating && !canSend)
                .scaleEffect(canSend || isGenerating ? 1.0 : 0.96)
                .animation(.spring(response: 0.25, dampingFraction: 0.7), value: canSend)
                .animation(.spring(response: 0.25, dampingFraction: 0.7), value: isGenerating)
                .padding(.trailing, 8)
                .padding(.bottom, 6)
                .accessibilityLabel(isGenerating ? "Arrêter la génération" : "Envoyer le message")
            }
            .background(
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 26, style: .continuous)
                            .fill(Color.white.opacity(0.035))
                    )
            )
            .overlay(
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .strokeBorder(
                        LinearGradient(
                            colors: [
                                isFocused ? ChatTheme.auroraCyan.opacity(0.6) : Color.white.opacity(0.20),
                                isFocused ? ChatTheme.electricBlue.opacity(0.3) : Color.white.opacity(0.06)
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        ),
                        lineWidth: isFocused ? 1.0 : 0.6
                    )
            )
            .shadow(
                color: isFocused
                    ? ChatTheme.electricBlue.opacity(0.22)
                    : Color.black.opacity(0.28),
                radius: isFocused ? 18 : 14,
                y: isFocused ? 6 : 4
            )
            .animation(.easeInOut(duration: 0.2), value: isFocused)
            .padding(.horizontal, 14)

            // Security & offline guarantee footer (automatically hidden when keyboard is open to preserve vertical space)
            if !isFocused {
                HStack(spacing: 6) {
                    Image(systemName: "lock.shield.fill")
                        .font(.system(size: 9, weight: .semibold))
                        .foregroundStyle(ChatTheme.emerald)

                    Text("100% Hors-ligne · Confidentialité totale · Non diagnostique")
                        .font(.system(size: 10, weight: .medium, design: .rounded))
                        .tracking(0.3)
                        .foregroundStyle(.white.opacity(0.38))
                }
                .padding(.bottom, 4)
                .transition(.opacity.combined(with: .move(edge: .bottom)))
            }
        }
        .padding(.top, 4)
        .animation(.snappy(duration: 0.25), value: isFocused)
    }

    private func primaryAction() {
        if isGenerating {
            HapticsHelper.playLightImpact()
            onStop()
        } else {
            sendDraft()
        }
    }

    private func sendDraft() {
        guard !isGenerating else { return }
        let text = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        HapticsHelper.playLightImpact()
        draft = ""
        onSend(text)
    }
}
