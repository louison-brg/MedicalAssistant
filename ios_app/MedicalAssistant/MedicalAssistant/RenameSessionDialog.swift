import SwiftUI

struct RenameSessionDialog: View {
    @Binding var isPresented: Bool
    @State var title: String
    let onSave: (String) -> Void

    @FocusState private var isFieldFocused: Bool

    var body: some View {
        ZStack {
            Color.black.opacity(0.55)
                .ignoresSafeArea()
                .onTapGesture {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        isPresented = false
                    }
                }

            VStack(spacing: 18) {
                HStack(spacing: 10) {
                    ZStack {
                        Circle()
                            .fill(ChatTheme.auroraCyan.opacity(0.18))
                            .frame(width: 38, height: 38)
                        Image(systemName: "pencil")
                            .font(.system(size: 16, weight: .bold))
                            .foregroundStyle(ChatTheme.auroraCyan)
                    }

                    VStack(alignment: .leading, spacing: 2) {
                        Text("Renommer la discussion")
                            .font(.system(.headline, design: .rounded, weight: .bold))
                            .foregroundStyle(.white)

                        Text("Donnez un titre clair à cette consultation")
                            .font(.system(.caption, design: .rounded))
                            .foregroundStyle(.white.opacity(0.5))
                    }

                    Spacer()
                }

                TextField("Titre de la consultation", text: $title)
                    .font(ChatTheme.inputFont)
                    .foregroundStyle(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .fill(Color.white.opacity(0.06))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .strokeBorder(
                                isFieldFocused
                                    ? ChatTheme.auroraCyan.opacity(0.6)
                                    : Color.white.opacity(0.15),
                                lineWidth: 1
                            )
                    )
                    .focused($isFieldFocused)
                    .submitLabel(.done)
                    .onSubmit(save)

                HStack(spacing: 12) {
                    Button("Annuler") {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            isPresented = false
                        }
                    }
                    .font(.system(.subheadline, design: .rounded, weight: .medium))
                    .foregroundStyle(.white.opacity(0.7))
                    .padding(.horizontal, 18)
                    .padding(.vertical, 10)
                    .background(Color.white.opacity(0.08), in: Capsule())
                    .buttonStyle(.plain)

                    Spacer()

                    Button {
                        save()
                    } label: {
                        HStack(spacing: 6) {
                            Image(systemName: "checkmark")
                                .font(.system(size: 12, weight: .bold))
                            Text("Enregistrer")
                                .font(.system(.subheadline, design: .rounded, weight: .semibold))
                        }
                        .foregroundStyle(.white)
                        .padding(.horizontal, 20)
                        .padding(.vertical, 10)
                        .background(
                            canSave
                                ? AnyShapeStyle(ChatTheme.userBubbleGradient)
                                : AnyShapeStyle(Color.white.opacity(0.12)),
                            in: Capsule()
                        )
                        .shadow(
                            color: canSave ? ChatTheme.royalBlue.opacity(0.4) : .clear,
                            radius: 8,
                            y: 2
                        )
                    }
                    .buttonStyle(.plain)
                    .disabled(!canSave)
                }
            }
            .padding(22)
            .background(
                RoundedRectangle(cornerRadius: 24, style: .continuous)
                    .fill(.ultraThinMaterial)
                    .overlay(
                        RoundedRectangle(cornerRadius: 24, style: .continuous)
                            .strokeBorder(
                                LinearGradient(
                                    colors: [
                                        ChatTheme.auroraCyan.opacity(0.4),
                                        Color.white.opacity(0.15),
                                        Color.clear
                                    ],
                                    startPoint: .topLeading,
                                    endPoint: .bottomTrailing
                                ),
                                lineWidth: 1
                            )
                    )
            )
            .shadow(color: Color.black.opacity(0.5), radius: 30, y: 12)
            .padding(.horizontal, 24)
            .transition(.scale(scale: 0.92).combined(with: .opacity))
        }
        .onAppear {
            isFieldFocused = true
        }
    }

    private var canSave: Bool {
        !title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func save() {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        HapticsHelper.playLightImpact()
        onSave(trimmed)
        withAnimation(.easeInOut(duration: 0.2)) {
            isPresented = false
        }
    }
}
