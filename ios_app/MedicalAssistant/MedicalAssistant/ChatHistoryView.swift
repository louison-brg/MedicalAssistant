import SwiftUI

struct ChatHistoryView: View {
    @ObservedObject var viewModel: ChatViewModel
    @Environment(\.dismiss) var dismiss

    var body: some View {
        NavigationView {
            List {
                if viewModel.sessions.isEmpty {
                    Text("Aucune discussion sauvegardée.")
                        .foregroundColor(.gray)
                        .italic()
                } else {
                    ForEach(viewModel.sessions) { session in
                        Button(action: {
                            viewModel.switchSession(to: session.id)
                            dismiss()
                        }) {
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(session.title)
                                        .font(.headline)
                                        .foregroundColor(viewModel.activeSessionId == session.id ? ChatTheme.accent : .primary)
                                    
                                    HStack {
                                        Text("\(session.messages.count) messages")
                                        Spacer()
                                        Text(session.updatedAt, style: .date)
                                    }
                                    .font(.caption)
                                    .foregroundColor(.gray)
                                }
                                
                                Spacer()
                                
                                if viewModel.activeSessionId == session.id {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundColor(ChatTheme.accent)
                                }
                            }
                            .padding(.vertical, 4)
                        }
                    }
                    .onDelete(perform: deleteSession)
                }
            }
            .navigationTitle("Historique")
            .navigationBarItems(
                leading: Button("Fermer") { dismiss() },
                trailing: EditButton()
            )
        }
    }
    
    private func deleteSession(offsets: IndexSet) {
        for index in offsets {
            let session = viewModel.sessions[index]
            viewModel.deleteSession(id: session.id)
        }
    }
}
