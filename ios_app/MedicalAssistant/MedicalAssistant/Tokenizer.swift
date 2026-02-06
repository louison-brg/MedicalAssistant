import Foundation

/// Tokenizer BPE compatible (Phi-3-like)
/// - Charge `vocab.json` (token -> id) et `merges.txt` (paires) depuis le bundle.
/// - Implémente l'encodage BPE et la reconstruction de texte.
/// - Gère les tokens spéciaux (BOS/EOS/PAD) via détection dans le vocab ou fallback.
final class Tokenizer {
    // MARK: - Storage
    private var tokenToId: [String: Int] = [:]
    private var idToToken: [Int: String] = [:]
    private var specialTokens: Set<String> = []

    // BPE merges: priorité par rang (plus petit = fusion plus prioritaire)
    private var bpeRanks: [String: Int] = [:] // key "A B" -> rank

    // Spéciaux (tentative de lecture depuis vocab; sinon fallback)
    private(set) var bosTokenId: Int = 1
    private(set) var eosTokenId: Int = 32000
    private(set) var padTokenId: Int = 32000

    // Cache BPE pour accélérer
    private var bpeCache: [String: [String]] = [:]

    private func loadTokenizerJSON(from url: URL) -> Bool {
        do {
            let data = try Data(contentsOf: url)
            guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return false
            }
            // model.vocab
            guard let model = root["model"] as? [String: Any],
                  let vocab = model["vocab"] as? [String: Int] else {
                return false
            }
            tokenToId = vocab
            idToToken = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })
            // merges
            if let merges = model["merges"] as? [String] {
                bpeRanks.removeAll(keepingCapacity: true)
                var rank = 0
                for line in merges {
                    let parts = line.split(separator: " ")
                    if parts.count == 2 {
                        bpeRanks["\(parts[0]) \(parts[1])"] = rank
                        rank += 1
                    }
                }
            }
            // added tokens (spéciaux compris)
            if let added = root["added_tokens"] as? [[String: Any]] {
                for entry in added {
                    guard let content = entry["content"] as? String,
                          let id = entry["id"] as? Int else { continue }
                    tokenToId[content] = id
                    idToToken[id] = content
                    if let isSpecial = entry["special"] as? Bool, isSpecial {
                        specialTokens.insert(content)
                    }
                }
            }
            // special tokens map
            if let stm = root["special_tokens_map"] as? [String: Any] {
                if let bosEntry = stm["bos_token"] as? [String: Any] {
                    if let bos = bosEntry["id"] as? Int { bosTokenId = bos }
                    if let content = bosEntry["content"] as? String { specialTokens.insert(content); tokenToId[content] = bosTokenId; idToToken[bosTokenId] = content }
                }
                if let eosEntry = stm["eos_token"] as? [String: Any] {
                    if let eos = eosEntry["id"] as? Int { eosTokenId = eos }
                    if let content = eosEntry["content"] as? String { specialTokens.insert(content); tokenToId[content] = eosTokenId; idToToken[eosTokenId] = content }
                }
                if let padEntry = stm["pad_token"] as? [String: Any] {
                    if let pad = padEntry["id"] as? Int { padTokenId = pad }
                    if let content = padEntry["content"] as? String { specialTokens.insert(content); tokenToId[content] = padTokenId; idToToken[padTokenId] = content }
                }
            }
            // Fallback via tokens nommés
            if let bos = tokenToId["<|bos|>"] ?? tokenToId["<bos>"] ?? tokenToId["<s>"] { bosTokenId = bos }
            if let eos = tokenToId["<|eos|>"] ?? tokenToId["<eos>"] ?? tokenToId["</s>"] ?? tokenToId["<|endoftext|>"] ?? tokenToId["<end>"] {
                eosTokenId = eos
            }
            if let pad = tokenToId["<|pad|>"] ?? tokenToId["<pad>"] ?? tokenToId["<|endoftext|>"] {
                padTokenId = pad
            }

            print("✅ Tokenizer(HF) chargé: vocab=\(tokenToId.count), merges=\(bpeRanks.count). BOS=\(bosTokenId), EOS=\(eosTokenId), PAD=\(padTokenId)")
            return true
        } catch {
            print("❌ Erreur lecture tokenizer.json:", error.localizedDescription)
            return false
        }
    }

    init() {
        loadResources()
    }

    // MARK: - Loading
    private func loadResources() {
        tokenToId.removeAll(keepingCapacity: false)
        idToToken.removeAll(keepingCapacity: false)
        specialTokens.removeAll(keepingCapacity: false)
        bpeRanks.removeAll(keepingCapacity: false)

        // 1) Bundle tokenizer.json
        if let jsonURL = Bundle.main.url(forResource: "tokenizer", withExtension: "json"), loadTokenizerJSON(from: jsonURL) {
            return
        }
        // 2) Documents/tokenizer.json (au cas où le fichier est copié là)
        if let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
            let jsonURL = docs.appendingPathComponent("tokenizer.json")
            if FileManager.default.fileExists(atPath: jsonURL.path), loadTokenizerJSON(from: jsonURL) {
                return
            }
        }
        // 3) Fallback legacy: vocab.json + merges.txt depuis le bundle
        guard
            let vocabURL = Bundle.main.url(forResource: "vocab", withExtension: "json"),
            let mergesURL = Bundle.main.url(forResource: "merges", withExtension: "txt")
        else {
            print("❌ Impossible de trouver tokenizer.json ni vocab.json/merges.txt dans le bundle.")
            return
        }
        do {
            let vocabData = try Data(contentsOf: vocabURL)
            if let dict = try JSONSerialization.jsonObject(with: vocabData) as? [String: Int] {
                tokenToId = dict
                idToToken = Dictionary(uniqueKeysWithValues: dict.map { ($1, $0) })
            } else {
                print("❌ vocab.json n'est pas un dictionnaire [String:Int].")
            }
            let mergesText = try String(contentsOf: mergesURL)
            var rank = 0
            mergesText.components(separatedBy: .newlines).forEach { line in
                let line = line.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !line.isEmpty, !line.hasPrefix("#") else { return }
                let parts = line.split(separator: " ")
                guard parts.count == 2 else { return }
                bpeRanks["\(parts[0]) \(parts[1])"] = rank
                rank += 1
            }
            if let bos = tokenToId["<bos>"] ?? tokenToId["<s>"] {
                bosTokenId = bos
                specialTokens.formUnion(["<bos>", "<s>"])
            }
            if let eos = tokenToId["<eos>"] ?? tokenToId["</s>"] ?? tokenToId["<|endoftext|>"] ?? tokenToId["<end>"] {
                eosTokenId = eos
                specialTokens.formUnion(["<eos>", "</s>", "<|endoftext|>", "<end>"])
            }
            if let pad = tokenToId["<pad>"] ?? tokenToId["<|endoftext|>"] {
                padTokenId = pad
                specialTokens.formUnion(["<pad>", "<|endoftext|>"])
            }
            print("✅ Tokenizer(legacy) chargé: vocab=\(tokenToId.count), merges=\(bpeRanks.count). BOS=\(bosTokenId), EOS=\(eosTokenId), PAD=\(padTokenId)")
        } catch {
            print("❌ Erreur lors du chargement du tokenizer :", error.localizedDescription)
        }
    }

    // MARK: - Public API

    /// Encode un texte en IDs, sans ajouter BOS/EOS automatiquement.
    /// Utilisez `prependBOS` / `appendEOS` si nécessaire.
    func encode(_ text: String) -> [Int] {
        guard !tokenToId.isEmpty else { return [] }

        let normalized = normalize(text)
        let symbols = bpeEncode(word: normalized)
        return symbols.flatMap { tokenIds(forSymbol: $0) }
    }

    /// Decode une séquence d'IDs vers du texte.
    func decode(_ tokens: [Int]) -> String {
        guard !idToToken.isEmpty else { return "" }
        var result = ""
        var byteBuffer: [UInt8] = []

        func flushBytes() {
            guard !byteBuffer.isEmpty else { return }
            if let chunk = String(bytes: byteBuffer, encoding: .utf8) {
                result.append(chunk)
            } else {
                byteBuffer.forEach { result.append(Character(UnicodeScalar($0))) }
            }
            byteBuffer.removeAll(keepingCapacity: true)
        }

        for id in tokens {
            guard let token = idToToken[id] else { continue }
            if specialTokens.contains(token) {
                flushBytes()
                continue
            }
            if let byte = byteTokenValue(token) {
                byteBuffer.append(byte)
                continue
            }
            flushBytes()
            result.append(token)
        }
        flushBytes()

        result = result.replacingOccurrences(of: "▁", with: " ")
        if result.hasPrefix(" ") {
            result.removeFirst()
        }
        return result
    }

    func id(for token: String) -> Int? {
        return tokenToId[token]
    }
    
    func isValid(id: Int) -> Bool {
        return idToToken[id] != nil
    }

    func token(for id: Int) -> String? {
        return idToToken[id]
    }

    // Utilitaires pour ajouter des spéciaux
    func prependBOS(_ ids: inout [Int]) { ids.insert(bosTokenId, at: 0) }
    func appendEOS(_ ids: inout [Int]) { ids.append(eosTokenId) }

    // MARK: - BPE Core

    private func bpeEncode(word: String) -> [String] {
        if let cached = bpeCache[word] { return cached }

        // Décomposer le mot en caractères (UTF-8 safe). Pour byte-level, transformer en bytes.
        var symbols = Array(word).map { String($0) }
        if symbols.count <= 1 {
            bpeCache[word] = symbols
            return symbols
        }

        // Construire les paires initiales
        var pairs = getPairs(symbols)
        if pairs.isEmpty {
            bpeCache[word] = symbols
            return symbols
        }

        // Itérer les fusions selon le rang
        while true {
            var minRank = Int.max
            var bestPair: (String, String)? = nil

            for (a, b) in pairs {
                let key = "\(a) \(b)"
                if let r = bpeRanks[key], r < minRank {
                    minRank = r
                    bestPair = (a, b)
                }
            }

            guard let (first, second) = bestPair else { break }

            var i = 0
            var newSymbols: [String] = []
            while i < symbols.count {
                if i < symbols.count - 1 && symbols[i] == first && symbols[i + 1] == second {
                    newSymbols.append(first + second)
                    i += 2
                } else {
                    newSymbols.append(symbols[i])
                    i += 1
                }
            }

            symbols = newSymbols
            if symbols.count == 1 { break }
            pairs = getPairs(symbols)
        }

        bpeCache[word] = symbols
        return symbols
    }

    private func getPairs(_ symbols: [String]) -> [(String, String)] {
        guard symbols.count >= 2 else { return [] }
        var pairs: [(String, String)] = []
        pairs.reserveCapacity(symbols.count - 1)
        for i in 0..<(symbols.count - 1) {
            pairs.append((symbols[i], symbols[i + 1]))
        }
        return pairs
    }

    private func normalize(_ text: String) -> String {
        // Applique le normalizer HF: prepend '▁' puis remplace les espaces.
        let replaced = text.replacingOccurrences(of: " ", with: "▁")
        return "▁" + replaced
    }

    private func tokenIds(forSymbol symbol: String) -> [Int] {
        if let id = tokenToId[symbol] {
            return [id]
        }

        var bytesIds: [Int] = []
        for byte in symbol.utf8 {
            let key = String(format: "<0x%02X>", byte)
            if let bid = tokenToId[key] {
                bytesIds.append(bid)
            }
        }
        if !bytesIds.isEmpty {
            return bytesIds
        }

        if let unk = tokenToId["<unk>"] {
            return [unk]
        }
        return []
    }

    private func byteTokenValue(_ token: String) -> UInt8? {
        guard token.count == 6,
              token.hasPrefix("<0x"),
              token.hasSuffix(">") else { return nil }
        let start = token.index(token.startIndex, offsetBy: 3)
        let end = token.index(before: token.endIndex)
        let hex = String(token[start..<end])
        return UInt8(hex, radix: 16)
    }
}
