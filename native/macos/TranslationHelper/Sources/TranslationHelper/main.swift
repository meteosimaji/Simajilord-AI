import Foundation
import NaturalLanguage
import Translation

private typealias JSONObject = [String: Any]

@main
struct TranslationHelper {
    static func main() async {
        while let line = readLine() {
            guard !line.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue
            }
            let response = await handle(line)
            write(response)
        }
    }

    private static func handle(_ line: String) async -> JSONObject {
        do {
            guard
                let data = line.data(using: .utf8),
                let request = try JSONSerialization.jsonObject(with: data) as? JSONObject,
                let operation = request["operation"] as? String
            else {
                return failure("translation.request_invalid", "Expected one JSON object.")
            }
            switch operation {
            case "detect":
                return try detect(request)
            case "languages":
                return await languages(request)
            case "translate":
                if #available(macOS 26.0, *) {
                    return await translate(request)
                }
                return failure(
                    "translation.helper_unavailable",
                    "Direct on-device translation requires macOS 26 or newer."
                )
            case "translate_batch":
                if #available(macOS 26.0, *) {
                    return await translateBatch(request)
                }
                return failure(
                    "translation.helper_unavailable",
                    "Direct on-device translation requires macOS 26 or newer."
                )
            default:
                return failure("translation.request_invalid", "Unknown operation.")
            }
        } catch {
            return failure("translation.helper_failed", String(describing: error))
        }
    }

    private static func detect(_ request: JSONObject) throws -> JSONObject {
        guard let text = request["text"] as? String, !text.isEmpty else {
            return failure("translation.text_required", "Text is required.")
        }
        let recognizer = NLLanguageRecognizer()
        recognizer.processString(text)
        guard let dominant = recognizer.dominantLanguage else {
            return failure(
                "translation.language_not_detected",
                "The source language could not be identified."
            )
        }
        let hypotheses = recognizer.languageHypotheses(withMaximum: 5)
            .map { language, confidence in
                [
                    "code": language.rawValue,
                    "confidence": confidence,
                ] as JSONObject
            }
            .sorted {
                ($0["confidence"] as? Double ?? 0) > ($1["confidence"] as? Double ?? 0)
            }
        let confidence = hypotheses.first {
            ($0["code"] as? String) == dominant.rawValue
        }?["confidence"] as? Double ?? 0
        return [
            "ok": true,
            "language": dominant.rawValue,
            "confidence": confidence,
            "hypotheses": hypotheses,
        ]
    }

    @available(macOS 15.0, *)
    private static func languages(_ request: JSONObject) async -> JSONObject {
        let availability = LanguageAvailability()
        let sourceIdentifier = request["source_language"] as? String
        let source = sourceIdentifier.map(Locale.Language.init(identifier:))
        let englishLocale = Locale(identifier: "en")
        var values: [JSONObject] = []
        for language in await availability.supportedLanguages {
            let code = language.minimalIdentifier
            let nativeLocale = Locale(identifier: code)
            let status: String
            if let source, source.minimalIdentifier != code {
                status = availabilityName(await availability.status(from: source, to: language))
            } else if source?.minimalIdentifier == code {
                status = "same_language"
            } else {
                status = "supported"
            }
            values.append([
                "code": code,
                "english_name": englishLocale.localizedString(forLanguageCode: code) ?? code,
                "native_name": nativeLocale.localizedString(forLanguageCode: code) ?? code,
                "availability": status,
            ])
        }
        values.sort {
            ($0["english_name"] as? String ?? "") < ($1["english_name"] as? String ?? "")
        }
        return ["ok": true, "languages": values]
    }

    @available(macOS 26.0, *)
    private static func translate(_ request: JSONObject) async -> JSONObject {
        guard
            let text = request["text"] as? String,
            !text.isEmpty,
            let targetIdentifier = request["target_language"] as? String,
            !targetIdentifier.isEmpty
        else {
            return failure("translation.request_invalid", "Text and target language are required.")
        }
        let sourceIdentifier: String
        if let supplied = request["source_language"] as? String, !supplied.isEmpty {
            sourceIdentifier = supplied
        } else if let detected = NLLanguageRecognizer.dominantLanguage(for: text)?.rawValue {
            sourceIdentifier = detected
        } else {
            return failure(
                "translation.language_not_detected",
                "The source language could not be identified."
            )
        }
        let source = Locale.Language(identifier: sourceIdentifier)
        let target = Locale.Language(identifier: targetIdentifier)
        let availability = LanguageAvailability()
        let status = await availability.status(from: source, to: target)
        switch status {
        case .unsupported:
            return failure(
                "translation.language_pair_unsupported",
                "The requested language pair is not supported by Apple Translation."
            )
        case .supported:
            return failure(
                "translation.language_pair_not_installed",
                "Install this language pair in macOS before translating."
            )
        case .installed:
            break
        @unknown default:
            return failure(
                "translation.language_pair_unsupported",
                "The language-pair status is unknown."
            )
        }
        do {
            let session = TranslationSession(installedSource: source, target: target)
            let response = try await session.translate(text)
            return [
                "ok": true,
                "source_language": response.sourceLanguage.minimalIdentifier,
                "target_language": response.targetLanguage.minimalIdentifier,
                "translated_text": response.targetText,
            ]
        } catch {
            return failure("translation.failed", String(describing: error))
        }
    }

    @available(macOS 26.0, *)
    private static func translateBatch(_ request: JSONObject) async -> JSONObject {
        guard
            let values = request["segments"] as? [JSONObject],
            !values.isEmpty,
            let targetIdentifier = request["target_language"] as? String,
            !targetIdentifier.isEmpty
        else {
            return failure(
                "translation.request_invalid",
                "Segments and target language are required."
            )
        }
        var requests: [TranslationSession.Request] = []
        var sourceTextByIdentifier: [String: String] = [:]
        for value in values {
            guard
                let identifier = value["identifier"] as? String,
                !identifier.isEmpty,
                let text = value["text"] as? String,
                !text.isEmpty,
                sourceTextByIdentifier[identifier] == nil
            else {
                return failure(
                    "translation.request_invalid",
                    "Every segment needs a unique identifier and non-empty text."
                )
            }
            sourceTextByIdentifier[identifier] = text
            requests.append(
                TranslationSession.Request(
                    sourceText: text,
                    clientIdentifier: identifier
                )
            )
        }
        let sourceIdentifier: String
        if let supplied = request["source_language"] as? String, !supplied.isEmpty {
            sourceIdentifier = supplied
        } else {
            let detectionText = values.compactMap { $0["text"] as? String }.joined(
                separator: "\n"
            )
            guard let detected = NLLanguageRecognizer.dominantLanguage(
                for: detectionText
            )?.rawValue else {
                return failure(
                    "translation.language_not_detected",
                    "The source language could not be identified."
                )
            }
            sourceIdentifier = detected
        }
        let source = Locale.Language(identifier: sourceIdentifier)
        let target = Locale.Language(identifier: targetIdentifier)
        let availability = LanguageAvailability()
        let status = await availability.status(from: source, to: target)
        switch status {
        case .unsupported:
            return failure(
                "translation.language_pair_unsupported",
                "The requested language pair is not supported by Apple Translation."
            )
        case .supported:
            return failure(
                "translation.language_pair_not_installed",
                "Install this language pair in macOS before translating."
            )
        case .installed:
            break
        @unknown default:
            return failure(
                "translation.language_pair_unsupported",
                "The language-pair status is unknown."
            )
        }
        do {
            let session = TranslationSession(installedSource: source, target: target)
            let responses = try await session.translations(from: requests)
            let byIdentifier = Dictionary(
                uniqueKeysWithValues: responses.compactMap { response in
                    response.clientIdentifier.map { ($0, response) }
                }
            )
            var translated: [JSONObject] = []
            for request in requests {
                guard
                    let identifier = request.clientIdentifier,
                    let sourceText = sourceTextByIdentifier[identifier],
                    let response = byIdentifier[identifier]
                else {
                    return failure(
                        "translation.failed",
                        "Apple Translation returned an incomplete batch."
                    )
                }
                translated.append([
                    "identifier": identifier,
                    "source_text": sourceText,
                    "translated_text": response.targetText,
                ])
            }
            return [
                "ok": true,
                "source_language": source.minimalIdentifier,
                "target_language": target.minimalIdentifier,
                "segments": translated,
            ]
        } catch {
            return failure("translation.failed", String(describing: error))
        }
    }

    private static func availabilityName(_ status: LanguageAvailability.Status) -> String {
        switch status {
        case .installed:
            return "installed"
        case .supported:
            return "supported"
        case .unsupported:
            return "unsupported"
        @unknown default:
            return "unknown"
        }
    }

    private static func failure(_ code: String, _ detail: String) -> JSONObject {
        [
            "ok": false,
            "error": code,
            "detail": detail,
        ]
    }

    private static func write(_ object: JSONObject) {
        guard
            JSONSerialization.isValidJSONObject(object),
            let data = try? JSONSerialization.data(withJSONObject: object),
            let line = String(data: data, encoding: .utf8)
        else {
            print(
                "{\"ok\":false,\"error\":\"translation.helper_failed\","
                    + "\"detail\":\"Could not encode a JSON response.\"}"
            )
            return
        }
        print(line)
        fflush(stdout)
    }
}
