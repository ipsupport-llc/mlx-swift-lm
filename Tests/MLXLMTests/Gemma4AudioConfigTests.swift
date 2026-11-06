// Copyright © 2026 Apple Inc.

import Foundation
import Testing

@testable import MLXVLM

/// The Gemma 4 processor config carries the audio token ids and the
/// `feature_extractor` block; both must survive an encode/decode round trip.
struct Gemma4AudioConfigTests {

    private static let json = """
        {
          "processor_class": "Gemma4Processor",
          "audio_token_id": 258881,
          "boa_token_id": 256000,
          "eoa_token_id": 258883,
          "feature_extractor": {
            "feature_extractor_type": "Gemma4AudioFeatureExtractor",
            "sampling_rate": 16000,
            "feature_size": 128,
            "frame_length_ms": 20.0,
            "hop_length_ms": 10.0,
            "input_scale_factor": 1.0,
            "mel_floor": 0.001
          }
        }
        """

    @Test("Audio keys decode into a feature extractor")
    func decodes() throws {
        let c = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self, from: Data(Self.json.utf8))
        #expect(c.audioTokenId == 258_881)
        #expect(c.boaTokenId == 256_000)
        #expect(c.eoaTokenId == 258_883)
        let fe = try #require(c.audioFeatures)
        #expect(fe.sampleRate == 16_000)
        #expect(fe.featureSize == 128)
        #expect(fe.frameLength == 320)
        #expect(fe.hopLength == 160)
    }

    @Test("Encode/decode keeps the audio keys and the feature extractor")
    func roundTrip() throws {
        let c = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self, from: Data(Self.json.utf8))
        let again = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self, from: JSONEncoder().encode(c))
        #expect(again.audioTokenId == c.audioTokenId)
        #expect(again.boaTokenId == c.boaTokenId)
        #expect(again.eoaTokenId == c.eoaTokenId)
        let fe = try #require(again.audioFeatures)
        #expect(fe.sampleRate == 16_000)
        #expect(fe.hopLength == 160)
    }

    @Test("A config without feature_extractor has no audio features")
    func imageOnly() throws {
        let c = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self,
            from: Data(#"{"processor_class": "Gemma4Processor"}"#.utf8))
        #expect(c.audioFeatures == nil)
        let again = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self, from: JSONEncoder().encode(c))
        #expect(again.audioFeatures == nil)
    }
}
