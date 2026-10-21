import { describe, expect, it } from 'vitest'
import { isSameModel, withoutUnsupportedOptions, type ModelCapabilities, type ModelFileCheck } from './model'

const whisper: ModelCapabilities = {
	engine: 'whisper',
	requires_vad: false,
	languages: ['en'],
	language_detection: true,
	streaming: true,
	translation: true,
	timestamps: true,
	text_prompts: true,
}

const options = { lang: 'en', init_prompt: 'Punctuate. Use full sentences.', translate: true, n_threads: 4 }

describe('withoutUnsupportedOptions', () => {
	it('sends everything to a model that takes prompts and translation', () => {
		expect(withoutUnsupportedOptions(options, whisper)).toEqual(options)
	})

	it('drops the prompt and translation for engines that reject them, keeping the rest', () => {
		expect(withoutUnsupportedOptions(options, { ...whisper, engine: 'parakeet', text_prompts: false, translation: false })).toEqual({
			lang: 'en',
			n_threads: 4,
		})
	})

	it('leaves the options alone when no capabilities are known yet', () => {
		expect(withoutUnsupportedOptions(options, null)).toEqual(options)
		expect(withoutUnsupportedOptions(options, undefined)).toBe(options)
	})
})

describe('isSameModel', () => {
	const healthy: ModelFileCheck = { path: '/models/ggml-large-v3-turbo.bin', valid: true, size: 1624555275, reason: null }

	it('reuses a healthy file under the URL name when the catalog publishes no size', () => {
		expect(isSameModel('ggml-large-v3-turbo.bin', healthy, {})).toBe(true)
	})

	it('reuses it when the published size matches and downloads again when it does not', () => {
		expect(isSameModel('ggml-large-v3-turbo.bin', healthy, { size: 1624555275 })).toBe(true)
		expect(isSameModel('ggml-large-v3-turbo.bin', healthy, { size: 1 })).toBe(false)
	})

	it('never reuses a corrupt or unchecked file', () => {
		expect(isSameModel('ggml-large-v3-turbo.bin', { ...healthy, valid: false, reason: 'truncated' }, {})).toBe(false)
		expect(isSameModel('ggml-large-v3-turbo.bin', undefined, {})).toBe(false)
	})

	it('never reuses the generic name, which different repos ship different weights under', () => {
		expect(isSameModel('ggml-model.bin', healthy, {})).toBe(false)
		expect(isSameModel('GGML-MODEL.BIN', healthy, {})).toBe(false)
	})

	it('does not trust a check that failed open when a size is published', () => {
		expect(isSameModel('ggml-large-v3-turbo.bin', { ...healthy, size: 0 }, { size: 1624555275 })).toBe(false)
	})
})
