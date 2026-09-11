"""Independent scalar audit of native autoregressive sampling evidence.

No GPU/model execution or production analysis numerical helpers are used.
FP32 subtraction, scalar libm exp, sequential FP64 normalization/CDF, MT19937,
and CDF boundary conventions are spelled out independently below.
"""

import argparse
import bisect
import datetime
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result

    def reject(value):
        raise ValueError('nonfinite JSON literal: ' + value)

    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=reject)

    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'nonfinite JSON number')
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)

    finite(value)
    return value


def identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 << 20), b''):
            digest.update(block)
    return {'path': str(path), 'size': path.stat().st_size,
            'sha256': digest.hexdigest()}


class Mt19937:
    """Direct 32-bit MT recurrence and tempering; independent of C++ runtime."""
    def __init__(self, seed):
        require(type(seed) is int and 0 <= seed < 2**32, 'invalid MT seed')
        self.state = [seed]
        for index in range(1, 624):
            previous = self.state[-1]
            self.state.append((1812433253 * (previous ^ (previous >> 30)) + index)
                              & 0xffffffff)
        self.index = 624

    def draw(self):
        if self.index == 624:
            for index in range(624):
                mixed = ((self.state[index] & 0x80000000) |
                         (self.state[(index + 1) % 624] & 0x7fffffff))
                self.state[index] = (self.state[(index + 397) % 624] ^
                                     (mixed >> 1) ^ (0x9908b0df if mixed & 1 else 0))
            self.index = 0
        value = self.state[self.index]
        self.index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9d2c5680
        value ^= (value << 15) & 0xefc60000
        value ^= value >> 18
        return value & 0xffffffff

    def cpp_state(self):
        return ' '.join(map(str, [*self.state, self.index]))


def canonical_uniform(low, high):
    require(all(type(word) is int and 0 <= word < 2**32 for word in (low, high)),
            'invalid MT word')
    result = (float(low) + float(high) * 4294967296.0) / 18446744073709551616.0
    return math.nextafter(1.0, 0.0) if result >= 1 else result


def sequential_sum(values):
    # Python >=3.12 sum(float_iterable) is compensated, unlike std::accumulate.
    total = 0.0
    for value in values:
        total += float(value)
    return total


def scalar_distribution(logits, temperature):
    logits = np.asarray(logits)
    require(logits.ndim == 1 and logits.size >= 2 and logits.dtype == np.float32
            and np.isfinite(logits).all(), 'need finite logical FP32 row')
    require(type(temperature) in (int, float) and math.isfinite(temperature)
            and temperature > 0, 'invalid temperature')
    maximum = np.max(logits)
    with np.errstate(over='ignore'):
        difference = np.subtract(logits, maximum, dtype=np.float32)
    weights = [math.exp(float(value) / float(temperature)) for value in difference]
    total = sequential_sum(weights)
    require(math.isfinite(total) and total > 0, 'invalid weight normalizer')
    probabilities = [value / total for value in weights]
    cdf = []
    running = 0.0
    for value in probabilities:
        running += value
        cdf.append(running)
    cdf[-1] = 1.0
    return probabilities, cdf, total


def scalar_sample(logits, temperature, uniform):
    require(math.isfinite(uniform) and 0 <= uniform < 1, 'invalid uniform')
    probabilities, cdf, total = scalar_distribution(logits, temperature)
    chosen = bisect.bisect_left(cdf, uniform)
    require(chosen < len(cdf), 'CDF did not select a token')
    return {'token_id': chosen, 'sampled_probability': probabilities[chosen],
            'cdf_lower': cdf[chosen - 1] if chosen else 0.0,
            'cdf_upper': cdf[chosen], 'unnormalized_weight_sum': total}


def validate_context(event, initial_count, generated_index, all_ids, context_length):
    absolute = initial_count + generated_index
    length = min(absolute, context_length)
    expected = {'index': generated_index, 'step': generated_index,
                'absolute_token_index': absolute, 'context_start': absolute - length,
                'context_length': length, 'output_row': length - 1,
                'token_id': all_ids[absolute]}
    require(all(type(event[key]) is int and event[key] == value
                for key, value in expected.items()), 'incorrect token context/history')
    return all_ids[absolute - length:absolute]


def token_bytes(document):
    # Reverse GPT-2's byte-to-Unicode alphabet, not Tokenizer.decode (which may
    # replace invalid UTF-8 or skip special tokens). Arbitrary byte pieces stay exact.
    direct = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    reverse = {chr(byte): byte for byte in direct}
    offset = 0
    for byte in range(256):
        if byte not in direct:
            reverse[chr(256 + offset)] = byte
            offset += 1
    result = {}
    for spelling, index in document['model']['vocab'].items():
        require(type(index) is int and index not in result, 'duplicate token ID')
        result[index] = bytes(reverse[character] for character in spelling)
    require(set(result) == set(range(50257)), 'wrong logical GPT-2 vocabulary')
    return result


def audit(root, output):
    root = Path(root).resolve()
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink(), 'audit output exists')
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    paths = [root / name for name in ('production_plan.json', 'production_result.json',
                                     'recorded_128_plan.json', 'recorded_128_result.json')]
    records = [identity(path) for path in paths]
    production_plan, production_result, plan, result = map(read_json, paths)
    require(production_result['returncode'] == result['returncode'] == 0 and
            production_result['all_inputs_byte_unchanged'] is True and
            result['all_inputs_byte_unchanged'] is True, 'native runs incomplete')
    authenticated = production_plan['inputs'] + plan['inputs'] + result['outputs'] + [production_result['output']]
    for record in authenticated:
        require(identity(record['path']) == record, 'initial input/output hash mismatch')
    native = root / 'recorded_128'
    metadata = read_json(native / 'metadata.json')
    require(metadata['complete'] is True and metadata['autoregressive'] is True,
            'incomplete/non-autoregressive metadata')
    require(metadata['prompt'] == 'to be or not to be' and metadata['seed'] == 17
            and metadata['rng_seed'] == 18 and metadata['temperature'] == .8
            and metadata['steps'] == 128 and metadata['context_length'] == 1024
            and metadata['vocab_size'] == 50257, 'different frozen generation parameters')
    initial = metadata['initial_token_ids']
    generated = metadata['generated_token_ids']
    all_ids = metadata['all_token_ids']
    require(initial == [1462, 307, 393, 407, 284, 307] and
            metadata['prompt_token_ids'] == initial and len(generated) == 128 and
            all_ids == initial + generated and metadata['token_ids'] == all_ids,
            'inconsistent complete token history')
    require(all(type(index) is int and 0 <= index < 50257 for index in all_ids),
            'invalid logical token ID')
    require(metadata['files']['logits'] == {'file': 'logits.f32', 'dtype': 'float32', 'shape': [128, 50257]},
            'wrong logical logit matrix descriptor')
    logits = np.fromfile(native / 'logits.f32', dtype='<f4').reshape(128, 50257)
    require(np.isfinite(logits).all(), 'nonfinite stored logits')
    require(np.array_equal(np.fromfile(native / 'tokens.i32', dtype='<i4'), all_ids),
            'binary token history differs')
    events = []
    for line in (native / 'events.jsonl').read_text().splitlines():
        # Each event is independently authenticated by the JSONL file SHA.
        events.append(json.loads(line))
    require(len(events) == 128, 'incorrect number of sampling events')
    tokenizer_path = Path(metadata['tokenizer_directory']) / 'tokenizer.json'
    dictionary = token_bytes(read_json(tokenizer_path))
    require(b''.join(dictionary[index] for index in initial) == metadata['prompt'].encode(),
            'initial exact token bytes do not match prompt')
    generated_raw = (native / 'generated.bin').read_bytes()
    require(generated_raw == b''.join(dictionary[index] for index in generated),
            'generated bytes differ from exact ID pieces')
    production_stdout = Path(production_result['output']['path']).read_bytes()
    checkpoint = metadata['checkpoint_directory']
    step = int(Path(checkpoint).name.removeprefix('step_'))
    prefix = f'loaded checkpoint: {checkpoint} (step {step})\n'.encode() + metadata['prompt'].encode()
    require(production_stdout.startswith(prefix) and production_stdout.endswith(b'\n'),
            'unexpected production stdout framing')
    production_generated = production_stdout[len(prefix):-1]
    require(production_generated.startswith(generated_raw), 'recorder differs from production prefix')
    random = Mt19937(18)
    require(random.cpp_state() == metadata['initial_rng_state'], 'initial full MT state differs')
    decisions = []
    byte_offset = 0
    for index, (event, row) in enumerate(zip(events, logits)):
        history = validate_context(event, len(initial), index, all_ids, 1024)
        chosen = event['token_id']
        piece = dictionary[chosen]
        require(event['piece_hex'] == piece.hex() and
                event['generated_byte_start'] == byte_offset and
                event['generated_byte_end'] == byte_offset + len(piece), 'piece/byte offsets differ')
        byte_offset += len(piece)
        require(event['logits_byte_offset'] == index * 50257 * 4 and event['temperature'] == .8,
                'event logit offset or temperature differs')
        low, high = random.draw(), random.draw()
        require((event['rng_word_0'], event['rng_word_1']) == (low, high), 'MT word mismatch')
        uniform = canonical_uniform(low, high)
        require(uniform == event['uniform'], 'canonical uniform mismatch')
        calculated = scalar_sample(row, .8, uniform)
        for field in ('token_id', 'sampled_probability', 'cdf_lower', 'cdf_upper'):
            require(calculated[field] == event[field], f'scalar sampling {field} differs at {index}')
        argmax = int(np.argmax(row))
        rank = 1 + int(np.count_nonzero(row > row[chosen])) + int(np.count_nonzero(row[:chosen] == row[chosen]))
        require(float(row[chosen]) == event['raw_logit'] and rank == event['raw_rank']
                and argmax == event['raw_argmax_token_id'], 'raw logit/rank/argmax differs')
        decisions.append({**event, 'independent_weight_sum': calculated['unnormalized_weight_sum'],
                          'uniform_minus_lower': uniform - calculated['cdf_lower'],
                          'upper_minus_uniform': calculated['cdf_upper'] - uniform,
                          'raw_argmax_piece_hex': dictionary[argmax].hex(),
                          'context_token_ids': history})
    require(random.cpp_state() == metadata['final_rng_state'], 'final full MT state differs')
    require(byte_offset == len(generated_raw), 'final byte cursor differs')
    word = b'shagemper'
    start = generated_raw.find(word)
    require(start >= 0, 'preselected production candidate not in recorded prefix')
    end = start + len(word)
    require((start == 0 or not bytes([generated_raw[start - 1]]).isalpha()) and
            end < len(generated_raw) and not bytes([generated_raw[end]]).isalpha(),
            'candidate lacks complete word boundaries')
    constituent = [event for event in decisions if event['generated_byte_start'] < end
                   and event['generated_byte_end'] > start]
    source_record = identity(__file__)
    records += [source_record]
    for record in authenticated + records:
        require(identity(record['path']) == record, 'input/output/source changed during audit')
    report = {'schema_version': 1, 'complete': True, 'passed': True,
              'started_utc': started,
              'completed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'source': source_record, 'python': platform.python_version(), 'numpy': np.__version__,
              'plans_and_results': records[:-1], 'authenticated_files': authenticated,
              'all_authenticated_files_unchanged': True,
              'checks': {'all128_fp32_subtract_double_exp_sum_cdf_exact': True,
                         'all128_selected_probabilities_and_intervals_exact': True,
                         'all256_MT_words_and_both_complete_states_exact': True,
                         'all128_uniforms_exact': True, 'all_context_windows_and_IDs_exact': True,
                         'all_piece_bytes_offsets_and_raw_ranks_exact': True,
                         'recorded_bytes_exact_prefix_of_production512': True},
              'generated_byte_count': len(generated_raw),
              'production_generated_byte_count': len(production_generated),
              'sampling_formula': 'exp(float64(float32(logit-max_float32))/temperature); scalar sequential float64 sum and normalization; sequential CDF with final entry1; bisect_left',
              'uniform_formula': '(uint32_word0 + uint32_word1*2^32)/2^64 in double, clamped below1 if rounded to1',
              'events': decisions,
              'candidate': {'word': word.decode(), 'generated_byte_start': start, 'generated_byte_end': end,
                            'constituent_events': constituent,
                            'whole_word_is_vocabulary_piece': word in dictionary.values(),
                            'space_prefixed_whole_word_is_vocabulary_piece': b' ' + word in dictionary.values()},
              'limits': ['No GPU/model forward, training, or corpus search was performed.',
                         'Recorder128 is proven byte-identical to a prefix, not the entire512-token production completion.',
                         'Candidate was supplied before this audit; the audit does not choose a favorable word.',
                         'CDF agreement explains sampling from recorded logits, not why the neural network produced those logits.']}
    with output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({'passed': True, 'artifact': identity(output),
                      'candidate_events': [{key: event[key] for key in ('index', 'token_id', 'piece_hex', 'raw_rank', 'raw_logit', 'sampled_probability', 'uniform', 'cdf_lower', 'cdf_upper')}
                                           for event in constituent]}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.root, args.output)


if __name__ == '__main__':
    main()
